# python -m app.train
# python -m app.train --epochs 5 --batch-size 32 --hide-prob 0.5 --checkpoint runs/wsol.pt

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NOTUMOR_NAMES = {"notumor", "no_tumor", "no-tumor"}

HIDE_GRIDS = (0, 4, 8, 16)
HEAD_LEVEL = 0.08
# Доля пика мелкой карты LayerCAM, из которой собирается рамка.
CAM_THRESHOLD = 0.5
CLASSIFIER_CHECKPOINT = Path("runs/best_classifier.pt")
LOCALIZER_CHECKPOINT = Path("runs/best_highlighter.pt")
TRAIN_DIR = Path("dt/Train")
TEST_DIR = Path("dt/Test")
VAL_FRACTION = 0.15
IMAGE_SIZE = 224
SEED = 42
WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--hide-prob",
        type=float,
        default=0.5,
        help="turns hide and seek on",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/wsol.pt"))
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_tumor_class(name: str) -> bool:
    return name.lower() not in NOTUMOR_NAMES


def class_names_from(train_dir: Path) -> list[str]:
    names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not names:
        raise SystemExit(f"В {train_dir} нет директорий классов")
    return names


def collect_samples(
    root: Path, classes: list[str], limit: int, seed: int
) -> list[tuple[Path, int]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for label, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit(f"Нет директории класса: {folder}")
        files = [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not files:
            raise SystemExit(f"Нет снимков в {folder}")
        grouped[label] = files

    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []
    for label, files in grouped.items():
        files = sorted(files)
        if limit > 0 and len(files) > limit:
            files = rng.sample(files, limit)
        samples.extend((path, label) for path in files)
    rng.shuffle(samples)
    return samples


def percentile_normalize(path: Path) -> Image.Image:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(gray, (1, 99))
    if high <= low:
        scaled = np.zeros(gray.shape, dtype=np.uint8)
    else:
        scaled = np.clip((gray - low) / (high - low), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


class HideAndSeek:
    def __init__(self, grids: tuple[int, ...], drop_prob: float) -> None:
        self.grids = grids
        self.drop_prob = drop_prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        grid = random.choice(self.grids)
        if grid <= 1 or self.drop_prob <= 0:
            return tensor
        _, height, width = tensor.shape
        cell_h = int(np.ceil(height / grid))
        cell_w = int(np.ceil(width / grid))
        drop = np.zeros((grid, grid), dtype=bool)
        for row in range(grid):
            for col in range(grid):
                drop[row, col] = random.random() < self.drop_prob
        if bool(drop.all()):
            drop[random.randrange(grid), random.randrange(grid)] = False
        if not bool(drop.any()):
            return tensor
        hidden = tensor.clone()
        for row in range(grid):
            for col in range(grid):
                if not drop[row, col]:
                    continue
                y0 = row * cell_h
                x0 = col * cell_w
                hidden[:, y0: y0 + cell_h, x0: x0 + cell_w] = 0
        return hidden


def build_transform(
    image_size: int, train: bool, hide_grids: tuple[int, ...], hide_prob: float
) -> transforms.Compose:
    steps: list = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if train and hide_prob > 0:
        steps.append(HideAndSeek(hide_grids, hide_prob))
    return transforms.Compose(steps)


class SliceDataset(Dataset):
    def __init__(
        self, samples: list[tuple[Path, int]], transform: transforms.Compose
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return self.transform(percentile_normalize(path)), label


class CamNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.fc = nn.Linear(backbone.fc.in_features, n_classes)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.features(images)
        return self.fc(maps.mean(dim=(2, 3)))

    def class_activation(
        self, maps: torch.Tensor, class_index: torch.Tensor
    ) -> torch.Tensor:
        weight = self.fc.weight.index_select(0, class_index)
        return torch.relu((maps * weight[:, :, None, None]).sum(dim=1))


def remap_torchvision(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("conv1."):
            mapped["stem.0." + key[len("conv1."):]] = value
        elif key.startswith("bn1."):
            mapped["stem.1." + key[len("bn1."):]] = value
        else:
            mapped[key] = value
    return mapped


def load_matching(model: nn.Module, state: dict[str, torch.Tensor]) -> set[str]:
    if "conv1.weight" in state and "stem.0.weight" not in state:
        state = remap_torchvision(state)
    own = model.state_dict()
    loaded: set[str] = set()
    for key, value in state.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            own[key] = value
            loaded.add(key)
    model.load_state_dict(own)
    return loaded


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("fc.")


def apply_train_mode(
    model: nn.Module, training: bool, update_backbone_bn: bool
) -> None:
    model.train(training)
    if training and not update_backbone_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    update_backbone_bn: bool,
) -> tuple[float, float, float]:
    training = optimizer is not None
    apply_train_mode(model, training, update_backbone_bn)
    total_loss = 0.0
    correct = 0
    seen = 0
    true_labels: list[np.ndarray] = []
    pred_labels: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch = labels.size(0)
        total_loss += loss.item() * batch
        predicted = logits.argmax(dim=1)
        correct += int((predicted == labels).sum().item())
        seen += batch
        true_labels.append(labels.detach().cpu().numpy())
        pred_labels.append(predicted.detach().cpu().numpy())

    y_true = np.concatenate(true_labels)
    y_pred = np.concatenate(pred_labels)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return total_loss / seen, correct / seen, macro_f1


def load_model(
    checkpoint: Path, device: torch.device
) -> tuple[CamNet, list[str], int, transforms.Compose]:
    if not checkpoint.is_file():
        raise SystemExit(f"Не найдено классификатора {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        classes = list(saved["classes"])
        image_size = int(saved["image_size"])
        state = saved["state_dict"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"Файл {checkpoint} некорректен") from exc

    model = CamNet(len(classes))
    loaded = load_matching(model, state)
    if "fc.weight" not in loaded:
        raise SystemExit(
            f"Классификатор {checkpoint} не содержит необходимой информации"
        )
    model.to(device)
    model.eval()
    transform = build_transform(image_size, False, HIDE_GRIDS, 0.0)
    return model, classes, image_size, transform


def head_mask(gray: np.ndarray) -> np.ndarray:
    head = gray > HEAD_LEVEL
    head = ndi.binary_opening(head, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndi.label(head)
    if count == 0:
        return np.ones(gray.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    head = labels == int(sizes.argmax())
    return ndi.binary_fill_holes(head)


def _upsample_unit(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    """Карта в [0, 1] размером снимка. Постоянная карта даёт нули."""
    cam = np.asarray(cam, dtype=np.float32)
    span = float(cam.max() - cam.min())
    if span < 1e-8:
        return np.zeros((height, width), dtype=np.float32)
    unit = (cam - float(cam.min())) / span
    tensor = torch.from_numpy(unit)[None, None]
    return F.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()


def _layer_cam(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
    """LayerCAM: у каждой точки карты свой положительный вес градиента."""
    weighted = torch.relu(gradient) * activation
    return torch.relu(weighted.sum(dim=0)).detach().cpu().numpy()


def class_layer_cams(
    model: CamNet, images: torch.Tensor, class_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """LayerCAM layer3 и layer4 для одного снимка.

    layer4 — та же грубая сетка 7×7, что и обычная карта класса.
    layer3 вдвое плотнее и держит край небольшой опухоли.
    """
    saved: dict[str, torch.Tensor] = {}
    handles = []
    layers = (("fine", model.layer3), ("coarse", model.layer4))
    for name, layer in layers:

        def remember(module, inputs, output, name=name):
            output.retain_grad()
            saved[name] = output

        def remember_grad(module, grad_in, grad_out, name=name):
            saved[f"{name}_grad"] = grad_out[0]

        handles.append(layer.register_forward_hook(remember))
        handles.append(layer.register_full_backward_hook(remember_grad))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            score = model(images)[0, class_index]
            score.backward()
        fine_grad = saved.get("fine_grad")
        coarse_grad = saved.get("coarse_grad")
        if fine_grad is None or coarse_grad is None:
            raise RuntimeError("нет градиента карты класса")
        fine = _layer_cam(saved["fine"][0], fine_grad[0])
        coarse = _layer_cam(saved["coarse"][0], coarse_grad[0])
    finally:
        for handle in handles:
            handle.remove()
        saved.clear()
        model.zero_grad(set_to_none=True)
    return fine, coarse


def _component(
    heatmap: np.ndarray,
    head: np.ndarray,
    level: float,
    row: int,
    col: int,
) -> np.ndarray | None:
    region = (heatmap >= level) & head
    labels, count = ndi.label(region)
    if count == 0:
        return None
    label = int(labels[row, col])
    if label == 0:
        return None
    return ndi.binary_fill_holes(labels == label)


def activation_mask(
    fine: np.ndarray,
    coarse: np.ndarray,
    width: int,
    height: int,
    head: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, float]:
    """Область layer3 вокруг пика, который подтверждает и layer4.

    Пик произведения отбрасывает яркий, но чужой очаг: глаз или сосуд,
    на котором мелкая карта вспыхнула, а карта класса — нет.
    """
    empty = np.zeros((height, width), dtype=bool)
    fine_up = _upsample_unit(fine, height, width)
    coarse_up = _upsample_unit(coarse, height, width)
    if float(fine_up.max()) < 1e-8:
        return empty, None, 0.0
    score = fine_up * coarse_up
    seeded = np.where(head, score, -1.0)
    peak_row, peak_col = np.unravel_index(int(np.argmax(seeded)), seeded.shape)
    mask = None
    for level in (threshold, 0.4, 0.3):
        mask = _component(fine_up, head, level, int(peak_row), int(peak_col))
        if mask is not None:
            break
    if mask is None:
        mask = _component(
            _upsample_unit(score, height, width),
            head,
            threshold,
            int(peak_row),
            int(peak_col),
        )
    if mask is None or not mask.any():
        return empty, None, 0.0
    rows, cols = np.nonzero(mask)
    box = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    fraction = float(mask.sum()) / float(max(int(head.sum()), 1))
    return mask, box, fraction


def overlay(
    image: Image.Image, mask: np.ndarray, box: tuple[int, int, int, int] | None
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mask.any():
        soft = np.clip(ndi.gaussian_filter(mask.astype(np.float32), 0.6), 0, 1)
        alpha = 0.55 * soft
        color = np.empty_like(base)
        color[..., 0] = 255
        color[..., 1] = 40 + 180 * soft
        color[..., 2] = 0
        base = base * (1.0 - alpha[..., None]) + color * alpha[..., None]
    painted = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if box is not None:
        draw = ImageDraw.Draw(painted)
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(0, 220, 255), width=3)
    return painted


def predict_class(
    model: CamNet,
    classes: list[str],
    transform: transforms.Compose,
    display: Image.Image,
    device: torch.device,
) -> tuple[str, float]:
    batch = transform(display).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch)[0], dim=0)
        class_index = int(probabilities.argmax().item())
    return classes[class_index], float(probabilities[class_index].item())


def highlight_class(
    model: CamNet,
    classes: list[str],
    transform: transforms.Compose,
    display: Image.Image,
    device: torch.device,
    class_name: str,
    threshold: float,
) -> Image.Image:
    if not is_tumor_class(class_name):
        return display.copy()
    try:
        class_index = classes.index(class_name)
    except ValueError as exc:
        known = ", ".join(classes)
        raise SystemExit(
            f"В модели подсветки нет класса {class_name}. Есть: {known}."
        ) from exc
    batch = transform(display).unsqueeze(0).to(device)
    fine, coarse = class_layer_cams(model, batch, class_index)
    gray = np.asarray(display.convert("L"), dtype=np.float32) / 255.0
    width, height = display.size
    mask, box, _fraction = activation_mask(
        fine, coarse, width, height, head_mask(gray), threshold
    )
    return overlay(display, mask, box)


def classify_image(
    image_path: Path,
    classifier_checkpoint: Path = CLASSIFIER_CHECKPOINT,
    localizer_checkpoint: Path = LOCALIZER_CHECKPOINT,
) -> tuple[Image.Image, str]:
    if not image_path.is_file():
        raise SystemExit(f"Нет файла: {image_path}")
    device = pick_device()
    display = percentile_normalize(image_path)
    classifier, class_names, _image_size, transform = load_model(
        classifier_checkpoint, device
    )
    class_name, _probability = predict_class(
        classifier, class_names, transform, display, device
    )
    if not is_tumor_class(class_name):
        return display, class_name
    localizer, localizer_classes, _image_size, localizer_transform = load_model(
        localizer_checkpoint, device
    )
    highlighted = highlight_class(
        localizer,
        localizer_classes,
        localizer_transform,
        display,
        device,
        class_name,
        CAM_THRESHOLD,
    )
    return highlighted, class_name


def assert_train_only(samples: list[tuple[Path, int]], train_dir: Path) -> None:
    root = train_dir.resolve()
    if any(part.lower() == "test" for part in root.parts):
        raise SystemExit(f"Обучение только из каталога Train, не из {train_dir}")
    for path, _label in samples:
        resolved = path.resolve()
        if any(part.lower() == "test" for part in resolved.parts):
            raise SystemExit(f"Для обучения нельзя брать снимки из Test: {resolved}")
        if root not in resolved.parents:
            raise SystemExit(f"Снимок для обучения не лежит в {root}: {resolved}")


class TrainMetrics(NamedTuple):
    best_val_f1: float
    best_val_acc: float
    best_val_loss: float
    best_train_f1: float
    best_train_acc: float
    best_train_loss: float
    test_macro_f1: float
    test_accuracy: float


def train_model(args: argparse.Namespace) -> TrainMetrics:
    if args.epochs < 1:
        raise SystemExit("--epochs должен быть >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size должен быть >= 1")
    if not 0 <= args.hide_prob <= 1:
        raise SystemExit("--hide-prob должен быть от 0 до 1")

    set_seed(SEED)
    device = pick_device()
    classes = class_names_from(TRAIN_DIR)
    train_samples = collect_samples(TRAIN_DIR, classes, 0, SEED)
    assert_train_only(train_samples, TRAIN_DIR)
    test_samples = collect_samples(TEST_DIR, classes, 0, SEED + 1)
    train_paths = {path.resolve() for path, _label in train_samples}
    leaked = [path for path, _label in test_samples if path.resolve() in train_paths]
    if leaked:
        raise SystemExit(f"Снимок Test попал в обучение: {leaked[0]}")

    labels = [label for _, label in train_samples]
    indexes = np.arange(len(train_samples))
    train_idx, val_idx = train_test_split(
        indexes,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=labels,
    )
    assert_train_only([train_samples[i] for i in train_idx], TRAIN_DIR)

    train_ds = SliceDataset(
        train_samples, build_transform(IMAGE_SIZE, True, HIDE_GRIDS, args.hide_prob)
    )
    eval_transform = build_transform(IMAGE_SIZE, False, HIDE_GRIDS, 0.0)
    val_ds = SliceDataset(train_samples, eval_transform)
    test_ds = SliceDataset(test_samples, eval_transform)

    train_loader = make_loader(
        Subset(train_ds, train_idx.tolist()), args.batch_size, True, WORKERS, device
    )
    val_loader = make_loader(
        Subset(val_ds, val_idx.tolist()), args.batch_size, False, WORKERS, device
    )
    test_loader = make_loader(test_ds, args.batch_size, False, WORKERS, device)

    model = CamNet(len(classes)).to(device)
    pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    load_matching(model, pretrained.state_dict())
    weights = class_weights([labels[i] for i in train_idx], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = 0.0
    best_train_f1 = 0.0
    best_train_acc = 0.0
    best_train_loss = 0.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        head_only = epoch == 1
        set_backbone_trainable(model, trainable=not head_only)
        if head_only:
            optimizer = torch.optim.AdamW(
                model.fc.parameters(), lr=1e-3, weight_decay=1e-4
            )
        else:
            backbone = [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("fc.")
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone, "lr": 1e-4},
                    {"params": model.fc.parameters(), "lr": 1e-3},
                ],
                weight_decay=1e-4,
            )

        train_loss, train_acc, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            update_backbone_bn=not head_only,
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, None, device, update_backbone_bn=False
        )
        print(
            f"Эпоха {epoch}/{args.epochs}  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} macro-F1 {train_f1:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} macro-F1 {val_f1:.3f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_f1 = train_f1
            best_train_acc = train_acc
            best_train_loss = train_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                },
                args.checkpoint,
            )

    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_matching(model, saved["state_dict"])
    model.eval()
    true_labels: list[np.ndarray] = []
    pred_labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu().numpy()
            pred_labels.append(predicted)
            true_labels.append(batch_labels.numpy())
    y_true = np.concatenate(true_labels)
    y_pred = np.concatenate(pred_labels)
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_accuracy = float((y_true == y_pred).mean())
    print(
        f"Чекпоинт {args.checkpoint}  "
        f"val macro-F1 {best_f1:.3f} acc {best_val_acc:.3f}  "
        f"test macro-F1 {test_macro_f1:.3f} acc {test_accuracy:.3f}"
    )
    return TrainMetrics(
        best_val_f1=float(best_f1),
        best_val_acc=float(best_val_acc),
        best_val_loss=float(best_val_loss),
        best_train_f1=float(best_train_f1),
        best_train_acc=float(best_train_acc),
        best_train_loss=float(best_train_loss),
        test_macro_f1=test_macro_f1,
        test_accuracy=test_accuracy,
    )


def main() -> None:
    train_model(parse_args())


if __name__ == "__main__":
    main()
