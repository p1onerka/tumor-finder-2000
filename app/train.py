# python -m app.train --epochs 1 --limit-per-class 64

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("dt/Train"))
    parser.add_argument("--test-dir", type=Path, default=Path("dt/Test"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Use at most this many images from each class in each split. 0 uses all.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/best.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_names_from(train_dir: Path) -> list[str]:
    names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not names:
        raise SystemExit(f"No class folders in {train_dir}")
    return names


def collect_samples(root: Path, classes: list[str], limit: int, seed: int) -> list[tuple[Path, int]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for label, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder: {folder}")
        files = [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not files:
            raise SystemExit(f"No images in {folder}")
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


class SliceDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return self.transform(percentile_normalize(path)), label


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    steps: list = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


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


def build_model(n_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def apply_train_mode(model: nn.Module, training: bool, update_backbone_bn: bool) -> None:
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
        correct += (predicted == labels).sum().item()
        seen += batch
        true_labels.append(labels.detach().cpu().numpy())
        pred_labels.append(predicted.detach().cpu().numpy())

    y_true = np.concatenate(true_labels)
    y_pred = np.concatenate(pred_labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return total_loss / seen, correct / seen, float(macro_f1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_labels: list[np.ndarray] = []
    pred_labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            pred_labels.append(logits.argmax(dim=1).cpu().numpy())
            true_labels.append(labels.numpy())
    return np.concatenate(true_labels), np.concatenate(pred_labels)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("fc.")


def print_split_counts(title: str, labels: list[int] | np.ndarray, classes: list[str]) -> None:
    counts = np.bincount(labels, minlength=len(classes))
    parts = [f"{name}={count}" for name, count in zip(classes, counts)]
    print(f"{title}: {int(counts.sum())} ({', '.join(parts)})")


def format_confusion(matrix: np.ndarray, classes: list[str]) -> str:
    labels = [name[:12] for name in classes]
    width = max(12, *(len(label) for label in labels))
    header = " " * (width + 2) + " ".join(label.rjust(width) for label in labels)
    rows = [header]
    for name, row in zip(labels, matrix):
        cells = " ".join(f"{int(value):{width}d}" for value in row)
        rows.append(f"{name:<{width}}  {cells}")
    return "\n".join(rows)


def print_next_step(
    train_loss: float,
    val_loss: float,
    report: dict,
    matrix: np.ndarray,
    classes: list[str],
    limited: bool,
) -> None:
    print("\nКуда смотреть дальше:")
    if limited:
        print(
            "- Сейчас взята только часть снимков. Эти цифры проверяют, что обучение идёт, "
            "и не являются итоговой оценкой. Следующий прогон — без --limit-per-class."
        )

    if train_loss + 0.15 < val_loss:
        print(
            f"- Train loss {train_loss:.3f} заметно ниже val loss {val_loss:.3f}: "
            "сеть запоминает обучающие срезы. Имеет смысл остановиться на эпохе с лучшим "
            "val macro-F1 и не наращивать эпохи вслепую."
        )

    f1_by_class = {name: report[name]["f1-score"] for name in classes}
    worst = min(f1_by_class, key=f1_by_class.get)
    if f1_by_class[worst] < 0.75:
        print(
            f"- Слабее всего класс {worst} (F1 {f1_by_class[worst]:.3f}). "
            "Смотрите его строку в матрице: столбец с самой большой ошибкой — класс, с которым его путают."
        )

    confused = matrix.copy()
    np.fill_diagonal(confused, 0)
    if confused.sum() > 0:
        true_i, pred_i = np.unravel_index(int(confused.argmax()), confused.shape)
        share = confused[true_i, pred_i] / max(matrix[true_i].sum(), 1)
        if share >= 0.15:
            print(
                f"- Самая частая ошибка: {classes[true_i]} предсказан как {classes[pred_i]} "
                f"({share:.0%} этого класса). Если эти срезы ещё и в разных проекциях, "
                "разберите глазами несколько таких пар, прежде чем менять архитектуру."
            )

    if report["macro avg"]["f1-score"] >= 0.85 and f1_by_class[worst] >= 0.75:
        print(
            "- Классы в этом прогоне разделяются. Дальше сравнивайте чекпоинты по val macro-F1 "
            "и по той же матрице на Test, а не по одной accuracy."
        )


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs должен быть >= 1")
    if not 0 < args.val_fraction < 0.5:
        raise SystemExit("--val-fraction должен быть между 0 и 0.5")

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    classes = class_names_from(args.train_dir)
    train_samples = collect_samples(args.train_dir, classes, args.limit_per_class, args.seed)
    test_samples = collect_samples(args.test_dir, classes, args.limit_per_class, args.seed + 1)
    labels = [label for _, label in train_samples]
    indexes = np.arange(len(train_samples))
    train_idx, val_idx = train_test_split(
        indexes,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    train_ds = SliceDataset(train_samples, build_transform(args.image_size, train=True))
    eval_transform = build_transform(args.image_size, train=False)
    val_ds = SliceDataset(train_samples, eval_transform)
    test_ds = SliceDataset(test_samples, eval_transform)

    train_loader = make_loader(Subset(train_ds, train_idx.tolist()), args.batch_size, True, args.workers, device)
    val_loader = make_loader(Subset(val_ds, val_idx.tolist()), args.batch_size, False, args.workers, device)
    test_loader = make_loader(test_ds, args.batch_size, False, args.workers, device)

    print(f"Устройство: {device}")
    print(f"Классы: {', '.join(classes)}")
    print_split_counts("Train", [labels[i] for i in train_idx], classes)
    print_split_counts("Val", [labels[i] for i in val_idx], classes)
    print_split_counts("Test", [label for _, label in test_samples], classes)

    model = build_model(len(classes)).to(device)
    weights = class_weights([labels[i] for i in train_idx], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = -1.0
    best_train_loss = 0.0
    best_val_loss = 0.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        head_only = epoch == 1
        set_backbone_trainable(model, trainable=not head_only)
        if head_only:
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
            phase = "только последний слой"
        else:
            backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone, "lr": 1e-4},
                    {"params": model.fc.parameters(), "lr": 1e-3},
                ],
                weight_decay=1e-4,
            )
            phase = "вся сеть"

        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, update_backbone_bn=not head_only
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, None, device, update_backbone_bn=False
        )
        print(
            f"Эпоха {epoch}/{args.epochs} ({phase})  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} macro-F1 {val_f1:.3f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_train_loss = train_loss
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                },
                args.checkpoint,
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    y_true, y_pred = predict(model, test_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print(f"\nTest по лучшему val macro-F1 ({best_f1:.3f}), чекпоинт {args.checkpoint}")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))
    print("Матрица ошибок (строка — истинный класс, столбец — предсказание):")
    print(format_confusion(matrix, classes))
    print_next_step(
        best_train_loss,
        best_val_loss,
        report,
        matrix,
        classes,
        limited=args.limit_per_class > 0,
    )


if __name__ == "__main__":
    main()
