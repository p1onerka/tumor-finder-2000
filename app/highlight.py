# python -m app.highlight dt/Test/pituitary/1322.jpg


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NOTUMOR_NAMES = {"notumor", "no_tumor", "no-tumor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Slice to classify and highlight.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/highlighted.png"))
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_tumor_class(name: str) -> bool:
    return name.lower() not in NOTUMOR_NAMES


def percentile_normalize(image: Image.Image) -> Image.Image:
    """Same 1st–99th percentile stretch as train.py."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(gray, (1, 99))
    if high <= low:
        scaled = np.zeros(gray.shape, dtype=np.uint8)
    else:
        scaled = np.clip((gray - low) / (high - low), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(checkpoint: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    if not checkpoint.is_file():
        raise SystemExit(f"Нет чекпоинта: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = list(saved["classes"])
    image_size = int(saved["image_size"])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    model.eval()
    return model, classes, image_size


def tumor_activation_map(model: nn.Module, batch: torch.Tensor, class_index: int) -> np.ndarray:
    """Grad-CAM on ResNet layer4 for one image. Values are in [0, 1]."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        activations.append(output)

    def save_gradient(_module: nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
        gradients.append(grad_output[0])

    forward_hook = model.layer4.register_forward_hook(save_activation)
    backward_hook = model.layer4.register_full_backward_hook(save_gradient)
    try:
        model.zero_grad(set_to_none=True)
        score = model(batch)[0, class_index]
        score.backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1))[0]
        cam = cam.detach()
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()
    finally:
        forward_hook.remove()
        backward_hook.remove()


def upsample_map(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    heat = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    heat = heat.resize(size, Image.BILINEAR)
    return np.asarray(heat, dtype=np.float32) / 255.0


def _bilateral_filter(values: np.ndarray, guide: np.ndarray, sigma_spatial: float, sigma_color: float) -> np.ndarray:
    """Normalized bilateral filter via a blurred bilateral grid."""
    height, width = values.shape
    step = max(float(sigma_spatial), 1.0)
    color_step = max(float(sigma_color), 1e-3)
    grid_y = int(np.ceil(height / step)) + 3
    grid_x = int(np.ceil(width / step)) + 3
    grid_z = int(np.ceil(1.0 / color_step)) + 3

    ys = np.clip(np.arange(height, dtype=np.float32) / step + 1.0, 0, grid_y - 1.001)
    xs = np.clip(np.arange(width, dtype=np.float32) / step + 1.0, 0, grid_x - 1.001)
    yy = np.broadcast_to(ys[:, None], (height, width))
    xx = np.broadcast_to(xs[None, :], (height, width))
    zz = np.clip(guide / color_step + 1.0, 0, grid_z - 1.001)

    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    z0 = np.floor(zz).astype(np.int32)
    wy = (yy - y0).astype(np.float32)
    wx = (xx - x0).astype(np.float32)
    wz = (zz - z0).astype(np.float32)

    grid_sum = np.zeros((grid_y, grid_x, grid_z), dtype=np.float32)
    grid_weight = np.zeros((grid_y, grid_x, grid_z), dtype=np.float32)

    def splat(iy: np.ndarray, ix: np.ndarray, iz: np.ndarray, weight: np.ndarray) -> None:
        np.add.at(grid_sum, (iy, ix, iz), values * weight)
        np.add.at(grid_weight, (iy, ix, iz), weight)

    y1 = y0 + 1
    x1 = x0 + 1
    z1 = z0 + 1
    splat(y0, x0, z0, (1 - wy) * (1 - wx) * (1 - wz))
    splat(y0, x0, z1, (1 - wy) * (1 - wx) * wz)
    splat(y0, x1, z0, (1 - wy) * wx * (1 - wz))
    splat(y0, x1, z1, (1 - wy) * wx * wz)
    splat(y1, x0, z0, wy * (1 - wx) * (1 - wz))
    splat(y1, x0, z1, wy * (1 - wx) * wz)
    splat(y1, x1, z0, wy * wx * (1 - wz))
    splat(y1, x1, z1, wy * wx * wz)

    # One grid cell is one sigma, so a unit Gaussian is the appearance kernel.
    grid_sum = gaussian_filter(grid_sum, sigma=1.0, mode="nearest")
    grid_weight = gaussian_filter(grid_weight, sigma=1.0, mode="nearest")

    def slice_grid(grid: np.ndarray) -> np.ndarray:
        c000 = grid[y0, x0, z0]
        c001 = grid[y0, x0, z1]
        c010 = grid[y0, x1, z0]
        c011 = grid[y0, x1, z1]
        c100 = grid[y1, x0, z0]
        c101 = grid[y1, x0, z1]
        c110 = grid[y1, x1, z0]
        c111 = grid[y1, x1, z1]
        c00 = c000 * (1 - wz) + c001 * wz
        c01 = c010 * (1 - wz) + c011 * wz
        c10 = c100 * (1 - wz) + c101 * wz
        c11 = c110 * (1 - wz) + c111 * wz
        c0 = c00 * (1 - wx) + c01 * wx
        c1 = c10 * (1 - wx) + c11 * wx
        return c0 * (1 - wy) + c1 * wy

    return slice_grid(grid_sum) / np.maximum(slice_grid(grid_weight), 1e-6)


def refine_with_crf(heatmap: np.ndarray, image: Image.Image) -> np.ndarray:
    """Snap a coarse CAM to tissue boundaries.

    Mean-field inference in a fully connected CRF: a bilateral kernel lets the
    label move through similar intensities, and a short spatial kernel removes
    speckle. The CAM unary stays in charge far from the activated region.
    """
    guide = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    cam = upsample_map(heatmap, image.size)
    # Only the peak is a sure tumor. The shoulder is uncertain, so the pairwise
    # terms can follow an intensity edge instead of the smooth CAM oval.
    probability = 1.0 / (1.0 + np.exp(-(cam - 0.72) / 0.07))
    probability = np.clip(probability, 1e-3, 1.0 - 1e-3)
    unary_fg = -np.log(probability)
    unary_bg = -np.log(1.0 - probability)

    foreground = probability.astype(np.float32)
    bilateral_weight = 5.0
    spatial_weight = 2.0
    for _ in range(5):
        background = 1.0 - foreground
        fg_bilateral = _bilateral_filter(foreground, guide, sigma_spatial=24, sigma_color=0.08)
        bg_bilateral = _bilateral_filter(background, guide, sigma_spatial=24, sigma_color=0.08)
        fg_spatial = gaussian_filter(foreground, sigma=3, mode="nearest")
        bg_spatial = gaussian_filter(background, sigma=3, mode="nearest")
        energy_fg = unary_fg + bilateral_weight * bg_bilateral + spatial_weight * bg_spatial
        energy_bg = unary_bg + bilateral_weight * fg_bilateral + spatial_weight * fg_spatial
        shift = np.minimum(energy_fg, energy_bg)
        fg_score = np.exp(-(energy_fg - shift))
        bg_score = np.exp(-(energy_bg - shift))
        foreground = (fg_score / (fg_score + bg_score)).astype(np.float32)

    # Drop leaks outside the classifier's attention. The boundary itself is
    # still free to sit on a tissue edge inside that neighborhood.
    gate = np.clip((cam - 0.12) / 0.15, 0, 1)
    return foreground * gate


def overlay_tumor(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """Tint the strong part of a [0, 1] heatmap red-yellow on a copy of image."""
    heat = np.asarray(
        Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    core = heat >= 0.45
    if not core.any():
        return image.copy()
    strength = np.clip((heat - 0.45) / 0.55, 0, 1) * core
    alpha = (0.35 + 0.45 * strength) * core
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.empty_like(base)
    color[..., 0] = 255
    color[..., 1] = 30 + 200 * strength
    color[..., 2] = 0
    blended = base * (1 - alpha[..., None]) + color * alpha[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def classify_image(
    image_path: Path,
    checkpoint: Path = Path("runs/best.pt"),
) -> tuple[Image.Image, str]:
    """Classify image_path with the checkpoint and return the slice plus the class.

    When the predicted class is a tumor, the slice is tinted on the region the
    model uses for that class, pulled onto the slice's tissue boundaries.
    notumor comes back without a tint.
    """
    if not image_path.is_file():
        raise SystemExit(f"Нет файла: {image_path}")

    device = pick_device()
    model, classes, image_size = load_model(checkpoint, device)
    with Image.open(image_path) as image:
        display = percentile_normalize(image)

    batch = eval_transform(image_size)(display).unsqueeze(0).to(device)
    with torch.no_grad():
        class_index = int(model(batch).argmax(dim=1).item())
    class_name = classes[class_index]
    if not is_tumor_class(class_name):
        return display, class_name

    heatmap = tumor_activation_map(model, batch, class_index)
    refined = refine_with_crf(heatmap, display)
    return overlay_tumor(display, refined), class_name


def main() -> None:
    args = parse_args()
    image, class_name = classify_image(args.image, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Класс: {class_name}")
    if is_tumor_class(class_name):
        print(f"Подсвеченный снимок: {args.output}")
    else:
        print(f"Опухоли нет, снимок без подсветки: {args.output}")


if __name__ == "__main__":
    main()
