from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from app.train import (
    CamNet,
    HideAndSeek,
    build_transform,
    class_names_from,
    classify_image,
    collect_samples,
    highlight_class,
    is_tumor_class,
    load_model,
    make_loader,
    set_seed,
    train_model,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _cpu_loader(xs: torch.Tensor, ys: torch.Tensor, batch_size: int, shuffle: bool = False):
    return make_loader(TensorDataset(xs, ys), batch_size, shuffle, 0, torch.device("cpu"))


@pytest.mark.parametrize(
    ("name", "tumor"),
    [
        ("glioma", True),
        ("meningioma", True),
        ("pituitary", True),
        ("", True),
        ("notumor", False),
        ("NoTumor", False),
        ("no_tumor", False),
        ("NO_TUMOR", False),
        ("no-tumor", False),
    ],
)
def test_is_tumor_class(name: str, tumor: bool) -> None:
    assert is_tumor_class(name) is tumor


def test_class_names_from_reads_sorted_directories(tmp_path) -> None:
    (tmp_path / "pituitary").mkdir()
    (tmp_path / "glioma").mkdir()
    (tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")
    assert class_names_from(tmp_path) == ["glioma", "pituitary"]


def test_class_names_from_rejects_an_empty_directory(tmp_path) -> None:
    with pytest.raises(SystemExit, match="нет директорий"):
        class_names_from(tmp_path)


def test_collect_samples_keeps_image_suffixes_and_labels(tmp_path) -> None:
    for name in ("a.jpg", "b.PNG", "c.jpeg", "d.bmp", "e.tif", "f.tiff", "g.webp", "skip.gif", "notes.txt"):
        _touch(tmp_path / "glioma" / name)
    _touch(tmp_path / "notumor" / "n.jpg")

    samples = collect_samples(tmp_path, ["notumor", "glioma"], limit=0, seed=1)
    by_name = {path.name: label for path, label in samples}
    assert set(by_name) == {"a.jpg", "b.PNG", "c.jpeg", "d.bmp", "e.tif", "f.tiff", "g.webp", "n.jpg"}
    assert by_name["n.jpg"] == 0
    assert by_name["a.jpg"] == 1
    again = collect_samples(tmp_path, ["notumor", "glioma"], limit=0, seed=1)
    assert samples == again


def test_collect_samples_rejects_a_missing_or_empty_class(tmp_path) -> None:
    with pytest.raises(SystemExit, match="Нет директории класса"):
        collect_samples(tmp_path, ["glioma"], limit=0, seed=0)

    (tmp_path / "glioma").mkdir()
    with pytest.raises(SystemExit, match="Нет снимков"):
        collect_samples(tmp_path, ["glioma"], limit=0, seed=0)

    _touch(tmp_path / "glioma" / "notes.txt")
    with pytest.raises(SystemExit, match="Нет снимков"):
        collect_samples(tmp_path, ["glioma"], limit=0, seed=0)


def test_hide_and_seek_returns_the_same_tensor_when_hiding_is_off() -> None:
    tensor = torch.ones(3, 4, 4)
    assert HideAndSeek((0, 1), 1.0)(tensor) is tensor
    assert HideAndSeek((8,), 0.0)(tensor) is tensor


def test_hide_and_seek_zeroes_only_the_chosen_cells(monkeypatch) -> None:
    tensor = torch.ones(3, 4, 4)
    draws = iter((0.1, 0.9, 0.9, 0.1))
    monkeypatch.setattr(random, "choice", lambda grids: 2)
    monkeypatch.setattr(random, "random", lambda: next(draws))

    hidden = HideAndSeek((2,), 0.5)(tensor)

    assert torch.equal(tensor, torch.ones_like(tensor))
    assert torch.equal(hidden[:, 0:2, 0:2], torch.zeros(3, 2, 2))
    assert torch.equal(hidden[:, 0:2, 2:4], torch.ones(3, 2, 2))
    assert torch.equal(hidden[:, 2:4, 0:2], torch.ones(3, 2, 2))
    assert torch.equal(hidden[:, 2:4, 2:4], torch.zeros(3, 2, 2))


@pytest.mark.parametrize("class_name", ["notumor", "No_Tumor", "no-tumor"])
def test_highlight_class_returns_a_copy_for_a_clean_slice(class_name: str) -> None:
    image = Image.new("RGB", (4, 4), (5, 6, 7))
    highlighted = highlight_class(None, ["notumor"], None, image, torch.device("cpu"), class_name, 0.5)
    assert highlighted is not image
    assert highlighted.tobytes() == image.tobytes()
    highlighted.putpixel((0, 0), (0, 0, 0))
    assert image.getpixel((0, 0)) == (5, 6, 7)


def test_highlight_class_rejects_an_unknown_tumor_name() -> None:
    image = Image.new("RGB", (4, 4))
    with pytest.raises(SystemExit, match="нет класса glioma"):
        highlight_class(None, ["meningioma", "notumor"], None, image, torch.device("cpu"), "glioma", 0.5)


def test_highlight_class_returns_an_image_of_the_same_size() -> None:
    image = Image.new("RGB", (24, 16), (30, 30, 30))
    before = image.tobytes()
    highlighted = highlight_class(
        CamNet(2).eval(),
        ["glioma", "notumor"],
        build_transform(64, False, (0,), 0.0),
        image,
        torch.device("cpu"),
        "glioma",
        0.5,
    )
    assert highlighted.size == image.size
    assert highlighted.mode == "RGB"
    assert image.tobytes() == before


def test_load_model_rejects_a_missing_or_incomplete_checkpoint(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(SystemExit, match="Не найдено"):
        load_model(missing, torch.device("cpu"))

    broken = tmp_path / "broken.pt"
    torch.save([1, 2, 3], broken)
    with pytest.raises(SystemExit, match="некорректен"):
        load_model(broken, torch.device("cpu"))

    partial = tmp_path / "partial.pt"
    state = CamNet(1).state_dict()
    del state["fc.weight"]
    torch.save({"state_dict": state, "classes": ["glioma"], "image_size": 32}, partial)
    with pytest.raises(SystemExit, match="не содержит"):
        load_model(partial, torch.device("cpu"))


def test_classify_image_requires_an_existing_file(tmp_path) -> None:
    with pytest.raises(SystemExit, match="Нет файла"):
        classify_image(tmp_path / "missing.jpg")


@pytest.mark.parametrize(
    ("epochs", "batch_size", "hide_prob", "text"),
    [
        (0, 1, 0.5, "epochs"),
        (1, 0, 0.5, "batch-size"),
        (1, 1, 1.1, "hide-prob"),
        (1, 1, -0.1, "hide-prob"),
    ],
)
def test_train_model_rejects_invalid_arguments(epochs, batch_size, hide_prob, text) -> None:
    args = argparse.Namespace(epochs=epochs, batch_size=batch_size, hide_prob=hide_prob, checkpoint=Path("unused.pt"))
    with pytest.raises(SystemExit, match=text):
        train_model(args)


def test_set_seed_resets_python_numpy_and_torch() -> None:
    set_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    set_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert first == second
