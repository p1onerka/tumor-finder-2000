from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.viewmodel import TumorViewModel


def _save(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 4), color).save(path)


def _viewmodel(classify=None, **kwargs) -> TumorViewModel:
    kwargs.setdefault("schedule", lambda fn: fn())
    kwargs.setdefault("runner", lambda work: work())
    if classify is not None:
        kwargs["classify"] = classify
    return TumorViewModel(**kwargs)


def test_failed_load_keeps_the_previous_slice(tmp_path) -> None:
    good = tmp_path / "good.png"
    bad = tmp_path / "notes.txt"
    _save(good, (8, 8, 8))
    bad.write_text("not an image", encoding="utf-8")
    viewmodel = _viewmodel()
    viewmodel.load_image(good)
    image = viewmodel.image
    messages: list[str] = []
    viewmodel.add_listener(lambda: messages.append(viewmodel.message))

    viewmodel.load_image(bad)

    assert viewmodel.image is image
    assert viewmodel.search_visible is True
    assert viewmodel.busy is False
    assert viewmodel.message == "Не удалось открыть файл"
    assert messages == ["Не удалось открыть файл"]


def test_failed_load_of_a_missing_file_sets_the_same_message(tmp_path) -> None:
    viewmodel = _viewmodel()
    viewmodel.load_image(tmp_path / "missing.png")
    assert viewmodel.image is None
    assert viewmodel.search_visible is False
    assert viewmodel.message == "Не удалось открыть файл"


def test_search_without_an_image_does_nothing() -> None:
    called: list[str] = []

    def classify(*_args):
        called.append("classify")

    viewmodel = _viewmodel(classify, runner=lambda work: called.append("run"))
    viewmodel.message = "no file"
    viewmodel.search()
    assert called == []
    assert viewmodel.message == "no file"
    assert viewmodel.busy is False


def test_search_while_busy_does_not_start_another(tmp_path) -> None:
    path = tmp_path / "slice.png"
    _save(path, (1, 1, 1))
    runs: list = []

    def classify():
        raise AssertionError("classify should not run")

    viewmodel = _viewmodel(classify, runner=runs.append)
    viewmodel.load_image(path)
    viewmodel.busy = True
    viewmodel.message = "Идёт поиск..."
    viewmodel.search()
    assert runs == []
    assert viewmodel.busy is True
    assert viewmodel.message == "Идёт поиск..."


@pytest.mark.parametrize(
    ("class_name", "label"),
    [
        ("glioma", "глиома"),
        ("meningioma", "менингиома"),
        ("pituitary", "опухоль гипофиза"),
    ],
)
def test_search_shows_a_tumor_label_and_the_highlight(tmp_path, class_name, label) -> None:
    path = tmp_path / "slice.png"
    _save(path, (1, 2, 3))
    found = Image.new("RGB", (4, 4), (9, 0, 0))
    messages: list[str] = []

    def classify(*_args):
        return found, class_name

    viewmodel = _viewmodel(classify)
    viewmodel.add_listener(lambda: messages.append(viewmodel.message))
    viewmodel.load_image(path)
    viewmodel.search()

    assert messages == ["", "Идёт поиск...", f"Обнаружена {label}"]
    assert viewmodel.message == f"Обнаружена {label}"
    assert viewmodel.image is found
    assert viewmodel.search_visible is False
    assert viewmodel.busy is False


def test_search_keeps_the_slice_when_the_highlight_is_not_an_image(tmp_path) -> None:
    path = tmp_path / "slice.png"
    _save(path, (3, 3, 3))

    def classify():
        return "not-an-image", "glioma"

    viewmodel = _viewmodel(classify)
    viewmodel.load_image(path)
    original = viewmodel.image
    viewmodel.search()
    assert viewmodel.image is original
    assert viewmodel.message == "Обнаружена глиома"
    assert viewmodel.search_visible is False


@pytest.mark.parametrize("class_name", ["notumor", "no_tumor", "no-tumor", "NoTumor"])
def test_search_reports_a_clean_slice(tmp_path, class_name) -> None:
    path = tmp_path / "slice.png"
    _save(path, (4, 5, 6))

    def classify(*_args):
        return Image.new("RGB", (4, 4), (255, 0, 0)), class_name

    viewmodel = _viewmodel(classify)
    viewmodel.load_image(path)
    original = viewmodel.image
    viewmodel.search()
    assert viewmodel.message == "Опухоль не обнаружена"
    assert viewmodel.image is original
    assert viewmodel.search_visible is True
    assert viewmodel.busy is False


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (SystemExit("сломанный чекпоинт"), "сломанный чекпоинт"),
        (SystemExit(), "Не удалось обработать снимок"),
        (SystemExit(""), "Не удалось обработать снимок"),
        (RuntimeError("cuda"), "Не удалось обработать снимок: cuda"),
    ],
)
def test_search_reports_classify_errors(tmp_path, error, message) -> None:
    path = tmp_path / "slice.png"
    _save(path, (1, 1, 1))

    def classify(*_args):
        raise error

    viewmodel = _viewmodel(classify)
    viewmodel.load_image(path)
    original = viewmodel.image
    viewmodel.search()
    assert viewmodel.message == message
    assert viewmodel.image is original
    assert viewmodel.search_visible is True
    assert viewmodel.busy is False


def test_search_result_is_dropped_after_a_newer_image_is_loaded(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _save(first, (1, 2, 3))
    _save(second, (4, 5, 6))
    highlighted = Image.new("RGB", (4, 4), (255, 0, 0))
    pending: list = []

    def classify(*_args):
        return highlighted, "glioma"

    viewmodel = _viewmodel(classify, runner=pending.append)
    viewmodel.load_image(first)
    viewmodel.search()
    assert viewmodel.busy is True
    assert viewmodel.message == "Идёт поиск..."

    viewmodel.load_image(second)
    assert viewmodel.message == ""
    assert viewmodel.busy is False
    pending[0]()

    assert viewmodel.image is not None
    assert viewmodel.image.getpixel((0, 0)) == (4, 5, 6)
    assert viewmodel.message == ""
    assert viewmodel.search_visible is True
    assert viewmodel.busy is False
