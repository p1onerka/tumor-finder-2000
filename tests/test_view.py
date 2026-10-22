from __future__ import annotations

import os
from PIL import Image
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.view import ImagePane, TumorView
from app.viewmodel import TumorViewModel

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# QWidget aborts the process if it is created before QApplication.
_qapp = QApplication.instance() or QApplication([])


def test_image_pane_starts_empty_and_ignores_a_null_pixmap() -> None:
    pane = ImagePane()
    assert pane.text() == "Снимок не выбран"
    pane.set_image(QPixmap())
    assert pane.text() == "Снимок не выбран"
    assert pane._source is None
    assert pane.pixmap().isNull()


def test_choose_file_loads_the_selected_slice(tmp_path, monkeypatch) -> None:
    path = tmp_path / "slice.png"
    Image.new("RGB", (3, 3), (4, 5, 6)).save(path)
    view = TumorView(TumorViewModel())
    monkeypatch.setattr(
        "app.view.QFileDialog.getOpenFileName", lambda *args: (str(path), "")
    )
    view._choose_file()
    assert view.viewmodel.image is not None
    assert view.viewmodel.image.getpixel((0, 0)) == (4, 5, 6)
    assert not view.search_button.isHidden()
    assert view.image_pane.text() == ""


def test_choose_file_ignores_none_img(monkeypatch) -> None:
    view = TumorView(TumorViewModel())
    monkeypatch.setattr("app.view.QFileDialog.getOpenFileName", lambda *args: ("", ""))
    view._choose_file()
    assert view.viewmodel.image is None
    assert view.image_pane.text() == "Снимок не выбран"


def test_render_shows_message_image_and_busy_state() -> None:
    viewmodel = TumorViewModel()
    view = TumorView(viewmodel)
    assert view.windowTitle() == "Поиск опухоли"
    assert view.choose_button.text() == "Выберите снимок"
    assert view.search_button.text() == "Искать опухоль"
    assert view.minimumWidth() == 960
    assert view.minimumHeight() == 560
    assert view.search_button.isHidden()

    viewmodel.busy = True
    viewmodel.search_visible = True
    viewmodel.message = "Идёт поиск..."
    viewmodel.image = Image.new("RGB", (3, 3), (7, 8, 9))
    view.render()

    assert not view.choose_button.isEnabled()
    assert not view.search_button.isEnabled()
    assert not view.search_button.isHidden()
    assert view.message_label.text() == "Идёт поиск..."
    assert view.message_label.property("filled") == "true"

    viewmodel.busy = False
    viewmodel.search_visible = False
    viewmodel.message = ""
    viewmodel.image = None
    view.render()

    assert view.choose_button.isEnabled()
    assert view.search_button.isHidden()
    assert view.message_label.text() == ""
    assert view.message_label.property("filled") == "false"
    assert view.image_pane.text() == "Снимок не выбран"
