from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.viewmodel import TumorViewModel

DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 680
MINIMUM_WIDTH = 960
MINIMUM_HEIGHT = 560
SIDE_WIDTH = 260
MESSAGE_HEIGHT = 96

STYLESHEET = """
QWidget#tumorWindow {
    background-color: #EBF5EE;
    color: #283044;
}
QLabel#imageLabel {
    background-color: #283044;
    color: #BFA89E;
    border: 3px solid #8B786D;
    border-radius: 8px;
    font-size: 16px;
}
QLabel#messageLabel {
    font-size: 16px;
    color: #283044;
    background-color: transparent;
    border: 2px solid transparent;
    border-radius: 8px;
    padding: 8px 12px;
}
QLabel#messageLabel[filled="true"] {
    background-color: #BFA89E;
    border-color: #8B786D;
}
QPushButton {
    background-color: #78A1BB;
    color: #283044;
    border: 1px solid #78A1BB;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #283044;
    color: #EBF5EE;
    border-color: #283044;
}
QPushButton:pressed {
    background-color: #8B786D;
    color: #EBF5EE;
    border-color: #8B786D;
}
QPushButton:disabled {
    background-color: #BFA89E;
    color: #283044;
    border-color: #8B786D;
}
QPushButton#searchButton {
    background-color: #283044;
    color: #EBF5EE;
    border-color: #283044;
}
QPushButton#searchButton:hover {
    background-color: #78A1BB;
    color: #283044;
    border-color: #78A1BB;
}
QPushButton#searchButton:pressed {
    background-color: #8B786D;
    color: #EBF5EE;
    border-color: #8B786D;
}
QPushButton#searchButton:disabled {
    background-color: #BFA89E;
    color: #283044;
    border-color: #8B786D;
}
"""


class ImagePane(QLabel):
    def __init__(self) -> None:
        super().__init__("Снимок не выбран")
        self._source: QPixmap | None = None
        self.setObjectName("imageLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap if pixmap is not None and not pixmap.isNull() else None
        self.setText("" if self._source is not None else "Снимок не выбран")
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def minimumSizeHint(self):
        # The file's pixel size must not push the window open.
        return self.minimumSize()

    def sizeHint(self):
        return self.minimumSize()

    def _rescale(self) -> None:
        if self._source is None:
            self.setPixmap(QPixmap())
            return
        area = self.contentsRect().size()
        if area.width() < 2 or area.height() < 2:
            return
        ratio = self.devicePixelRatioF()
        scaled = self._source.scaled(
            max(1, int(area.width() * ratio)),
            max(1, int(area.height() * ratio)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        self.setPixmap(scaled)


class TumorView(QWidget):
    def __init__(self, viewmodel: TumorViewModel) -> None:
        super().__init__()
        self.viewmodel = viewmodel
        self.setObjectName("tumorWindow")
        self.setWindowTitle("Поиск опухоли")

        self.choose_button = QPushButton("Выберите снимок")
        self.choose_button.setObjectName("chooseButton")
        self.choose_button.setFixedHeight(44)
        self.choose_button.clicked.connect(self._choose_file)

        self.image_pane = ImagePane()

        self.message_label = QLabel()
        self.message_label.setObjectName("messageLabel")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setMinimumHeight(MESSAGE_HEIGHT)

        self.search_button = QPushButton("Искать опухоль")
        self.search_button.setObjectName("searchButton")
        self.search_button.setFixedHeight(44)
        self.search_button.clicked.connect(viewmodel.search)

        for button in (self.choose_button, self.search_button):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        side = QWidget()
        side.setFixedWidth(SIDE_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(16)
        side_layout.addWidget(self.choose_button)
        side_layout.addWidget(self.search_button)
        side_layout.addWidget(self.message_label)
        side_layout.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        layout.addWidget(self.image_pane, stretch=1)
        layout.addWidget(side)

        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(MINIMUM_WIDTH, MINIMUM_HEIGHT)
        self.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)

        viewmodel.add_listener(self.render)
        self.render()

    def _choose_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите снимок",
            "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;Все файлы (*)",
        )
        if filename:
            self.viewmodel.load_image(Path(filename))

    def render(self) -> None:
        viewmodel = self.viewmodel
        self.choose_button.setEnabled(not viewmodel.busy)
        self.search_button.setEnabled(not viewmodel.busy)
        self.search_button.setVisible(viewmodel.search_visible)

        self.image_pane.set_image(None if viewmodel.image is None else _pixmap(viewmodel.image))

        self.message_label.setText(viewmodel.message)
        self.message_label.setProperty("filled", "true" if viewmodel.message else "false")
        self.message_label.style().unpolish(self.message_label)
        self.message_label.style().polish(self.message_label)


def _pixmap(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    buffer = rgb.tobytes("raw", "RGB")
    qimage = QImage(buffer, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
