from __future__ import annotations

import sys

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

from app.view import TumorView
from app.viewmodel import TumorViewModel


class _UiBridge(QObject):
    """Deliver view-model updates to the Qt thread that owns the window."""

    requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.requested.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def schedule(self, fn) -> None:
        self.requested.emit(fn)

    def _run(self, fn) -> None:
        fn()


def main() -> None:
    app = QApplication(sys.argv)
    bridge = _UiBridge()
    viewmodel = TumorViewModel(schedule=bridge.schedule)
    window = TumorView(viewmodel)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
