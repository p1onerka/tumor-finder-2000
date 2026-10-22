from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.train import (
    CLASSIFIER_CHECKPOINT,
    LOCALIZER_CHECKPOINT,
    classify_image,
    is_tumor_class,
)

TUMOR_LABELS = {
    "glioma": "глиома",
    "meningioma": "менингиома",
    "pituitary": "опухоль гипофиза",
}

Classify = Callable[[Path, Path, Path], tuple[Image.Image, str]]
Schedule = Callable[[Callable[[], None]], None]
Runner = Callable[[Callable[[], None]], None]


class TumorViewModel:
    def __init__(
        self,
        checkpoint: Path = CLASSIFIER_CHECKPOINT,
        localizer_checkpoint: Path = LOCALIZER_CHECKPOINT,
        classify: Classify = classify_image,
        schedule: Schedule | None = None,
        runner: Runner | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._localizer_checkpoint = localizer_checkpoint
        self._classify = classify
        self._schedule = schedule or (lambda fn: fn())
        self._runner = runner or self._start_thread
        self._listeners: list[Callable[[], None]] = []
        self._generation = 0
        self._image_path: Path | None = None
        self.image: Image.Image | None = None
        self.message = ""
        self.search_visible = False
        self.busy = False

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def load_image(self, path: Path) -> None:
        path = Path(path)
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except Exception:
            self.message = "Не удалось открыть файл"
            self._notify()
            return
        self._generation += 1
        self.busy = False
        self._image_path = path
        self.image = image
        self.message = ""
        self.search_visible = True
        self._notify()

    def search(self) -> None:
        if self.busy or self._image_path is None:
            return
        self.busy = True
        self.message = "Идёт поиск..."
        self._notify()
        generation = self._generation
        path = self._image_path

        def work() -> None:
            outcome = self._run_classify(path)
            self._schedule(lambda: self._finish(generation, outcome))

        self._runner(work)

    def _run_classify(self, path: Path) -> tuple[str, ...]:
        try:
            image, class_name = self._classify(
                path, self._checkpoint, self._localizer_checkpoint
            )
        except SystemExit as exc:
            return ("error", str(exc) or "Не удалось обработать снимок")
        except Exception as exc:
            return ("error", f"Не удалось обработать снимок: {exc}")
        return ("ok", image, class_name)

    def _finish(self, generation: int, outcome: tuple[str, ...]) -> None:
        if generation != self._generation:
            return
        self.busy = False
        if outcome[0] == "error":
            self.message = str(outcome[1])
        else:
            image = outcome[1]
            class_name = str(outcome[2])
            if is_tumor_class(class_name):
                if isinstance(image, Image.Image):
                    self.image = image
                label = TUMOR_LABELS.get(class_name, class_name)
                self.message = f"Обнаружена {label}"
                self.search_visible = False
            else:
                self.message = "Опухоль не обнаружена"
        self._emit()

    def _notify(self) -> None:
        self._schedule(self._emit)

    def _emit(self) -> None:
        for listener in list(self._listeners):
            listener()

    @staticmethod
    def _start_thread(work: Callable[[], None]) -> None:
        threading.Thread(target=work, daemon=True).start()
