"""Progress reporting and cancellation shared by the CLI and the GUI."""
from __future__ import annotations

import threading
from typing import Callable, Optional

# report(stage, done, total): stage is one of STAGE_NAMES; done/total = 0 when there is no count.
ReportFn = Callable[[str, float, float], None]

STAGE_NAMES = {
    "download": "Скачивание видео",
    "audio": "Извлечение звука",
    "transcribe": "Распознавание речи",
    "llm_start": "Запуск LLM",
    "highlights": "Поиск вирусных моментов",
    "rerank": "Переоценка кандидатов",
    "render": "Кадрирование и рендер",
    "done": "Готово",
}


class Cancelled(Exception):
    pass


class Reporter:
    """Callable passed down the pipeline; raises Cancelled once cancel() was requested."""

    def __init__(self, callback: Optional[ReportFn] = None):
        self.callback = callback
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def __call__(self, stage: str, done: float = 0.0, total: float = 0.0) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        if self.callback:
            self.callback(stage, done, total)
