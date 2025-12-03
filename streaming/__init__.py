from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


def suppress_terminal_output(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that suppresses stdout/stderr when `self.verbose` is False."""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self_obj = args[0]
        if getattr(self_obj, "verbose", True):
            return func(*args, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # additional suppression of low-level stderr writes
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            try:
                _ = sys.stderr.flush()
                _ = os.dup2(devnull, 2)
                return func(*args, **kwargs)
            finally:
                _ = os.dup2(old_stderr, 2)
                os.close(old_stderr)
                os.close(devnull)

    return wrapper
