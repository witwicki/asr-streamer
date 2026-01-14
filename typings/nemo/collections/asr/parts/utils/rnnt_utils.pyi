from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Hypothesis:
    text: str | None
    tokens: Sequence[int] | None = None
