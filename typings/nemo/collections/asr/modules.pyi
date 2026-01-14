from __future__ import annotations

from typing import Sequence

from torch import Tensor


class CacheAwareStreamingConfig:
    pre_encode_cache_size: int | Sequence[int]


class ConformerEncoder:
    att_context_size: Sequence[int]
    streaming_cfg: CacheAwareStreamingConfig

    def set_default_att_context_size(self, context: Sequence[int]) -> None: ...

    def get_initial_cache_state(self, batch_size: int = 1) -> tuple[Tensor, Tensor, Tensor]: ...
