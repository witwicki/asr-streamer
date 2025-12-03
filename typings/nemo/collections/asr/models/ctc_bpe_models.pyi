from __future__ import annotations

from omegaconf import DictConfig
from torch.nn import Module


class EncDecCTCModelBPE(Module):
    _cfg: DictConfig

    @classmethod
    def from_config_dict(cls, config: DictConfig) -> EncDecCTCModelBPE: ...
