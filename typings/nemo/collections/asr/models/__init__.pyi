from __future__ import annotations

from typing import Sequence

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from nemo.collections.asr.modules import ConformerEncoder
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class ASRModel(Module):
    device: torch.device
    cfg: DictConfig
    _cfg: DictConfig


class EncDecRNNTModel(ASRModel):
    encoder: ConformerEncoder

    def change_decoding_strategy(self, decoder_type: str | DictConfig | None = None, decoding_cfg: DictConfig | None = None) -> None: ...


class EncDecHybridRNNTCTCBPEModel(EncDecRNNTModel):
    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> EncDecHybridRNNTCTCBPEModel: ...

    def conformer_stream_step(
        self,
        *,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        cache_last_channel: Tensor,
        cache_last_time: Tensor,
        cache_last_channel_len: Tensor,
        keep_all_outputs: bool,
        drop_extra_pre_encoded: int | None,
        return_transcription: bool,
        return_log_probs: bool,
        previous_hypotheses: Sequence[Hypothesis] | Sequence[Sequence[Hypothesis]] | None,
        previous_pred_out: list[Tensor] | None,
    ) -> tuple[
        list[Tensor],
        Sequence[Hypothesis] | Sequence[Sequence[Hypothesis]],
        Tensor,
        Tensor,
        Tensor,
        Sequence[Hypothesis] | Sequence[Sequence[Hypothesis]],
    ]: ...
