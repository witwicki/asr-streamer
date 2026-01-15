# ASR STREAMER

## Description

**asr-streamer** is intended as a platform for evaluating and deploying ASR models on streaming audio inputs.

Key features:

- Invocation of _streaming_ inference models for automated speech recognition, starting with [NVIDIA's Fast-Conformer Model](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- Publishing of recognized speech over TCP to multiple ports simultaneously
- Press-to-talk UI by remote control, so the user can decide when to speech transcriptions are recorded and published

## Prerequisites

### Portaudio

#### debian / ubuntu instructions

```bash
sudo apt install libportaudio2
```

#### macos install instructions

```bash
xcode-select --install
brew install portaudio
```

### uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Basic usage

```bash
uv run nemo-streaming-asr --help
```

## Install as a tool to run from anywhere in userspace

```bash
./install-as-uv-tool
nemo-streaming-asr --help
```
