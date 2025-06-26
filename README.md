# STREAMING ASR SERVER

## Description

The beginnings of a platform for invoking ASR models on streaming audio inputs.

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

## Installation

```bash
git clone https://github.com/witwicki/asr-streamer.git
cd asr-streamer
uv sync
uv tool install .
```

## Basic usage

```bash
uv run nemo_streaming_asr --help
```

### and if you happen to be outside of the project directory...
```bash
uv run --project ~/projects/dev/asr-streamer nemo_streaming_asr
```
