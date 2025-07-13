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

## Basic usage

```bash
uv run nemo-streaming-asr --help
```

## Install as a tool to run from anywhere in userspace
```bash
./install-as-tool
nemo-streaming-asr --help
```
