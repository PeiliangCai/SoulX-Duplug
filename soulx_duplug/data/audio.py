from __future__ import annotations

import io
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioData:
    waveform: np.ndarray
    sample_rate: int
    channels: int

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.waveform.shape[0]) / float(self.sample_rate)


@dataclass(frozen=True)
class AudioMetadata:
    duration: float
    sample_rate: int
    channels: int


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        return signed.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def _read_wav(path: Path) -> AudioData:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    samples = _decode_pcm(frames, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return AudioData(samples.astype(np.float32, copy=False), sample_rate, channels)


def _read_with_soundfile(path: Path) -> AudioData:
    try:
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("soundfile is not installed") from exc
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    channels = data.shape[1]
    mono = data.mean(axis=1)
    return AudioData(mono.astype(np.float32, copy=False), int(sample_rate), channels)


def _read_with_ffmpeg(path: Path, target_sample_rate: int | None = None) -> AudioData:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available")
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1"]
    if target_sample_rate is not None:
        command.extend(["-ar", str(target_sample_rate)])
    command.append("-")
    proc = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    samples = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    if target_sample_rate is None:
        raise RuntimeError("ffmpeg reader requires target_sample_rate to report metadata")
    return AudioData(samples, target_sample_rate, 1)


def read_audio(path: str | Path, *, ffmpeg_sample_rate: int | None = None) -> AudioData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return _read_wav(path)
    try:
        return _read_with_soundfile(path)
    except RuntimeError:
        try:
            return _read_with_ffmpeg(path, target_sample_rate=ffmpeg_sample_rate)
        except RuntimeError as exc:
            raise RuntimeError(
                f"cannot read {path}. Install soundfile/torchaudio or ffmpeg for non-WAV audio."
            ) from exc


def crop_audio(audio: AudioData, start: float | None = None, end: float | None = None) -> AudioData:
    if start is None and end is None:
        return audio
    start_sample = 0 if start is None else max(0, int(round(start * audio.sample_rate)))
    end_sample = audio.waveform.shape[0] if end is None else max(start_sample, int(round(end * audio.sample_rate)))
    end_sample = min(end_sample, audio.waveform.shape[0])
    return AudioData(audio.waveform[start_sample:end_sample], audio.sample_rate, audio.channels)


def read_audio_segment(
    path: str | Path,
    *,
    start: float | None = None,
    end: float | None = None,
    ffmpeg_sample_rate: int | None = None,
) -> AudioData:
    return crop_audio(read_audio(path, ffmpeg_sample_rate=ffmpeg_sample_rate), start=start, end=end)


def audio_metadata(path: str | Path) -> AudioMetadata:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
        return AudioMetadata(frames / float(sample_rate), sample_rate, channels)
    audio = read_audio(path)
    return AudioMetadata(audio.duration, audio.sample_rate, audio.channels)


def resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"invalid sample rates: {source_rate} -> {target_rate}")
    if waveform.size == 0:
        return waveform.astype(np.float32)
    target_length = max(1, int(round(waveform.shape[0] * target_rate / source_rate)))
    old_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=True)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(new_positions, old_positions, waveform).astype(np.float32)


def write_wav_pcm16(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def normalize_audio_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    target_sample_rate: int = 16000,
    force: bool = False,
    segment_start: float | None = None,
    segment_end: float | None = None,
) -> AudioMetadata:
    target_path = Path(target_path)
    if target_path.exists() and not force:
        return audio_metadata(target_path)
    audio = read_audio_segment(
        source_path,
        start=segment_start,
        end=segment_end,
        ffmpeg_sample_rate=target_sample_rate,
    )
    waveform = resample_linear(audio.waveform, audio.sample_rate, target_sample_rate)
    write_wav_pcm16(target_path, waveform, target_sample_rate)
    return AudioMetadata(len(waveform) / float(target_sample_rate), target_sample_rate, 1)


def make_test_wav(path: str | Path, *, sample_rate: int = 44100, seconds: float = 0.1) -> None:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    waveform = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    write_wav_pcm16(path, waveform.astype(np.float32), sample_rate)
