"""Общие фикстуры pytest."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def sample_voice_metrics() -> dict[str, int | float]:
    """Возвращает пример набора голосовых метрик для тестов.

    Returns:
        Словарь с числовыми метриками голоса для модулей `qwen` и `process/main`.
    """
    return {
        "duration_sec": 120.5,
        "num_segments": 15,
        "pitch_hz_median": 180.0,
        "pitch_hz_p10": 150.0,
        "pitch_hz_p90": 220.0,
        "intonation_range_st_p10_p90": 7.5,
        "speech_rate_syllables_per_sec": 4.2,
        "speech_rate_wpm_est": 93.0,
        "loudness_dbfs_mean": -18.0,
        "loudness_dbfs_p90": -12.0,
        "timbre_centroid_hz_mean": 1200.0,
        "timbre_rolloff_hz_mean": 2400.0,
        "timbre_flatness_mean": 0.35,
    }


@pytest.fixture
def sample_diarization_segments() -> list[dict[str, str | float]]:
    """Возвращает пример сегментов диаризации в плоском формате.

    Returns:
        Список словарей с ключами `speaker`, `start`, `end`.
    """
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5},
        {"speaker": "SPEAKER_01", "start": 2.5, "end": 5.0},
        {"speaker": "SPEAKER_00", "start": 5.0, "end": 8.0},
    ]


@pytest.fixture
def sample_diarization_data(
    sample_diarization_segments: list[dict[str, str | float]],
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Преобразует плоские сегменты в формат `group_segments_by_speaker`.

    Args:
        sample_diarization_segments: Список исходных сегментов диаризации.

    Returns:
        Словарь `{speaker: {"segments": [(start, end), ...]}}`.
    """
    speakers: dict[str, dict[str, list[tuple[float, float]]]] = {}
    seg: dict[str, str | float]
    for seg in sample_diarization_segments:
        spk: str = str(seg["speaker"])
        if spk not in speakers:
            speakers[spk] = {"segments": []}
        speakers[spk]["segments"].append((float(seg["start"]), float(seg["end"])))
    return speakers


@pytest.fixture
def temp_audio_dir() -> Generator[Path, None, None]:
    """Создаёт временную директорию для тестовых артефактов.

    Yields:
        Путь к временной директории.
    """
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def short_mono_wav(temp_audio_dir: Path) -> Path:
    """Создаёт короткий моно WAV-файл для тестов `voice_assessment`.

    Args:
        temp_audio_dir: Временная директория для записи тестового файла.

    Returns:
        Путь к созданному WAV-файлу.
    """
    path: Path = temp_audio_dir / "test.wav"
    sr: int = 16000
    duration: float = 1.0
    n: int = int(sr * duration)
    t: np.ndarray = np.linspace(0, duration, n, dtype=np.float32)
    wav: np.ndarray = (
        0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        + 0.01 * np.random.randn(n).astype(np.float32)
    )
    import soundfile as sf

    sf.write(str(path), wav, sr)
    return path
