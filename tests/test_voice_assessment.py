"""Тесты для voice_assessment.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_assessment import (
    VoiceMetrics,
    assess_voices,
    _extract_segments,
    _slice,
    _pitch_summaries,
    _estimate_pitch_for_frame,
    _frame_loudness_dbfs,
    _timbre_stats,
    _estimate_syllable_peaks,
    _compute_segment_metrics,
    _compute_metrics_for_speaker,
)


class TestVoiceMetrics:
    """Проверки сериализации датакласса `VoiceMetrics`."""

    def test_to_dict(self) -> None:
        m: VoiceMetrics = VoiceMetrics(
            duration_sec=10.0,
            num_segments=2,
            pitch_hz_median=200.0,
            pitch_hz_p10=180.0,
            pitch_hz_p90=220.0,
            intonation_range_st_p10_p90=5.0,
            speech_rate_syllables_per_sec=4.0,
            speech_rate_wpm_est=90.0,
            loudness_dbfs_mean=-20.0,
            loudness_dbfs_p90=-15.0,
            timbre_centroid_hz_mean=1000.0,
            timbre_rolloff_hz_mean=2000.0,
            timbre_flatness_mean=0.3,
        )
        d: dict[str, int | float | None] = m.to_dict()
        assert d["duration_sec"] == 10.0
        assert d["num_segments"] == 2
        assert d["pitch_hz_median"] == 200.0
        assert d["timbre_flatness_mean"] == 0.3

    def test_to_dict_with_nones(self) -> None:
        m: VoiceMetrics = VoiceMetrics(
            duration_sec=0.0,
            num_segments=0,
            pitch_hz_median=None,
            pitch_hz_p10=None,
            pitch_hz_p90=None,
            intonation_range_st_p10_p90=None,
            speech_rate_syllables_per_sec=None,
            speech_rate_wpm_est=None,
            loudness_dbfs_mean=None,
            loudness_dbfs_p90=None,
            timbre_centroid_hz_mean=None,
            timbre_rolloff_hz_mean=None,
            timbre_flatness_mean=None,
        )
        d = m.to_dict()
        assert d["pitch_hz_median"] is None


class TestExtractSegments:
    """Проверки извлечения и валидации сегментов спикера."""

    def test_valid_tuples(self) -> None:
        payload: dict[str, list[tuple[float, float]]] = {
            "segments": [(0.0, 1.0), (2.0, 3.0)]
        }
        assert _extract_segments(payload) == [(0.0, 1.0), (2.0, 3.0)]

    def test_valid_lists(self) -> None:
        payload: dict[str, list[list[float]]] = {
            "segments": [[0.0, 1.0], [2.0, 3.0]]
        }
        assert _extract_segments(payload) == [(0.0, 1.0), (2.0, 3.0)]

    def test_skips_invalid_end_before_start(self) -> None:
        payload: dict[str, list[tuple[float, float]]] = {
            "segments": [(1.0, 0.5)]
        }
        assert _extract_segments(payload) == []

    def test_not_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            _extract_segments("not a dict")

    def test_segments_not_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence"):
            _extract_segments({"segments": 123})

    def test_invalid_segment_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid segment"):
            _extract_segments({"segments": ["not a pair"]})

    def test_missing_segments_defaults_empty(self) -> None:
        assert _extract_segments({"other": 1}) == []


class TestSlice:
    """Проверки вырезания временного диапазона из waveform."""

    def test_normal(self) -> None:
        wav: np.ndarray = np.arange(100, dtype=np.float32)
        out: np.ndarray = _slice(wav, 10, 1.0, 3.0)
        assert out.shape == (20,)
        np.testing.assert_array_almost_equal(out, wav[10:30])

    def test_empty_if_end_before_start(self) -> None:
        wav: np.ndarray = np.arange(100, dtype=np.float32)
        out: np.ndarray = _slice(wav, 10, 3.0, 1.0)
        assert out.size == 0

    def test_clamps_to_bounds(self) -> None:
        wav: np.ndarray = np.arange(100, dtype=np.float32)
        out: np.ndarray = _slice(wav, 10, -1.0, 100.0)
        assert out.size <= 100


class TestPitchSummaries:
    """Проверки агрегирования распределения pitch по сегменту."""

    def test_empty(self) -> None:
        assert _pitch_summaries([]) == (None, None, None, None)

    def test_single_value(self) -> None:
        med: float | None
        p10: float | None
        p90: float | None
        rng: float | None
        med, p10, p90, rng = _pitch_summaries([200.0])
        assert med == 200.0
        assert p10 == p90 == 200.0
        assert rng == 0.0

    def test_multiple_values(self) -> None:
        vals: list[float] = [100.0, 150.0, 200.0, 250.0, 300.0]
        med, p10, p90, rng = _pitch_summaries(vals)
        assert med == 200.0
        assert p10 is not None and p90 is not None and med is not None
        assert p10 <= med <= p90
        assert rng is not None and rng > 0


class TestEstimatePitchForFrame:
    """Проверки оценки F0 по одному кадру сигнала."""

    def test_low_energy_returns_none(self) -> None:
        x: np.ndarray = np.zeros(256, dtype=np.float32)
        assert _estimate_pitch_for_frame(
            x, 16000, energy_thr=0.1, lag_min=32, lag_max=320, f_min=50.0, f_max=500.0
        ) is None

    def test_f0_out_of_range_returns_none(self) -> None:
        sr: int = 16000
        n: int = 512
        x: np.ndarray = (
            np.sin(2 * np.pi * 30 * np.linspace(0, n / sr, n)) * 0.5
        ).astype(np.float32)
        result: float | None = _estimate_pitch_for_frame(
            x, sr, energy_thr=0.01, lag_min=32, lag_max=200, f_min=80.0, f_max=400.0
        )
        assert result is None or (80 <= result <= 400)

    def test_sinusoid_gives_plausible_f0(self) -> None:
        sr: int = 16000
        f0: float = 200.0
        n: int = 512
        t: np.ndarray = np.linspace(0, n / sr, n, dtype=np.float32)
        x: np.ndarray = (np.sin(2 * np.pi * f0 * t) * 0.5).astype(np.float32)
        lag_min: int = int(sr / 500)
        lag_max: int = int(sr / 50)
        result: float | None = _estimate_pitch_for_frame(
            x,
            sr,
            energy_thr=0.01,
            lag_min=lag_min,
            lag_max=lag_max,
            f_min=50.0,
            f_max=500.0,
        )
        assert result is not None
        assert 150 <= result <= 250


class TestFrameLoudnessDbfs:
    """Проверки расчёта громкости по кадрам в dBFS."""

    def test_empty_segment(self) -> None:
        seg: np.ndarray = np.zeros(10, dtype=np.float32)
        assert _frame_loudness_dbfs(seg, 16000, frame_ms=25.0, hop_ms=10.0) == []

    def test_returns_list(self) -> None:
        seg: np.ndarray = np.random.randn(16000).astype(np.float32) * 0.1
        out: list[float] = _frame_loudness_dbfs(
            seg, 16000, frame_ms=25.0, hop_ms=10.0
        )
        assert len(out) > 0
        assert all(isinstance(v, float) for v in out)


class TestTimbreStats:
    """Проверки расчёта спектральных метрик тембра."""

    def test_too_short_returns_none(self) -> None:
        seg: np.ndarray = np.zeros(100, dtype=np.float32)
        assert _timbre_stats(seg, 16000, rolloff=0.85) == (None, None, None)

    def test_long_enough_returns_triple(self) -> None:
        seg: np.ndarray = np.random.randn(2048).astype(np.float32) * 0.2
        c, r, f = _timbre_stats(seg, 16000, rolloff=0.85)
        assert c is not None and r is not None and f is not None
        assert 0 <= f <= 1


class TestEstimateSyllablePeaks:
    """Проверки грубой оценки числа слоговых пиков."""

    def test_too_short_returns_zero(self) -> None:
        seg: np.ndarray = np.zeros(100, dtype=np.float32)
        assert _estimate_syllable_peaks(seg, 16000) == 0

    def test_silence_returns_zero_or_small(self) -> None:
        seg: np.ndarray = np.zeros(16000, dtype=np.float32)
        n: int = _estimate_syllable_peaks(seg, 16000)
        assert n == 0


class TestComputeSegmentMetrics:
    """Проверки сводного расчёта метрик для одного сегмента."""

    def test_short_segment_returns_zeros(self) -> None:
        wav: np.ndarray = np.zeros(16000, dtype=np.float32)
        dur, p, db, c, r, f, syll = _compute_segment_metrics(
            wav, 16000, 0.0, 0.01, frame_ms=25.0, hop_ms=10.0, rolloff=0.85
        )
        assert dur == 0.0
        assert len(p) == 0
        assert syll == 0

    def test_longer_segment_returns_metrics(self) -> None:
        sr: int = 16000
        wav: np.ndarray = (
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2)) * 0.3
        ).astype(np.float32)
        dur, p, db, c, r, f, syll = _compute_segment_metrics(
            wav, sr, 0.0, 0.5, frame_ms=25.0, hop_ms=10.0, rolloff=0.85
        )
        assert dur > 0
        assert isinstance(p, list)
        assert isinstance(db, list)


class TestComputeMetricsForSpeaker:
    """Проверки агрегации сегментных метрик на уровне спикера."""

    def test_empty_segments(self) -> None:
        wav: np.ndarray = np.zeros(16000, dtype=np.float32)
        m: VoiceMetrics = _compute_metrics_for_speaker(
            wav=wav, sr=16000, segments=[], frame_ms=25.0, hop_ms=10.0, rolloff=0.85
        )
        assert m.duration_sec == 0.0
        assert m.num_segments == 0
        assert m.pitch_hz_median is None

    def test_one_segment(self) -> None:
        sr: int = 16000
        wav: np.ndarray = (
            np.sin(2 * np.pi * 200 * np.linspace(0, 1.0, sr)) * 0.3
        ).astype(np.float32)
        m: VoiceMetrics = _compute_metrics_for_speaker(
            wav=wav,
            sr=sr,
            segments=[(0.0, 1.0)],
            frame_ms=25.0,
            hop_ms=10.0,
            rolloff=0.85,
        )
        assert m.duration_sec > 0
        assert m.num_segments == 1


class TestAssessVoices:
    """Проверки верхнеуровневой функции оценки голосов по диаризации."""

    def test_raises_if_diarization_not_mapping(self) -> None:
        with pytest.raises(ValueError, match="diarization"):
            assess_voices({"diarization": None}, "/fake/path.wav")
        with pytest.raises(ValueError, match="diarization"):
            assess_voices({"diarization": "str"}, "/fake/path.wav")

    @patch("voice_assessment.sf.read")
    def test_returns_metrics_per_speaker(
        self,
        mock_read: MagicMock,
        short_mono_wav: Path,
    ) -> None:
        sr: int = 16000
        wav: np.ndarray = np.random.randn(sr * 2).astype(np.float32) * 0.2
        mock_read.return_value = (wav, sr)

        data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {
            "diarization": {
                "SPEAKER_00": {"segments": [(0.0, 0.5), (1.0, 1.5)]},
                "SPEAKER_01": {"segments": [(0.5, 1.0)]},
            }
        }
        result: dict[str, dict[str, object]] = assess_voices(
            data, "/any/path.wav"
        )
        assert "SPEAKER_00" in result and "SPEAKER_01" in result
        spk: str
        metrics: dict[str, object]
        for spk, metrics in result.items():
            assert "duration_sec" in metrics
            assert "num_segments" in metrics
            assert "pitch_hz_median" in metrics
