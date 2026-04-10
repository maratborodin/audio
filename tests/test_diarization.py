"""Тесты для diarization.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from diarization import get_diarization, group_segments_by_speaker


class TestGroupSegmentsBySpeaker:
    """Проверки группировки плоских сегментов по идентификатору спикера."""

    def test_empty(self) -> None:
        """Проверяет возврат пустого словаря для пустого списка сегментов.

        Returns:
            `None`.
        """
        assert group_segments_by_speaker([]) == {}

    def test_single_speaker(
        self,
        sample_diarization_segments: list[dict[str, str | float]],
    ) -> None:
        """Проверяет корректную группировку сегментов одного спикера.

        Args:
            sample_diarization_segments: Пример сегментов диаризации.

        Returns:
            `None`.
        """
        one: list[dict[str, str | float]] = [sample_diarization_segments[0]]
        out: dict[str, dict[str, list[tuple[float, float]]]] = (
            group_segments_by_speaker(one)
        )
        assert list(out.keys()) == ["SPEAKER_00"]
        assert out["SPEAKER_00"]["segments"] == [(0.0, 2.5)]

    def test_multiple_speakers(
        self,
        sample_diarization_segments: list[dict[str, str | float]],
    ) -> None:
        """Проверяет разбиение сегментов между несколькими спикерами.

        Args:
            sample_diarization_segments: Пример сегментов диаризации.

        Returns:
            `None`.
        """
        out: dict[str, dict[str, list[tuple[float, float]]]] = (
            group_segments_by_speaker(sample_diarization_segments)
        )
        assert "SPEAKER_00" in out and "SPEAKER_01" in out
        assert out["SPEAKER_00"]["segments"] == [(0.0, 2.5), (5.0, 8.0)]
        assert out["SPEAKER_01"]["segments"] == [(2.5, 5.0)]

    def test_segment_format(self) -> None:
        """Проверяет сохранение порядка и формата пар `(start, end)`.

        Returns:
            `None`.
        """
        segments: list[dict[str, str | float]] = [
            {"speaker": "A", "start": 1.0, "end": 2.0},
            {"speaker": "A", "start": 3.0, "end": 4.0},
        ]
        out: dict[str, dict[str, list[tuple[float, float]]]] = (
            group_segments_by_speaker(segments)
        )
        assert out["A"]["segments"] == [(1.0, 2.0), (3.0, 4.0)]


class TestGetDiarization:
    """Проверки оркестрации вызова pyannote и преобразования его результата."""

    @patch("diarization.soundfile.read")
    @patch("diarization.Pipeline")
    def test_returns_grouped_speakers(
        self,
        mock_pipeline_cls: MagicMock,
        mock_sf_read: MagicMock,
    ) -> None:
        """Проверяет возврат сгруппированных сегментов из результата pyannote.

        Args:
            mock_pipeline_cls: Мок класса `Pipeline`.
            mock_sf_read: Мок чтения WAV-файла через `soundfile.read`.

        Returns:
            `None`.
        """
        mock_sf_read.return_value = (
            np.zeros(16000 * 2, dtype=np.float32),
            16000,
        )
        mock_pipeline: MagicMock = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        class FakeTurn:
            def __init__(self, start: float, end: float) -> None:
                self.start: float = start
                self.end: float = end

        mock_diarization: MagicMock = MagicMock()
        mock_diarization.itertracks.return_value = [
            (FakeTurn(0.0, 1.0), None, "SPEAKER_00"),
            (FakeTurn(1.0, 2.0), None, "SPEAKER_01"),
        ]
        mock_pipeline.return_value.speaker_diarization = mock_diarization

        result: dict[str, dict[str, list[tuple[float, float]]]] = (
            get_diarization("/fake/path.wav")
        )

        assert "SPEAKER_00" in result and "SPEAKER_01" in result
        assert result["SPEAKER_00"]["segments"] == [(0.0, 1.0)]
        assert result["SPEAKER_01"]["segments"] == [(1.0, 2.0)]
        mock_sf_read.assert_called_once()
        mock_pipeline.assert_called_once()
