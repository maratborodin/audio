"""Тесты для process.py."""
from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def process_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Импортирует `process` с заглушками тяжёлых зависимостей.

    Args:
        monkeypatch: Фикстура monkeypatch.

    Returns:
        Перезагруженный модуль `process`.
    """
    fake_torch: ModuleType = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.Tensor = object
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb
            return None

    fake_torch.no_grad = lambda: _NoGrad()

    fake_transformers: ModuleType = ModuleType("transformers")

    class FakeAutoModelForSpeechSeq2Seq:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            _ = args, kwargs
            model: MagicMock = MagicMock()
            model.device = "cpu"
            return model

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            _ = args, kwargs
            return MagicMock()

    fake_transformers.AutoModelForSpeechSeq2Seq = FakeAutoModelForSpeechSeq2Seq
    fake_transformers.AutoProcessor = FakeAutoProcessor

    fake_diarization: ModuleType = ModuleType("diarization")
    fake_diarization.get_diarization = lambda audio_source: {}

    fake_qwen: ModuleType = ModuleType("qwen")
    fake_qwen.describe_all_voices_with_qwen = lambda metrics: {}

    fake_voice_assessment: ModuleType = ModuleType("voice_assessment")
    fake_voice_assessment.assess_voices = lambda data, audio_path: {}

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "diarization", fake_diarization)
    monkeypatch.setitem(sys.modules, "qwen", fake_qwen)
    monkeypatch.setitem(sys.modules, "voice_assessment", fake_voice_assessment)

    import process as process_imported

    return importlib.reload(process_imported)


class TestWhisperRuntime:
    """Проверки загрузки и прямого вызова Whisper runtime."""

    def test_load_whisper_pipeline(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет создание `WhisperRuntime` через конструкторы HF.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_model: MagicMock = MagicMock()
        fake_processor: MagicMock = MagicMock()

        monkeypatch.setattr(process_module.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            process_module.AutoModelForSpeechSeq2Seq,
            "from_pretrained",
            lambda *args, **kwargs: fake_model,
        )
        monkeypatch.setattr(
            process_module.AutoProcessor,
            "from_pretrained",
            lambda *args, **kwargs: fake_processor,
        )

        runtime = process_module.load_whisper_pipeline()

        assert runtime.model is fake_model
        assert runtime.processor is fake_processor
        fake_model.to.assert_called_once_with("cpu")

    def test_load_audio_array_for_asr_stereo_to_mono(
        self,
        process_module: ModuleType,
    ) -> None:
        """Проверяет преобразование stereo WAV в mono массив.

        Returns:
            `None`.
        """
        stereo: np.ndarray = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        buf: io.BytesIO = io.BytesIO()
        import soundfile as sf

        sf.write(buf, stereo, 16000, format="WAV")
        buf.seek(0)

        audio_array: np.ndarray
        sample_rate: int
        audio_array, sample_rate = process_module._load_audio_array_for_asr(buf)

        assert sample_rate == 16000
        assert audio_array.shape == (2,)
        np.testing.assert_allclose(
            audio_array,
            np.array([0.3, 0.7], dtype=np.float32),
            atol=2e-5,
        )

    def test_load_audio_array_for_asr_wraps_exception(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет оборачивание ошибки чтения аудио в `RuntimeError`.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        monkeypatch.setattr(process_module.sf, "read", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")))

        with pytest.raises(RuntimeError, match="Failed to load audio"):
            process_module._load_audio_array_for_asr("/bad.wav")

    def test_transcribe_audio_with_pipe(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет прямой вызов `model.generate()` и `batch_decode()`.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_processor: MagicMock = MagicMock()
        fake_model: MagicMock = MagicMock()
        fake_model.device = "cpu"
        fake_features: MagicMock = MagicMock()
        fake_features.to.return_value = fake_features
        fake_processor.return_value = SimpleNamespace(input_features=fake_features)
        fake_model.generate.return_value = [[1, 2, 3]]
        fake_processor.batch_decode.return_value = ["decoded text"]
        runtime = process_module.WhisperRuntime(
            model=fake_model,
            processor=fake_processor,
            device="cpu",
        )
        monkeypatch.setattr(
            process_module,
            "_load_audio_array_for_asr",
            lambda audio_source: (np.zeros(32, dtype=np.float32), 16000),
        )

        result: dict[str, object] = process_module._transcribe_audio_with_pipe(
            runtime,
            "/fake.wav",
        )

        assert result == {"text": "decoded text"}
        fake_model.generate.assert_called_once()

    def test_transcribe_audio_with_pipe_wraps_errors(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет оборачивание ошибок генерации Whisper.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_processor: MagicMock = MagicMock(side_effect=ValueError("oops"))
        runtime = process_module.WhisperRuntime(
            model=MagicMock(),
            processor=fake_processor,
            device="cpu",
        )
        monkeypatch.setattr(
            process_module,
            "_load_audio_array_for_asr",
            lambda audio_source: (np.zeros(8, dtype=np.float32), 16000),
        )

        with pytest.raises(RuntimeError, match="Whisper transcription failed"):
            process_module._transcribe_audio_with_pipe(runtime, "/fake.wav")


class TestExtractionAndMerging:
    """Проверки шагов извлечения аудио и объединения результатов."""

    def test_extract_audio_bytes(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет конвертацию видео-байтов в WAV-байты через subprocess.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        monkeypatch.setattr(
            process_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout=b"wav-data"),
        )

        result: bytes = process_module.extract_audio_bytes(b"video")

        assert result == b"wav-data"

    def test_extract_audio_and_transcribe(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        temp_audio_dir: Path,
    ) -> None:
        """Проверяет disk-based путь извлечения аудио и транскрибации.

        Args:
            monkeypatch: Фикстура monkeypatch.
            temp_audio_dir: Временная директория.

        Returns:
            `None`.
        """
        fake_ffmpeg: MagicMock = MagicMock()
        monkeypatch.setitem(sys.modules, "ffmpeg", fake_ffmpeg)
        monkeypatch.setattr(
            process_module,
            "_transcribe_audio_with_pipe",
            lambda pipe, audio_source: {"text": "hello"},
        )

        output_audio: str
        result: dict[str, object]
        output_audio, result = process_module.extract_audio_and_transcribe(
            "/tmp/file.webm",
            "file",
            str(temp_audio_dir),
            MagicMock(),
        )

        assert output_audio.endswith("file.wav")
        assert result["text"] == "hello"

    def test_assess_and_merge_voice_metrics(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет объединение voice metrics с данными диаризации.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        monkeypatch.setattr(
            process_module,
            "assess_voices",
            lambda data, audio_path: {
                "SPEAKER_00": {"pitch_hz_median": 120},
                "SPEAKER_99": {"pitch_hz_median": 200},
            },
        )
        data: dict[str, object] = {
            "diarization": {"SPEAKER_00": {"segments": []}}
        }

        result: dict[str, dict[str, object]] = process_module.assess_and_merge_voice_metrics(
            data,
            "/tmp/audio.wav",
        )

        assert result["SPEAKER_00"]["pitch_hz_median"] == 120
        assert data["diarization"]["SPEAKER_00"]["voice_metrics"]["pitch_hz_median"] == 120

    def test_merge_voice_descriptions(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет запись описаний голоса в диаризацию.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        monkeypatch.setattr(
            process_module,
            "describe_all_voices_with_qwen",
            lambda metrics: {"SPEAKER_00": "low voice", "SPEAKER_99": "skip"},
        )
        data: dict[str, object] = {
            "diarization": {"SPEAKER_00": {"segments": []}}
        }

        process_module.merge_voice_descriptions_into_diarization(
            data,
            {"SPEAKER_00": {"pitch_hz_median": 100}},
        )

        assert data["diarization"]["SPEAKER_00"]["voice_description"] == "low voice"


class TestProcessingFlows:
    """Проверки полных обработчиков одного файла и batch-режима."""

    def test_process_video_file(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        temp_audio_dir: Path,
    ) -> None:
        """Проверяет полную disk-based обработку и сохранение JSON.

        Args:
            monkeypatch: Фикстура monkeypatch.
            temp_audio_dir: Временная директория.

        Returns:
            `None`.
        """
        monkeypatch.setattr(
            process_module,
            "extract_audio_and_transcribe",
            lambda *args: ("/tmp/audio.wav", {"text": "hello"}),
        )
        monkeypatch.setattr(
            process_module,
            "get_diarization",
            lambda output_audio: {"SPEAKER_00": {"segments": [(0.0, 1.0)]}},
        )
        monkeypatch.setattr(
            process_module,
            "assess_and_merge_voice_metrics",
            lambda data, output_audio: {"SPEAKER_00": {"pitch_hz_median": 100}},
        )
        monkeypatch.setattr(
            process_module,
            "merge_voice_descriptions_into_diarization",
            lambda data, metrics_by_speaker: data["diarization"]["SPEAKER_00"].update(
                {"voice_description": "low"}
            ),
        )

        process_module.process_video_file(
            "/tmp/in.webm",
            "sample",
            str(temp_audio_dir),
            str(temp_audio_dir),
            MagicMock(),
        )

        payload: dict[str, object] = json.loads(
            (temp_audio_dir / "sample.json").read_text(encoding="utf-8")
        )
        assert payload["transcription"] == "hello"
        assert payload["diarization"]["SPEAKER_00"]["voice_description"] == "low"

    def test_process_video_bytes_with_local_save(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        temp_audio_dir: Path,
    ) -> None:
        """Проверяет in-memory обработку и локальное сохранение JSON.

        Args:
            monkeypatch: Фикстура monkeypatch.
            temp_audio_dir: Временная директория.

        Returns:
            `None`.
        """
        logs: list[str] = []
        monkeypatch.setattr(process_module, "extract_audio_bytes", lambda video_bytes: b"wav")
        monkeypatch.setattr(
            process_module,
            "_transcribe_audio_with_pipe",
            lambda pipe, audio_source: {"text": "text"},
        )
        monkeypatch.setattr(
            process_module,
            "get_diarization",
            lambda audio_source: {"SPEAKER_00": {"segments": [(0.0, 1.0)]}},
        )
        monkeypatch.setattr(
            process_module,
            "assess_and_merge_voice_metrics",
            lambda data, output_audio: {"SPEAKER_00": {"pitch_hz_median": 100}},
        )
        monkeypatch.setattr(
            process_module,
            "merge_voice_descriptions_into_diarization",
            lambda data, metrics_by_speaker: data["diarization"]["SPEAKER_00"].update(
                {"voice_description": "desc"}
            ),
        )

        result: dict[str, object] = process_module.process_video_bytes(
            b"video",
            "sample",
            str(temp_audio_dir),
            MagicMock(),
            progress_callback=logs.append,
            save_output_locally=True,
        )

        assert result["transcription"] == "text"
        assert (temp_audio_dir / "sample.json").exists()
        assert "Transcribing with Whisper" in logs

    def test_process_video_bytes_without_local_save(
        self,
        process_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        temp_audio_dir: Path,
    ) -> None:
        """Проверяет отключение локального сохранения итогового JSON.

        Args:
            monkeypatch: Фикстура monkeypatch.
            temp_audio_dir: Временная директория.

        Returns:
            `None`.
        """
        monkeypatch.setattr(process_module, "extract_audio_bytes", lambda video_bytes: b"wav")
        monkeypatch.setattr(
            process_module,
            "_transcribe_audio_with_pipe",
            lambda pipe, audio_source: {"text": "text"},
        )
        monkeypatch.setattr(process_module, "get_diarization", lambda audio_source: {})
        monkeypatch.setattr(
            process_module,
            "assess_and_merge_voice_metrics",
            lambda data, output_audio: {},
        )
        monkeypatch.setattr(
            process_module,
            "merge_voice_descriptions_into_diarization",
            lambda data, metrics_by_speaker: None,
        )

        process_module.process_video_bytes(
            b"video",
            "sample",
            str(temp_audio_dir),
            MagicMock(),
            save_output_locally=False,
        )

        assert not (temp_audio_dir / "sample.json").exists()

    def test_get_data_processes_only_files(
        self,
        process_module,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет batch-обработку только файлов в локальной директории.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        processed: list[tuple[str, str]] = []
        monkeypatch.setattr(process_module.os, "makedirs", lambda *args, **kwargs: None)
        monkeypatch.setattr(process_module, "load_whisper_pipeline", lambda: MagicMock())
        monkeypatch.setattr(process_module.os, "listdir", lambda path: ["a.webm", "dir"])
        monkeypatch.setattr(
            process_module.os.path,
            "isfile",
            lambda path: str(path).endswith(".webm"),
        )
        monkeypatch.setattr(
            process_module,
            "process_video_file",
            lambda video_file, file_name, audio_path, data_path, pipe: processed.append(
                (video_file, file_name)
            ),
        )

        process_module.get_data("/input")

        assert processed == [("/input/a.webm", "a")]
