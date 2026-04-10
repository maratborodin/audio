"""Тесты для qwen.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qwen import (
    build_voice_description_prompt,
    try_extract_description_after_marker,
    _truncate_safe_result,
    describe_voice_with_qwen,
    describe_all_voices_with_qwen,
)


class TestBuildVoiceDescriptionPrompt:
    """Проверки генерации промпта по числовым голосовым метрикам."""

    def test_empty_metrics(self) -> None:
        prompt: str = build_voice_description_prompt({})
        assert "Голос. Метрики:" in prompt
        assert "duration_sec: None" in prompt
        assert "Ты — эксперт" in prompt

    def test_with_speaker(self, sample_voice_metrics: dict[str, int | float]) -> None:
        prompt: str = build_voice_description_prompt(
            sample_voice_metrics, speaker="SPEAKER_01"
        )
        assert "Голос (SPEAKER_01)" in prompt
        assert "duration_sec: 120.5" in prompt
        assert "pitch_hz_median: 180.0" in prompt

    def test_partial_metrics(self) -> None:
        prompt: str = build_voice_description_prompt(
            {"pitch_hz_median": 200, "duration_sec": 10}
        )
        assert "pitch_hz_median: 200" in prompt
        assert "timbre_flatness_mean: None" in prompt


class TestTryExtractDescriptionAfterMarker:
    """Проверки извлечения короткого описания после маркера в ответе LLM."""

    def test_marker_not_present(self) -> None:
        assert (
            try_extract_description_after_marker(
                "no marker here", "Резюме:", lambda t: True
            )
            is None
        )

    def test_extract_two_lines(self) -> None:
        text: str = "Preamble\nКороткое резюме (1-2 предложения):\nНизкий голос.\nБыстрый темп.\nTrailing"
        result: str | None = try_extract_description_after_marker(
            text, "Короткое резюме (1-2 предложения):", lambda t: True
        )
        assert result == "Низкий голос. Быстрый темп."

    def test_plausible_rejects(self) -> None:
        text: str = "Короткое резюме:\nМетрики: 100"
        result: str | None = try_extract_description_after_marker(
            text, "Короткое резюме:", lambda t: "Метрики:" not in t
        )
        assert result is None

    def test_plausible_accepts(self) -> None:
        text: str = "Короткое резюме:\nТёплый низкий голос."
        result: str | None = try_extract_description_after_marker(
            text,
            "Короткое резюме:",
            lambda t: len(t) < 100 and "Метрики" not in t,
        )
        assert result == "Тёплый низкий голос."

    def test_empty_after_marker(self) -> None:
        text: str = "Короткое резюме:\n\n\n"
        result: str | None = try_extract_description_after_marker(
            text, "Короткое резюме:", lambda t: True
        )
        assert result is None

    def test_plausible_returns_false_for_empty(self) -> None:
        text: str = "Короткое резюме:\n  \n  "
        result: str | None = try_extract_description_after_marker(
            text, "Короткое резюме:", lambda t: bool(t.strip())
        )
        assert result is None

    def test_single_line_after_marker(self) -> None:
        text: str = "Предисловие\nКороткое резюме:\nОдин абзац."
        result: str | None = try_extract_description_after_marker(
            text, "Короткое резюме:", lambda t: True
        )
        assert result == "Один абзац."

    def test_whitespace_only_after_marker_returns_none(self) -> None:
        text: str = "Короткое резюме:\n   \n\t  "
        result: str | None = try_extract_description_after_marker(
            text, "Короткое резюме:", lambda t: True
        )
        assert result is None


class TestTruncateSafeResult:
    """Проверки безопасной усечки слишком длинного результата LLM."""

    def test_empty(self) -> None:
        assert _truncate_safe_result("") == ""
        assert _truncate_safe_result(None or "") == ""

    def test_short_string(self) -> None:
        s: str = "hello"
        assert _truncate_safe_result(s) == s
        assert _truncate_safe_result(s, max_len=10) == s

    def test_truncate(self) -> None:
        s: str = "a" * 1500
        out: str = _truncate_safe_result(s, max_len=100)
        assert len(out) == 103  # 100 + "..."
        assert out.endswith("...")
        assert out == s[:100].rstrip() + "..."

    def test_exactly_max_len_returns_unchanged(self) -> None:
        s: str = "x" * 100
        assert _truncate_safe_result(s, max_len=100) == s
        assert _truncate_safe_result(s, max_len=101) == s


class TestDescribeVoiceWithQwen:
    """Проверки запуска `llama-cli` и разбора ответа модели Qwen."""

    @patch("qwen.subprocess.run")
    def test_returns_extracted_description(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Бла бла.\nКороткое резюме (1-2 предложения):\nНизкий тёплый голос. Средний темп.",
            stderr="",
        )
        result: str = describe_voice_with_qwen(
            sample_voice_metrics, speaker="SPEAKER_00"
        )
        assert result == "Низкий тёплый голос. Средний темп."
        assert mock_run.called

    @patch("qwen.subprocess.run")
    def test_returns_truncated_on_no_marker(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        long_text: str = "X" * 1500
        mock_run.return_value = MagicMock(
            returncode=0, stdout=long_text, stderr=""
        )
        result: str = describe_voice_with_qwen(sample_voice_metrics)
        assert result.endswith("...")
        assert len(result) == 1003

    @patch("qwen.subprocess.run")
    def test_empty_stdout_returns_empty(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="  \n  ", stderr="")
        result: str = describe_voice_with_qwen(sample_voice_metrics)
        assert result == ""

    @patch("qwen.subprocess.run")
    def test_nonzero_returncode_raises(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="llama error"
        )
        with pytest.raises(RuntimeError, match="llama-cli"):
            describe_voice_with_qwen(sample_voice_metrics)

    @patch("qwen.subprocess.run")
    def test_uses_second_marker_if_first_not_found(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Some text.\nКороткое резюме:\nГолос средний.",
            stderr="",
        )
        result: str = describe_voice_with_qwen(sample_voice_metrics)
        assert result == "Голос средний."

    @patch("qwen.subprocess.run")
    def test_first_marker_rejected_by_plausible_uses_second(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Короткое резюме (1-2 предложения):\nМетрики: 100.\n\n"
                "Короткое резюме:\nТёплый низкий голос."
            ),
            stderr="",
        )
        result: str = describe_voice_with_qwen(sample_voice_metrics)
        assert result == "Тёплый низкий голос."

    @patch("qwen.subprocess.run")
    def test_extracted_text_too_long_rejected_then_truncate(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        prefix: str = "Короткое резюме (1-2 предложения):\n"
        long_desc: str = "Y" * 1000
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=prefix + long_desc,
            stderr="",
        )
        result: str = describe_voice_with_qwen(sample_voice_metrics)
        assert result.endswith("...")
        assert len(result) == 1003

    @patch("qwen.subprocess.run")
    def test_subprocess_called_with_prompt_and_model_args(
        self,
        mock_run: MagicMock,
        sample_voice_metrics: dict[str, int | float],
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Короткое резюме:\nОк.", stderr=""
        )
        describe_voice_with_qwen(
            sample_voice_metrics,
            speaker="SPK",
            model_path="custom.gguf",
            n_ctx=2048,
            max_tokens=256,
        )
        call_args: list[str] = mock_run.call_args[0][0]
        assert call_args[0] == "llama-cli"
        assert "-m" in call_args and "custom.gguf" in call_args
        assert "--ctx-size" in call_args and "2048" in call_args
        assert "--n-predict" in call_args and "256" in call_args
        assert "Голос (SPK)" in " ".join(call_args)


class TestDescribeAllVoicesWithQwen:
    """Проверки пакетной генерации описаний для нескольких спикеров."""

    @patch("qwen.describe_voice_with_qwen")
    def test_calls_per_speaker(self, mock_describe: MagicMock) -> None:
        mock_describe.side_effect = lambda m, **kw: f"desc_{kw.get('speaker', '')}"
        metrics: dict[str, dict[str, int]] = {
            "SPEAKER_00": {"pitch_hz_median": 100},
            "SPEAKER_01": {"pitch_hz_median": 200},
        }
        result: dict[str, str] = describe_all_voices_with_qwen(metrics)
        assert result == {
            "SPEAKER_00": "desc_SPEAKER_00",
            "SPEAKER_01": "desc_SPEAKER_01",
        }
        assert mock_describe.call_count == 2

    @patch("qwen.describe_voice_with_qwen")
    def test_empty_input(self, mock_describe: MagicMock) -> None:
        result: dict[str, str] = describe_all_voices_with_qwen({})
        assert result == {}
        mock_describe.assert_not_called()

    @patch("qwen.describe_voice_with_qwen")
    def test_passes_model_params_to_describe_voice(
        self, mock_describe: MagicMock
    ) -> None:
        mock_describe.return_value = "desc"
        describe_all_voices_with_qwen(
            {"S1": {"pitch_hz_median": 100}},
            model_path="m.gguf",
            n_ctx=1024,
            temperature=0.5,
        )
        mock_describe.assert_called_once()
        kwargs: dict[str, object] = mock_describe.call_args[1]
        assert kwargs["model_path"] == "m.gguf"
        assert kwargs["n_ctx"] == 1024
        assert kwargs["temperature"] == 0.5
        assert kwargs["speaker"] == "S1"


class TestQwenMain:
    """Покрытие блока if __name__ == '__main__'."""

    @patch("qwen.describe_voice_with_qwen")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_main_exits_if_output_data_missing(
        self,
        mock_isdir: MagicMock,
        mock_listdir: MagicMock,
        mock_describe: MagicMock,
    ) -> None:
        mock_isdir.return_value = False
        with pytest.raises(SystemExit):
            import runpy
            import qwen as qwen_module
            runpy.run_path(qwen_module.__file__, run_name="__main__")

    @patch("qwen.describe_voice_with_qwen")
    @patch("qwen.json.load")
    @patch("builtins.open", create=True)
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_main_exits_if_no_json_files(
        self,
        mock_isdir: MagicMock,
        mock_listdir: MagicMock,
        mock_open: MagicMock,
        mock_json: MagicMock,
        mock_describe: MagicMock,
    ) -> None:
        mock_isdir.return_value = True
        mock_listdir.return_value = ["readme.txt"]
        with pytest.raises(SystemExit):
            import runpy
            import qwen as qwen_module
            runpy.run_path(qwen_module.__file__, run_name="__main__")

    @patch("qwen.subprocess.run")
    @patch("qwen.json.load")
    @patch("builtins.open", create=True)
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_main_prints_description_for_first_speaker_with_metrics(
        self,
        mock_isdir: MagicMock,
        mock_listdir: MagicMock,
        mock_open: MagicMock,
        mock_json: MagicMock,
        mock_subprocess: MagicMock,
    ) -> None:
        # runpy.run_path перезапускает весь файл и перезаписывает describe_voice_with_qwen,
        # поэтому мокаем subprocess.run: тогда реальный describe_voice_with_qwen вызовет мок и не пойдёт в llama
        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.json"]
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda *a: None
        mock_json.return_value = {
            "diarization": {
                "SPEAKER_00": {"voice_metrics": {"pitch_hz_median": 200}, "segments": []},
            }
        }
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Короткое резюме:\nНизкий голос.",
            stderr="",
        )

        import runpy
        import qwen as qwen_module
        runpy.run_path(qwen_module.__file__, run_name="__main__")

        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]
        assert "llama-cli" in call_args
        assert "-p" in call_args
        full_cmd: str = " ".join(call_args)
        assert "SPEAKER_00" in full_cmd

    @patch("qwen.subprocess.run")
    @patch("qwen.json.load")
    @patch("builtins.open", create=True)
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_main_skips_speaker_without_voice_metrics(
        self,
        mock_isdir: MagicMock,
        mock_listdir: MagicMock,
        mock_open: MagicMock,
        mock_json: MagicMock,
        mock_subprocess: MagicMock,
    ) -> None:
        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.json"]
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda *a: None
        mock_json.return_value = {
            "diarization": {
                "SPEAKER_00": {"segments": []},
                "SPEAKER_01": {"voice_metrics": {"pitch_hz_median": 100}, "segments": []},
            }
        }
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Короткое резюме:\nГолос.",
            stderr="",
        )

        import runpy
        import qwen as qwen_module
        runpy.run_path(qwen_module.__file__, run_name="__main__")

        mock_subprocess.assert_called_once()
        full_cmd = " ".join(mock_subprocess.call_args[0][0])
        assert "SPEAKER_01" in full_cmd
