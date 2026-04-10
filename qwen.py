from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, Mapping, Optional

def build_voice_description_prompt(
    voice_metrics: Mapping[str, Any],
    *,
    speaker: Optional[str] = None,
) -> str:
    """Генерирует промпт для Qwen по числовым метрикам голоса.

    Ожидаемый формат метрик — результат voice_assessment.assess_voices()[speaker].
    В промпт входят инструкции по интерпретации и блок «Короткое резюме».

    Args:
        voice_metrics: Словарь метрик (duration_sec, pitch_hz_*, loudness_*, timbre_* и т.д.).
        speaker: Опциональный идентификатор спикера для подписи в промпте.

    Returns:
        Строка промпта для LLM.
    """
    spk: str = f" ({speaker})" if speaker else ""

    def g(key: str) -> Any:
        """Возвращает значение метрики или `None`, если ключ отсутствует.

        Args:
            key: Имя метрики в словаре `voice_metrics`.

        Returns:
            Значение метрики или `None`.
        """
        return voice_metrics.get(key, None)

    return (
        "Ты — эксперт по фонетике и акустике речи. "
        "По числовым акустическим метрикам сформируй понятное текстовое описание голоса человека.\n\n"
        f"Голос{spk}. Метрики:\n"
        f"- duration_sec: {g('duration_sec')}\n"
        f"- num_segments: {g('num_segments')}\n"
        f"- pitch_hz_median: {g('pitch_hz_median')}\n"
        f"- pitch_hz_p10: {g('pitch_hz_p10')}\n"
        f"- pitch_hz_p90: {g('pitch_hz_p90')}\n"
        f"- intonation_range_st_p10_p90: {g('intonation_range_st_p10_p90')}\n"
        f"- speech_rate_syllables_per_sec: {g('speech_rate_syllables_per_sec')}\n"
        f"- speech_rate_wpm_est: {g('speech_rate_wpm_est')}\n"
        f"- loudness_dbfs_mean: {g('loudness_dbfs_mean')}\n"
        f"- loudness_dbfs_p90: {g('loudness_dbfs_p90')}\n"
        f"- timbre_centroid_hz_mean: {g('timbre_centroid_hz_mean')}\n"
        f"- timbre_rolloff_hz_mean: {g('timbre_rolloff_hz_mean')}\n"
        f"- timbre_flatness_mean: {g('timbre_flatness_mean')}\n\n"
        "Интерпретация:\n"
        "- Тембр: опиши как более «тёплый/тёмный» vs «яркий/светлый» (ориентируйся на centroid/rolloff), "
        "а также как «шумный/шероховатый» vs «чистый/гармонический» (ориентируйся на flatness).\n"
        "- Высота: опиши как низкая/средняя/высокая на основе pitch_hz_median.\n"
        "- Интонационный диапазон: опиши как узкий/средний/широкий на основе intonation_range_st_p10_p90 "
        "(в полутонах).\n"
        "- Темп речи: опиши как медленный/средний/быстрый на основе speech_rate_wpm_est "
        "(это грубая оценка по акустическим признакам).\n"
        "- Громкость: опиши как тихая/средняя/громкая на основе loudness_dbfs_mean и loudness_dbfs_p90.\n\n"
        "Формат ответа (строго):\n"
        # "1) Тембр: ...\n"
        # "2) Высота: ...\n"
        # "3) Интонационный диапазон: ...\n"
        # "4) Темп речи: ...\n"
        # "5) Громкость: ...\n"
        # "6) 
        "Короткое резюме (1-2 предложения): ...\n"
    )


def describe_voice_with_qwen(
    voice_metrics: Mapping[str, Any],
    *,
    speaker: Optional[str] = None,
    model_path: str = "qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx: int = 4096,
    n_threads: int = 8,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Запускает локальную Qwen GGUF модель через llama-cli и возвращает текстовое описание голоса.

    Строит промпт через build_voice_description_prompt, вызывает subprocess (llama-cli),
    из ответа извлекает блок после маркера «Короткое резюме» или обрезает ответ до max_len.

    Args:
        voice_metrics: Словарь метрик голоса.
        speaker: Опциональный идентификатор спикера.
        model_path: Путь к GGUF-модели.
        n_ctx: Размер контекста.
        n_threads: Число потоков.
        temperature: Температура сэмплирования.
        max_tokens: Максимум токенов ответа.

    Returns:
        Текстовое описание голоса или обрезанный ответ. Пустая строка при пустом stdout.

    Raises:
        RuntimeError: При ненулевом returncode llama-cli (в сообщении stderr).
    """
    prompt: str = build_voice_description_prompt(voice_metrics, speaker=speaker)
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        [
            "llama-cli",
            "-m", model_path,
            "-p", prompt,
            "--threads", str(n_threads),
            "--ctx-size", str(n_ctx),
            "--temp", str(temperature),
            "--n-predict", str(max_tokens),
            "--simple-io",
            "--log-disable",
            "--log-verbosity", "1",
            "--no-display-prompt",
            "-no-cnv",
            "--single-turn",
            "--no-show-timings",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli - error: {proc.stderr}")
    result: str = proc.stdout.strip()
    if not result:
        return ""

    def _is_plausible_description(text: str) -> bool:
        """Проверяет, похож ли фрагмент на итоговое голосовое описание.

        Args:
            text: Кандидат на краткое описание голоса.

        Returns:
            `True`, если текст выглядит как короткое описание, а не утечка промпта.
        """
        if not text or len(text) > 800:
            return False
        prompt_leak: tuple[str, ...] = ("pitch_hz", "duration_sec", "Метрики:", "Интерпретация:")
        return not any(leak in text for leak in prompt_leak)

    markers: tuple[str, ...] = ("Короткое резюме (1-2 предложения):", "Короткое резюме:")
    marker: str
    for marker in markers:
        voice_desc: str | None = try_extract_description_after_marker(
            result, marker, _is_plausible_description
        )
        if voice_desc is not None:
            return voice_desc

    return _truncate_safe_result(result, max_len=1000)


def _truncate_safe_result(result: str, max_len: int = 1000) -> str:
    """Обрезает строку до max_len символов и добавляет суффикс «...».

    Args:
        result: Исходная строка.
        max_len: Максимальная длина до обрезки (суффикс «...» добавляется после обрезки).

    Returns:
        Пустая строка для пустого result, иначе result либо result[:max_len].rstrip() + «...».
    """
    if not result:
        return ""
    if len(result) <= max_len:
        return result
    return result[:max_len].rstrip() + "..."


def try_extract_description_after_marker(
    result: str,
    marker: str,
    is_plausible: Callable[[str], bool],
) -> Optional[str]:
    """Извлекает описание из ответа LLM после указанного маркера (1–2 строки) с проверкой через is_plausible.

    Ищет marker в result, берёт текст после него, собирает до двух непустых строк в одну,
    передаёт в is_plausible; при успехе возвращает эту строку.

    Args:
        result: Полный ответ LLM.
        marker: Строка-маркер (например, «Короткое резюме:»).
        is_plausible: Функция str -> bool; возвращает True, если текст подходит как описание.

    Returns:
        Извлечённая и проверенная строка описания или None.
    """
    if marker not in result:
        return None
    parts: list[str] = result.split(marker, 1)
    if len(parts) != 2:
        return None
    after: str = parts[1]
    lines: list[str] = [ln.strip() for ln in after.split("\n") if ln.strip()][:2]
    voice_desc: str = " ".join(lines).strip()
    if voice_desc and is_plausible(voice_desc):
        return voice_desc
    return None


def describe_all_voices_with_qwen(
    metrics_by_speaker: Mapping[str, Mapping[str, Any]],
    *,
    model_path: str = "qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx: int = 4096,
    n_threads: int = 8,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> dict[str, str]:
    """Получает текстовое описание голоса для каждого спикера через describe_voice_with_qwen.

    Для каждой пары (speaker, metrics) вызывает describe_voice_with_qwen с теми же
    параметрами модели (model_path, n_ctx, n_threads, temperature, max_tokens).

    Args:
        metrics_by_speaker: Словарь {speaker: voice_metrics}.
        model_path: Путь к GGUF-модели.
        n_ctx: Размер контекста.
        n_threads: Число потоков.
        temperature: Температура.
        max_tokens: Максимум токенов.

    Returns:
        Словарь {speaker: text_description}.
    """
    out: dict[str, str] = {}
    for speaker, metrics in metrics_by_speaker.items():
        out[str(speaker)] = describe_voice_with_qwen(
            metrics,
            speaker=str(speaker),
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return out


if __name__ == "__main__":
    import os

    output_data_dir: str = "./output_data"
    if not os.path.isdir(output_data_dir):
        print(f"Папка не найдена: {output_data_dir}")
        raise SystemExit(1)

    json_files: list[str] = sorted(
        f for f in os.listdir(output_data_dir) if f.endswith(".json")
    )
    if not json_files:
        print("В папке нет .json файлов.")
        raise SystemExit(1)

    filename: str = json_files[0]
    filepath: str = os.path.join(output_data_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
    diarization: dict[str, Any] = data.get("diarization", {})

    for speaker, payload in diarization.items():
        if not isinstance(payload, dict):
            continue
        metrics: Any = payload.get("voice_metrics")
        if not metrics:
            continue
        print(f"--- тест: {filename} / {speaker} ---")
        desc: str = describe_voice_with_qwen(metrics, speaker=speaker)
        print(desc)
        break