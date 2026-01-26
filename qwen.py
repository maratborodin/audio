from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def build_voice_description_prompt(
    voice_metrics: Mapping[str, Any],
    *,
    speaker: Optional[str] = None,
) -> str:
    """
    Генерирует промпт для Qwen по числовым метрикам голоса.
    Ожидаемый формат метрик — результат voice_assessment.assess_voices()[speaker].
    """
    spk = f" ({speaker})" if speaker else ""

    # Safely pick fields (None-safe stringification).
    def g(key: str) -> Any:
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
        "1) Тембр: ...\n"
        "2) Высота: ...\n"
        "3) Интонационный диапазон: ...\n"
        "4) Темп речи: ...\n"
        "5) Громкость: ...\n"
        "6) Короткое резюме (1-2 предложения): ...\n"
    )


def describe_voice_with_qwen(
    voice_metrics: Mapping[str, Any],
    *,
    model_path: str,
    speaker: Optional[str] = None,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> str:
    """
    Запускает локальную Qwen GGUF модель через llama-cpp-python и возвращает текстовое описание.

    Требования:
      pip install llama-cpp-python
      model_path: путь к .gguf (например Qwen2.5-7B-Instruct-*.gguf)
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        raise ImportError(
            "Не найден пакет 'llama-cpp-python'. Установите: pip install llama-cpp-python"
        ) from e

    prompt = build_voice_description_prompt(voice_metrics, speaker=speaker)
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)

    # Qwen-instruct models generally work well via chat completion API
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Ты отвечаешь по-русски, кратко и по делу."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]["content"]


def describe_all_voices_with_qwen(
    metrics_by_speaker: Mapping[str, Mapping[str, Any]],
    *,
    model_path: str,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> Dict[str, str]:
    """
    Для удобства: принимает {speaker: metrics} и возвращает {speaker: text_description}.
    Модель загружается один раз.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        raise ImportError(
            "Не найден пакет 'llama-cpp-python'. Установите: pip install llama-cpp-python"
        ) from e

    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
    out: Dict[str, str] = {}
    for speaker, metrics in metrics_by_speaker.items():
        prompt = build_voice_description_prompt(metrics, speaker=str(speaker))
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Ты отвечаешь по-русски, кратко и по делу."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out[str(speaker)] = resp["choices"][0]["message"]["content"]
    return out