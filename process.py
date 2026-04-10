from __future__ import annotations

import io
import json
import logging
import os
import subprocess
from typing import Any, Callable

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from diarization import get_diarization
from qwen import describe_all_voices_with_qwen
from voice_assessment import assess_voices

logger: logging.Logger = logging.getLogger(__name__)

# Paths used by local-directory workflow (kept for backward compatibility)
DEFAULT_AUDIO_PATH: str = "./output_audio"
DEFAULT_DATA_PATH: str = "./output_data"


class WhisperRuntime:
    """Контейнер с компонентами Whisper для прямого вызова `model.generate()`.

    Attributes:
        model: Загруженная HuggingFace-модель Whisper.
        processor: Процессор Whisper для подготовки входов и декодирования.
        device: Идентификатор устройства выполнения (`cpu`, `cuda:0` и т.д.).
    """

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        """Инициализирует runtime-компоненты Whisper.

        Args:
            model: Загруженная модель Whisper.
            processor: Процессор Whisper.
            device: Устройство выполнения модели.
        """
        self.model: Any = model
        self.processor: Any = processor
        self.device: str = device


# ---------------------------------------------------------------------------
# Whisper pipeline loader (reusable singleton helper)
# ---------------------------------------------------------------------------


def load_whisper_pipeline(
    model_id: str = "openai/whisper-large-v3-turbo",
) -> WhisperRuntime:
    """Загружает модель Whisper и возвращает runtime для ASR.

    Определяет устройство (CUDA / CPU) и dtype автоматически. Вызов занимает
    несколько минут при первом запуске из-за загрузки весов.

    Args:
        model_id: Идентификатор модели HuggingFace.

    Returns:
        Объект `WhisperRuntime` с моделью, processor и устройством.
    """
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model: Any = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor: Any = AutoProcessor.from_pretrained(model_id)

    return WhisperRuntime(model=model, processor=processor, device=device)


# ---------------------------------------------------------------------------
# In-memory audio extraction
# ---------------------------------------------------------------------------


def extract_audio_bytes(video_bytes: bytes) -> bytes:
    """Конвертирует видеофайл из байт в WAV-байты через ffmpeg (pipe, без диска).

    ffmpeg получает видео через stdin и отдаёт моно PCM 16 kHz WAV в stdout.
    При ошибке ffmpeg бросает subprocess.CalledProcessError.

    Args:
        video_bytes: Содержимое видеофайла в памяти.

    Returns:
        Байты WAV-файла (PCM s16le, 16 kHz, mono).

    Raises:
        subprocess.CalledProcessError: Если ffmpeg завершился с ненулевым кодом.
    """
    proc: subprocess.CompletedProcess[bytes] = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", "pipe:0",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            "pipe:1",
        ],
        input=video_bytes,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def _load_audio_array_for_asr(audio_source: str | io.IOBase) -> tuple[Any, int]:
    """Загружает WAV в mono float32 массив для Whisper.

    Args:
        audio_source: Путь к WAV или file-like объект.

    Returns:
        Кортеж `(audio_array, sample_rate)`.

    Raises:
        RuntimeError: Если `soundfile` не смог прочитать входной WAV.
    """
    try:
        audio_array: Any
        sample_rate: int
        audio_array, sample_rate = sf.read(audio_source, dtype="float32")
    except Exception as exc:
        raise RuntimeError("Failed to load audio for ASR") from exc
    if getattr(audio_array, "ndim", 1) == 2:
        audio_array = audio_array.mean(axis=1)
    return audio_array, int(sample_rate)


def _transcribe_audio_with_pipe(
    pipe: WhisperRuntime,
    audio_source: str | io.IOBase,
) -> dict[str, Any]:
    """Транскрибирует уже декодированное аудио прямым вызовом Whisper model.generate().

    Такой вызов обходит внутренний загрузчик `torchcodec` в transformers, который
    может падать на macOS из-за отсутствующих FFmpeg dylib.

    Args:
        pipe: Объект `WhisperRuntime`.
        audio_source: Путь к WAV или file-like объект.

    Returns:
        Результат ASR в формате `{"text": ...}`.

    Raises:
        RuntimeError: Если декодирование или генерация Whisper завершились с ошибкой.
    """
    audio_array: Any
    sample_rate: int
    audio_array, sample_rate = _load_audio_array_for_asr(audio_source)
    processor: Any = pipe.processor
    model: Any = pipe.model

    try:
        inputs: Any = processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features: torch.Tensor = inputs.input_features.to(model.device)

        with torch.no_grad():
            generated_ids: Any = model.generate(
                input_features,
                task="transcribe",
            )

        text: str = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
    except Exception as exc:
        raise RuntimeError("Whisper transcription failed") from exc
    return {"text": text}


# ---------------------------------------------------------------------------
# Core processing functions (disk-based, used by local workflow)
# ---------------------------------------------------------------------------


def extract_audio_and_transcribe(
    video_file: str,
    file_name: str,
    audio_path: str,
    pipe: WhisperRuntime,
) -> tuple[str, dict[str, Any]]:
    """Извлекает аудио из видео в WAV и выполняет распознавание речи.

    Конвертирует видео в моно WAV 16 kHz через ffmpeg и прогоняет аудио через
    пайплайн ASR (Whisper). Перезаписывает существующий файл при наличии.

    Args:
        video_file: Путь к файлу видео.
        file_name: Имя файла без расширения (используется для выходного WAV).
        audio_path: Директория для сохранения WAV-файла.
        pipe: Runtime Whisper для распознавания речи.

    Returns:
        Кортеж (путь к выходному WAV, результат pipe — dict с ключом 'text').

    Raises:
        ffmpeg.Error: Если ffmpeg не смог извлечь аудио.
        RuntimeError: Если Whisper не смог выполнить транскрибацию.
    """
    import ffmpeg as _ffmpeg  # local import to avoid circular at module level

    output_audio: str = os.path.join(audio_path, f"{file_name}.wav")
    (
        _ffmpeg.input(video_file)
        .output(output_audio, vn=None, acodec="pcm_s16le", ar=16000, ac=1)
        .overwrite_output()
        .run()
    )
    result: dict[str, Any] = _transcribe_audio_with_pipe(pipe, output_audio)
    return output_audio, result


def assess_and_merge_voice_metrics(
    data: dict[str, Any],
    output_audio: str | io.IOBase,
) -> dict[str, dict[str, Any]]:
    """Оценивает голоса спикеров и записывает метрики в данные диаризации.

    Вызывает assess_voices по data и output_audio, затем для каждого спикера
    записывает его метрики в data['diarization'][speaker]['voice_metrics'].
    Спикеры, отсутствующие в data['diarization'] или с не-dict значением, пропускаются.

    Args:
        data: Словарь с ключом 'diarization' (спикер -> dict с 'segments').
        output_audio: Путь к WAV-файлу или file-like объект (BytesIO).

    Returns:
        Словарь {speaker: metrics} из assess_voices.
    """
    metrics_by_speaker: dict[str, dict[str, Any]] = assess_voices(
        data, audio_path=output_audio
    )
    speaker: str
    metrics: dict[str, Any]
    for speaker, metrics in metrics_by_speaker.items():
        if speaker in data["diarization"] and isinstance(
            data["diarization"][speaker], dict
        ):
            data["diarization"][speaker]["voice_metrics"] = metrics
    return metrics_by_speaker


def merge_voice_descriptions_into_diarization(
    data: dict[str, Any],
    metrics_by_speaker: dict[str, dict[str, Any]],
) -> None:
    """Получает текстовые описания голосов через Qwen и записывает их в данные диаризации.

    Вызывает describe_all_voices_with_qwen(metrics_by_speaker) и для каждого спикера
    записывает описание в data['diarization'][speaker]['voice_description'].
    Спикеры, отсутствующие в data['diarization'] или с не-dict значением, пропускаются.

    Args:
        data: Словарь с ключом 'diarization' (изменяется на месте).
        metrics_by_speaker: Словарь {speaker: voice_metrics} для LLM-описания.
    """
    descriptions: dict[str, str] = describe_all_voices_with_qwen(metrics_by_speaker)
    speaker: str
    desc: str
    for speaker, desc in descriptions.items():
        if speaker in data["diarization"] and isinstance(
            data["diarization"][speaker], dict
        ):
            data["diarization"][speaker]["voice_description"] = desc


def process_video_file(
    video_file: str,
    file_name: str,
    audio_path: str,
    data_path: str,
    pipe: Any,
) -> None:
    """Обрабатывает один видеофайл: транскрипция, диаризация, оценка голосов, LLM-описание и сохранение в JSON.

    Последовательно: извлекает аудио и транскрибирует; выполняет диаризацию;
    считает метрики голосов и записывает их в данные; получает текстовые описания
    голосов через Qwen и записывает в данные; сохраняет итоговый словарь в
    {data_path}/{file_name}.json.

    Args:
        video_file: Путь к файлу видео.
        file_name: Имя файла без расширения.
        audio_path: Директория для временного WAV.
        data_path: Директория для выходного JSON.
        pipe: Пайплайн ASR (Whisper).
    """
    output_audio: str
    result: dict[str, Any]
    output_audio, result = extract_audio_and_transcribe(
        video_file, file_name, audio_path, pipe
    )
    diarization_data: dict[str, Any] = get_diarization(output_audio)

    data: dict[str, Any] = {
        "transcription": result["text"],
        "diarization": diarization_data,
    }

    metrics_by_speaker: dict[str, dict[str, Any]] = assess_and_merge_voice_metrics(
        data, output_audio
    )
    merge_voice_descriptions_into_diarization(data, metrics_by_speaker)

    output_data_file: str = os.path.join(data_path, f"{file_name}.json")
    with open(output_data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


# ---------------------------------------------------------------------------
# In-memory processing (used by S3 / web workflow)
# ---------------------------------------------------------------------------


def process_video_bytes(
    video_bytes: bytes,
    file_name: str,
    data_path: str,
    pipe: WhisperRuntime,
    *,
    progress_callback: Callable[[str], None] | None = None,
    save_output_locally: bool = True,
) -> dict[str, Any]:
    """Обрабатывает видеофайл из байт в памяти: без сохранения видео/WAV на диск.

    Конвертирует видео в WAV через ffmpeg (pipe), транскрибирует через Whisper,
    выполняет диаризацию и оценку голоса через BytesIO. Сохраняет результат в
    {data_path}/{file_name}.json.

    Args:
        video_bytes: Байты видеофайла (скачаны из S3 или другого источника).
        file_name: Имя файла без расширения — используется для выходного JSON.
        data_path: Директория для выходного JSON.
        pipe: Runtime Whisper для ASR.
        progress_callback: Опциональная функция (message) -> None для логирования.
        save_output_locally: Нужно ли сохранять итоговый JSON на локальный диск.

    Returns:
        Итоговый словарь с ключами 'transcription' и 'diarization'.

    Raises:
        subprocess.CalledProcessError: При ошибке ffmpeg.
        RuntimeError: Если шаги транскрибации, диаризации или оценки голоса завершились с ошибкой.
    """

    def _log(msg: str) -> None:
        """Пишет шаг обработки в стандартный лог и callback прогресса.

        Args:
            msg: Сообщение о текущем шаге обработки.
        """
        logger.info("[%s] %s", file_name, msg)
        if progress_callback is not None:
            progress_callback(msg)

    _log("Extracting audio via ffmpeg pipe")
    wav_bytes: bytes = extract_audio_bytes(video_bytes)
    _log(f"WAV ready: {len(wav_bytes)} bytes")

    # Transcription — read WAV bytes into numpy array, no temp file needed
    _log("Transcribing with Whisper")
    asr_result: dict[str, Any] = _transcribe_audio_with_pipe(
        pipe,
        io.BytesIO(wav_bytes),
    )
    _log("Transcription done")

    # Diarization — pass BytesIO directly (soundfile handles file-like objects)
    _log("Running speaker diarization")
    diarization_data: dict[str, Any] = get_diarization(io.BytesIO(wav_bytes))
    _log(f"Diarization done: {len(diarization_data)} speakers")

    data: dict[str, Any] = {
        "transcription": asr_result["text"],
        "diarization": diarization_data,
    }

    # Voice assessment — pass BytesIO (assess_voices / sf.read accept file-like)
    _log("Assessing voice metrics")
    metrics_by_speaker: dict[str, dict[str, Any]] = assess_and_merge_voice_metrics(
        data, io.BytesIO(wav_bytes)
    )
    _log("Voice metrics done")

    _log("Generating voice descriptions with Qwen")
    merge_voice_descriptions_into_diarization(data, metrics_by_speaker)
    _log("Qwen descriptions done")

    if save_output_locally:
        os.makedirs(data_path, exist_ok=True)
        out_path: str = os.path.join(data_path, f"{file_name}.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(data, ensure_ascii=False, indent=4))
        _log(f"Result saved to {out_path}")

    return data


# ---------------------------------------------------------------------------
# Local directory workflow (backward-compatible entry point)
# ---------------------------------------------------------------------------


def get_data(video_path: str) -> None:
    """Обрабатывает все видеофайлы в директории: загрузка Whisper, цикл по файлам, вызов process_video_file.

    Создаёт директории DEFAULT_AUDIO_PATH и DEFAULT_DATA_PATH при необходимости.
    Загружает модель openai/whisper-large-v3-turbo и пайплайн ASR (на GPU при наличии CUDA).
    Для каждого элемента в video_path, являющегося файлом, вызывает process_video_file.

    Args:
        video_path: Путь к директории с видеофайлами.
    """
    os.makedirs(DEFAULT_AUDIO_PATH, exist_ok=True)
    os.makedirs(DEFAULT_DATA_PATH, exist_ok=True)

    pipe: Any = load_whisper_pipeline()

    item: str
    for item in os.listdir(video_path):
        video_file: str = os.path.join(video_path, item)
        file_name: str = item.split(".")[0]
        if not os.path.isfile(video_file):
            continue
        logger.info("Processing %s", video_file)
        process_video_file(
            video_file, file_name, DEFAULT_AUDIO_PATH, DEFAULT_DATA_PATH, pipe
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_data("./input_video/")
