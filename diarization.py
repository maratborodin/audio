from __future__ import annotations

import os
from typing import Any, IO

import soundfile
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv()
TOKEN: str | None = os.environ.get("TOKEN")


def group_segments_by_speaker(
    segments: list[dict[str, Any]],
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Группирует сегменты диаризации по спикерам.

    Преобразует список сегментов с полями 'speaker', 'start', 'end' в словарь,
    где каждому спикеру соответствует список пар (start, end).

    Args:
        segments: Список dict с ключами 'speaker', 'start', 'end'.

    Returns:
        Словарь {speaker: {'segments': [(start, end), ...]}}.
    """
    speakers: dict[str, dict[str, list[tuple[float, float]]]] = {}
    segment: dict[str, Any]
    for segment in segments:
        if segment["speaker"] not in speakers:
            speakers[segment["speaker"]] = {"segments": []}
        speakers[segment["speaker"]]["segments"].append((segment["start"], segment["end"]))
    return speakers


def get_diarization(
    output_audio: str | IO[bytes],
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Выполняет диаризацию спикеров по аудиофайлу.

    Загружает pyannote speaker-diarization-community-1, читает WAV через soundfile,
    передаёт waveform в пайплайн на CPU и группирует результат по спикерам через
    group_segments_by_speaker.

    Args:
        output_audio: Путь к моно WAV-файлу (float32) или file-like объект (BytesIO).

    Returns:
        Словарь {speaker: {'segments': [(start, end), ...]}} в формате
        group_segments_by_speaker.

    Raises:
        ValueError: Если пайплайн диаризации не был загружен.
    """
    pipeline_or_none: Pipeline | None = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=TOKEN
    )
    if pipeline_or_none is None:
        raise ValueError("Failed to load diarization pipeline")
    pipeline: Pipeline = pipeline_or_none
    device: torch.device = torch.device("cpu")
    pipeline.to(device)
    wave_form_np: Any
    sample_rate: Any
    wave_form_np, sample_rate = soundfile.read(output_audio, dtype="float32")
    wave_form: torch.Tensor = torch.from_numpy(wave_form_np).unsqueeze(0)
    audio_dict: dict[str, Any] = {"waveform": wave_form, "sample_rate": sample_rate}
    hook: ProgressHook
    with ProgressHook() as hook:
        output: Any = pipeline(audio_dict, hook=hook)
    segments_list: list[dict[str, Any]] = []

    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        segments_list.append(
            {"speaker": str(speaker), "start": float(turn.start), "end": float(turn.end)}
        )

    return group_segments_by_speaker(segments_list)

# print (get_diarization('./output_audio/Рафаэль_Рише_покинул_＂Трактор＂_День_с_Алексеем_Шевченко_Bytpl2vSpNE.wav'))

#
# [{'speaker': 'SPEAKER_02', 'start': 0.03096875, 'end': 3.42284375},
#  {'speaker': 'SPEAKER_00', 'start': 4.199093749999999, 'end': 8.586593750000002},
#  {'speaker': 'SPEAKER_00', 'start': 8.89034375, 'end': 13.193468750000001},
#  {'speaker': 'SPEAKER_02', 'start': 18.96471875, 'end': 19.977218750000002},
#  {'speaker': 'SPEAKER_03', 'start': 23.38596875, 'end': 23.402843750000002},
#  {'speaker': 'SPEAKER_02', 'start': 23.402843750000002, 'end': 23.85846875},
#  {'speaker': 'SPEAKER_01', 'start': 32.751593750000005, 'end': 42.16784375},
#  {'speaker': 'SPEAKER_01', 'start': 45.25596875, 'end': 57.861593750000004},
#  {'speaker': 'SPEAKER_01', 'start': 58.485968750000005, 'end': 163.66784375},
#  {'speaker': 'SPEAKER_01', 'start': 164.12346875, 'end': 175.56471875},
#  {'speaker': 'SPEAKER_01', 'start': 178.85534375, 'end': 311.89784375},
#  {'speaker': 'SPEAKER_01', 'start': 315.12096875000003, 'end': 328.67159375},
#  {'speaker': 'SPEAKER_03', 'start': 337.12596875, 'end': 349.09034375000005},
#  {'speaker': 'SPEAKER_03', 'start': 349.47846875000005, 'end': 352.33034375},
#  {'speaker': 'SPEAKER_02', 'start': 351.87471875, 'end': 352.49909375000004},
#  {'speaker': 'SPEAKER_03', 'start': 352.49909375000004, 'end': 352.65096875},
#  {'speaker': 'SPEAKER_02', 'start': 352.65096875, 'end': 352.73534375},
#  {'speaker': 'SPEAKER_02', 'start': 353.27534375000005, 'end': 361.91534375000003},
#  {'speaker': 'SPEAKER_03', 'start': 358.67534375, 'end': 358.87784375},
#  {'speaker': 'SPEAKER_03', 'start': 358.92846875000004, 'end': 358.96221875000003},
#  {'speaker': 'SPEAKER_03', 'start': 359.06346875, 'end': 359.78909375},
#  {'speaker': 'SPEAKER_02', 'start': 362.59034375000005, 'end': 365.13846875},
#  {'speaker': 'SPEAKER_03', 'start': 365.42534375, 'end': 370.96034375000005}]
