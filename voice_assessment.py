from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks


Segment = tuple[float, float]


@dataclass(frozen=True)
class VoiceMetrics:
    """Набор акустических метрик голоса по сегментам спикера.

    Attributes:
        duration_sec: Суммарная длительность сегментов, с.
        num_segments: Количество сегментов.
        pitch_hz_median: Медиана F0, Гц.
        pitch_hz_p10: 10-й перцентиль F0, Гц.
        pitch_hz_p90: 90-й перцентиль F0, Гц.
        intonation_range_st_p10_p90: Диапазон интонации в полутонах (p10–p90).
        speech_rate_syllables_per_sec: Слогов в секунду.
        speech_rate_wpm_est: Грубая оценка слов в минуту.
        loudness_dbfs_mean: Средняя громкость, dBFS.
        loudness_dbfs_p90: 90-й перцентиль громкости, dBFS.
        timbre_centroid_hz_mean: Средний спектральный центроид, Гц.
        timbre_rolloff_hz_mean: Средняя частота спектрального rolloff, Гц.
        timbre_flatness_mean: Средняя спектральная плоскостность.
    """
    duration_sec: float
    num_segments: int

    # Pitch / intonation
    pitch_hz_median: Optional[float]
    pitch_hz_p10: Optional[float]
    pitch_hz_p90: Optional[float]
    intonation_range_st_p10_p90: Optional[float]

    # Speech rate (acoustic estimate)
    speech_rate_syllables_per_sec: Optional[float]
    speech_rate_wpm_est: Optional[float]

    # Loudness
    loudness_dbfs_mean: Optional[float]
    loudness_dbfs_p90: Optional[float]

    # Timbre
    timbre_centroid_hz_mean: Optional[float]
    timbre_rolloff_hz_mean: Optional[float]
    timbre_flatness_mean: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        """Преобразует метрики в словарь для сериализации (JSON, логирование).

        Returns:
            Словарь с теми же полями, что и атрибуты датакласса (числа или None).
        """
        return {
            "duration_sec": float(self.duration_sec),
            "num_segments": int(self.num_segments),
            "pitch_hz_median": self.pitch_hz_median,
            "pitch_hz_p10": self.pitch_hz_p10,
            "pitch_hz_p90": self.pitch_hz_p90,
            "intonation_range_st_p10_p90": self.intonation_range_st_p10_p90,
            "speech_rate_syllables_per_sec": self.speech_rate_syllables_per_sec,
            "speech_rate_wpm_est": self.speech_rate_wpm_est,
            "loudness_dbfs_mean": self.loudness_dbfs_mean,
            "loudness_dbfs_p90": self.loudness_dbfs_p90,
            "timbre_centroid_hz_mean": self.timbre_centroid_hz_mean,
            "timbre_rolloff_hz_mean": self.timbre_rolloff_hz_mean,
            "timbre_flatness_mean": self.timbre_flatness_mean,
        }


def assess_voices(
    data: Mapping[str, Any],
    audio_path: Union[str, io.IOBase],
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    rolloff: float = 0.85,
) -> dict[str, dict[str, Any]]:
    """Оценивает голосовые характеристики по каждому speaker из data['diarization'].

    Входные данные: data с ключом 'diarization' — словарь {speaker: {'segments': [(start, end), ...]}}.
    Для каждого спикера загружает аудио по audio_path, извлекает сегменты, считает метрики
    (питч, громкость, тембр, темп/слоги) и возвращает их в виде словаря to_dict().

    Args:
        data: Словарь с ключом 'diarization' (формат process/process_video_file).
        audio_path: Путь к моно WAV-файлу или file-like объект (BytesIO).
        frame_ms: Длина кадра в мс для питча/громкости.
        hop_ms: Шаг кадра в мс.
        rolloff: Доля энергии для спектрального rolloff (0..1).

    Returns:
        Словарь {speaker: {метрики в виде dict}}.

    Raises:
        ValueError: Если data['diarization'] не является mapping.
    """
    diar = data.get("diarization")
    if not isinstance(diar, Mapping):
        raise ValueError("data['diarization'] must be a mapping: {speaker: {'segments': [...]}}")

    wav, sr = _load_mono_audio(audio_path)
    results: dict[str, dict[str, Any]] = {}
    for speaker, payload in diar.items():
        segments = _extract_segments(payload)
        metrics = _compute_metrics_for_speaker(
            wav=wav,
            sr=sr,
            segments=segments,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            rolloff=rolloff,
        )
        results[str(speaker)] = metrics.to_dict()
    return results


def _load_mono_audio(path: Union[str, io.IOBase]) -> tuple[np.ndarray, int]:
    """Загружает аудиофайл как моно float32 и частоту дискретизации.

    Args:
        path: Путь к WAV/файлу или file-like объект (BytesIO), поддерживаемый soundfile.

    Returns:
        Кортеж (waveform shape (n_samples,), sample_rate).
    """
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    return wav, int(sr)


def _extract_segments(payload: Any) -> list[Segment]:
    """Извлекает список сегментов (start, end) из payload диаризации спикера.

    Поддерживает кортежи и списки пар чисел; сегменты с end <= start отбрасываются.

    Args:
        payload: Словарь с ключом 'segments' — последовательность пар (start, end)
            или [start, end] (после JSON).

    Returns:
        Список кортежей (float, float) с end > start.

    Raises:
        ValueError: Если payload не mapping, segments не sequence или элемент сегмента неверный.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("speaker diarization payload must be a mapping like {'segments': [...]} ")
    segs: Any = payload.get("segments", [])
    if not isinstance(segs, Sequence):
        raise ValueError("payload['segments'] must be a sequence")

    out: list[Segment] = []
    item: Any
    for item in segs:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ):
            start: float = float(item[0])
            end: float = float(item[1])
            if end > start:
                out.append((start, end))
        else:
            raise ValueError(f"Invalid segment entry: {item!r}. Expected (start,end).")
    return out


def _compute_segment_metrics(
    wav: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    *,
    frame_ms: float,
    hop_ms: float,
    rolloff: float,
) -> tuple[
    float,
    list[float],
    list[float],
    Optional[float],
    Optional[float],
    Optional[float],
    int,
]:
    """Вычисляет метрики для одного сегмента: длительность, питч, громкость, тембр, слоги.

    Сегменты короче 50 мс возвращают нулевые значения и пустые списки.

    Args:
        wav: Моно waveform (float32).
        sr: Частота дискретизации.
        start_s: Начало сегмента в секундах.
        end_s: Конец сегмента в секундах.
        frame_ms: Длина кадра в мс.
        hop_ms: Шаг кадра в мс.
        rolloff: Параметр спектрального rolloff.

    Returns:
        Кортеж (dur, pitch_list, loudness_frames_list, centroid_mean, rolloff_mean,
        flatness_mean, syllable_peaks). Для короткого сегмента — (0, [], [], None, None, None, 0).
    """
    seg: np.ndarray = _slice(wav, sr, start_s, end_s)
    if seg.size < int(0.05 * sr):  # too short
        return (0.0, [], [], None, None, None, 0)
    dur: float = seg.size / sr
    p: list[float] = _estimate_pitch_hz(seg, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    db_frames: list[float] = _frame_loudness_dbfs(
        seg, sr, frame_ms=frame_ms, hop_ms=hop_ms
    )
    c_mean: Optional[float]
    r_mean: Optional[float]
    f_mean: Optional[float]
    c_mean, r_mean, f_mean = _timbre_stats(seg, sr, rolloff=rolloff)
    syllable_peaks: int = _estimate_syllable_peaks(seg, sr)
    return (dur, p, db_frames, c_mean, r_mean, f_mean, syllable_peaks)


def _compute_metrics_for_speaker(
    *,
    wav: np.ndarray,
    sr: int,
    segments: Sequence[Segment],
    frame_ms: float,
    hop_ms: float,
    rolloff: float,
) -> VoiceMetrics:
    """Агрегирует метрики по всем сегментам спикера в один объект VoiceMetrics.

    Суммирует длительность, объединяет питч/громкость по кадрам, усредняет тембровые
    метрики, считает слоги в секунду и грубую оценку WPM. При пустых сегментах или
    нулевой суммарной длительности возвращает VoiceMetrics с нулями и None.

    Args:
        wav: Моно waveform.
        sr: Частота дискретизации.
        segments: Список пар (start_s, end_s).
        frame_ms: Длина кадра в мс.
        hop_ms: Шаг кадра в мс.
        rolloff: Параметр rolloff для тембра.

    Returns:
        Экземпляр VoiceMetrics.
    """
    eps: float = 1e-12
    if not segments:
        return VoiceMetrics(
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

    # Aggregate over segments (weighted by segment duration)
    total_dur: float = 0.0
    pitch_all: list[float] = []
    loud_frames_db: list[float] = []
    centroid_vals: list[float] = []
    rolloff_vals: list[float] = []
    flatness_vals: list[float] = []
    syllable_peaks: int = 0

    start_s: float
    end_s: float
    for start_s, end_s in segments:
        dur: float
        p: list[float]
        db_frames: list[float]
        c_mean: Optional[float]
        r_mean: Optional[float]
        f_mean: Optional[float]
        syll: int
        dur, p, db_frames, c_mean, r_mean, f_mean, syll = _compute_segment_metrics(
            wav, sr, start_s, end_s,
            frame_ms=frame_ms, hop_ms=hop_ms, rolloff=rolloff,
        )
        total_dur += dur
        pitch_all.extend(p)
        loud_frames_db.extend(db_frames)
        if c_mean is not None:
            centroid_vals.append(c_mean)
        if r_mean is not None:
            rolloff_vals.append(r_mean)
        if f_mean is not None:
            flatness_vals.append(f_mean)
        syllable_peaks += syll

    if total_dur <= 0:
        return VoiceMetrics(
            duration_sec=0.0,
            num_segments=len(segments),
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

    pitch_hz_median: Optional[float]
    pitch_hz_p10: Optional[float]
    pitch_hz_p90: Optional[float]
    int_rng: Optional[float]
    pitch_hz_median, pitch_hz_p10, pitch_hz_p90, int_rng = _pitch_summaries(pitch_all)

    loud_mean: Optional[float] = (
        float(np.mean(loud_frames_db)) if loud_frames_db else None
    )
    loud_p90: Optional[float] = (
        float(np.percentile(loud_frames_db, 90)) if loud_frames_db else None
    )

    centroid_mean: Optional[float] = (
        float(np.average(centroid_vals)) if centroid_vals else None
    )
    rolloff_mean: Optional[float] = (
        float(np.average(rolloff_vals)) if rolloff_vals else None
    )
    flatness_mean: Optional[float] = (
        float(np.average(flatness_vals)) if flatness_vals else None
    )

    syll_per_sec: Optional[float] = (
        float(syllable_peaks / total_dur) if total_dur > eps else None
    )
    # Для русской речи усреднённо ~2.7 слога на слово (очень грубо, но даёт масштаб).
    wpm_est: Optional[float] = (
        float(syll_per_sec * 60.0 / 2.7) if syll_per_sec is not None else None
    )

    return VoiceMetrics(
        duration_sec=float(total_dur),
        num_segments=len(segments),
        pitch_hz_median=pitch_hz_median,
        pitch_hz_p10=pitch_hz_p10,
        pitch_hz_p90=pitch_hz_p90,
        intonation_range_st_p10_p90=int_rng,
        speech_rate_syllables_per_sec=syll_per_sec,
        speech_rate_wpm_est=wpm_est,
        loudness_dbfs_mean=loud_mean,
        loudness_dbfs_p90=loud_p90,
        timbre_centroid_hz_mean=centroid_mean,
        timbre_rolloff_hz_mean=rolloff_mean,
        timbre_flatness_mean=flatness_mean,
    )


def _slice(wav: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Вырезает фрагмент аудио по временному интервалу в секундах.

    Границы приводятся к сэмплам и ограничиваются [0, n]; при b <= a возвращается пустой массив.

    Args:
        wav: Одномерный массив сэмплов.
        sr: Частота дискретизации.
        start_s: Начало интервала, с.
        end_s: Конец интервала, с.

    Returns:
        Массив float32 сэмплов или пустой массив.
    """
    n: int = wav.size
    a: int = max(0, min(n, int(round(start_s * sr))))
    b: int = max(0, min(n, int(round(end_s * sr))))
    if b <= a:
        return np.zeros((0,), dtype=np.float32)
    return wav[a:b]


def _estimate_pitch_for_frame(
    x: np.ndarray,
    sr: int,
    *,
    energy_thr: float,
    lag_min: int,
    lag_max: int,
    f_min: float,
    f_max: float,
) -> Optional[float]:
    """Оценивает частоту основного тона (F0) для одного оконного кадра методом автокорреляции.

    Порог по энергии отсекает тишину; автокорреляция через FFT; поиск лага в [lag_min, lag_max];
    F0 = sr / lag; проверка попадания в [f_min, f_max].

    Args:
        x: Один кадр (окно уже применено).
        sr: Частота дискретизации.
        energy_thr: Минимальная RMS для учёта кадра.
        lag_min: Минимальный лаг (соответствует f_max).
        lag_max: Максимальный лаг (соответствует f_min).
        f_min: Минимальная допустимая F0, Гц.
        f_max: Максимальная допустимая F0, Гц.

    Returns:
        F0 в Гц или None при тишине/ошибке/вне диапазона.
    """
    rms: float = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < energy_thr:
        return None

    x = x - float(np.mean(x))
    n: int = int(2 ** math.ceil(math.log2(x.size * 2)))
    X: np.ndarray = np.fft.rfft(x, n=n)
    ac: np.ndarray = np.fft.irfft(X * np.conj(X), n=n).astype(np.float32)
    ac = ac[: x.size]
    if ac[0] <= 1e-8:
        return None

    ac /= float(ac[0])
    search: np.ndarray = ac[lag_min : min(lag_max, ac.size)]
    if search.size == 0:
        return None
    k: int = int(np.argmax(search)) + lag_min
    if k <= 0:
        return None
    f0: float = float(sr / k)
    if f_min <= f0 <= f_max:
        return f0
    return None


def _estimate_pitch_hz(seg: np.ndarray, sr: int, *, frame_ms: float, hop_ms: float) -> list[float]:
    """Оценка основного тона (F0) по сегменту методом автокорреляции по кадрам.

    Разбивает сегмент на перекрывающиеся кадры (Hanning), для каждого кадра вызывает
    _estimate_pitch_for_frame в диапазоне 50–500 Гц; тихие кадры и сегменты с низкой
    общей энергией пропускаются.

    Args:
        seg: Сегмент аудио (float32).
        sr: Частота дискретизации.
        frame_ms: Длина кадра, мс.
        hop_ms: Шаг кадра, мс.

    Returns:
        Список значений F0 в Гц по озвученным кадрам (может быть пустым).
    """
    frame: int = int(round(sr * frame_ms / 1000.0))
    hop: int = int(round(sr * hop_ms / 1000.0))
    if frame <= 0 or hop <= 0 or seg.size < frame:
        return []

    # Speech pitch range (Hz)
    f_min: float = 50.0
    f_max: float = 500.0
    lag_min: int = int(sr / f_max)
    lag_max: int = int(sr / f_min)
    if lag_max <= lag_min or lag_max <= 1:
        return []

    # Window for frames
    win: np.ndarray = np.hanning(frame).astype(np.float32)

    # Energy threshold to skip silence
    rms_all: float = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    if not np.isfinite(rms_all) or rms_all <= 1e-6:
        return []
    energy_thr: float = float(rms_all * 0.2)

    pitches: list[float] = []
    i: int
    for i in range(0, seg.size - frame + 1, hop):
        x: np.ndarray = (seg[i : i + frame] * win).astype(np.float32)
        f0: Optional[float] = _estimate_pitch_for_frame(
            x, sr,
            energy_thr=energy_thr, lag_min=lag_min, lag_max=lag_max,
            f_min=f_min, f_max=f_max,
        )
        if f0 is not None:
            pitches.append(f0)
    return pitches


def _pitch_summaries(
    pitch_values_hz: Sequence[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Считает перцентили питча и диапазон в полутонах между p10 и p90.

    Args:
        pitch_values_hz: Список значений F0, Гц.

    Returns:
        Кортеж (median, p10, p90, range_semitone). При пустом вводе или p10/p90 <= 0
        диапазон None.
    """
    if not pitch_values_hz:
        return None, None, None, None
    p: np.ndarray = np.asarray(pitch_values_hz, dtype=np.float32)
    p10: float = float(np.percentile(p, 10))
    p50: float = float(np.percentile(p, 50))
    p90: float = float(np.percentile(p, 90))
    if p10 <= 0 or p90 <= 0:
        return p50, p10, p90, None
    # semitone range between p10 and p90
    rng_st: float = float(12.0 * math.log2(p90 / p10))
    return p50, p10, p90, rng_st


def _frame_loudness_dbfs(seg: np.ndarray, sr: int, *, frame_ms: float, hop_ms: float) -> list[float]:
    """Вычисляет громкость по кадрам в dBFS (относительно 1.0 full-scale).

    После расчёта отбрасываются кадры с уровнем ниже (p10 - 10 dB) для сглаживания границ.

    Args:
        seg: Сегмент аудио.
        sr: Частота дискретизации.
        frame_ms: Длина кадра, мс.
        hop_ms: Шаг кадра, мс.

    Returns:
        Список значений громкости в dB по кадрам (может быть пустым).
    """
    frame: int = int(round(sr * frame_ms / 1000.0))
    hop: int = int(round(sr * hop_ms / 1000.0))
    if frame <= 0 or hop <= 0 or seg.size < frame:
        return []
    eps: float = 1e-12
    out: list[float] = []
    i: int
    for i in range(0, seg.size - frame + 1, hop):
        chunk: np.ndarray = seg[i : i + frame]
        rms: float = float(np.sqrt(np.mean(chunk * chunk) + eps))
        db: float = 20.0 * math.log10(rms + eps)  # dBFS relative to 1.0 full-scale float
        out.append(db)
    # drop extreme silence frames (helps diarization boundaries)
    if out:
        thr: float = float(np.percentile(out, 10) - 10.0)
        out = [v for v in out if v >= thr]
    return out


def _timbre_stats(
    seg: np.ndarray, sr: int, *, rolloff: float
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Считает спектральные метрики тембра: центроид, rolloff, flatness.

    Короткие сегменты (< 0.1 с или меньше n_fft сэмплов) возвращают (None, None, None).
    Окно Hanning, n_fft 1024 при sr >= 16000 иначе 512.

    Args:
        seg: Сегмент аудио.
        sr: Частота дискретизации.
        rolloff: Доля энергии для расчёта частоты rolloff (0..1).

    Returns:
        Кортеж (spectral_centroid_mean_hz, spectral_rolloff_mean_hz, spectral_flatness_mean).
    """
    if seg.size < int(0.1 * sr):
        return None, None, None
    n_fft: int = 1024 if sr >= 16000 else 512
    hop: int = max(1, n_fft // 4)
    win: np.ndarray = np.hanning(n_fft).astype(np.float32)
    if seg.size < n_fft:
        return None, None, None

    frames: list[np.ndarray] = []
    i: int
    for i in range(0, seg.size - n_fft + 1, hop):
        frames.append(seg[i : i + n_fft] * win)
    if not frames:
        return None, None, None
    F: np.ndarray = np.fft.rfft(np.stack(frames, axis=0), n=n_fft, axis=1)
    mag: np.ndarray = np.abs(F).astype(np.float32) + 1e-10
    power: np.ndarray = mag * mag

    freqs: np.ndarray = np.linspace(
        0.0, sr / 2.0, mag.shape[1], dtype=np.float32
    )
    centroid: np.ndarray = (mag * freqs[None, :]).sum(axis=1) / mag.sum(axis=1)
    centroid_mean: float = float(np.mean(centroid))

    cumsum: np.ndarray = np.cumsum(power, axis=1)
    total: np.ndarray = cumsum[:, -1] + 1e-10
    target: np.ndarray = total * float(rolloff)
    idx: np.ndarray = (cumsum >= target[:, None]).argmax(axis=1)
    rolloff_hz: np.ndarray = freqs[idx]
    rolloff_mean: float = float(np.mean(rolloff_hz))

    flatness: np.ndarray = (
        np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    )
    flatness_mean: float = float(np.mean(flatness))
    return centroid_mean, rolloff_mean, flatness_mean


def _estimate_syllable_peaks(seg: np.ndarray, sr: int) -> int:
    """Грубая оценка числа слоговых ядер по огибающей громкости.

    Полоса 300–3400 Гц, RMS-огибающая по кадрам 25 мс / 10 мс, поиск пиков с порогом
    и минимальным расстоянием ~80 мс между пиками.

    Args:
        seg: Сегмент аудио.
        sr: Частота дискретизации.

    Returns:
        Количество найденных пиков (оценка слогов).
    """
    if seg.size < int(0.2 * sr):
        return 0
    try:
        b: np.ndarray
        a: np.ndarray
        b, a = butter(N=2, Wn=(300 / (sr / 2), 3400 / (sr / 2)), btype="bandpass")
        y: np.ndarray = filtfilt(b, a, seg).astype(np.float32)
    except Exception:
        y = seg

    frame: int = int(round(sr * 0.025))
    hop: int = int(round(sr * 0.010))
    if y.size < frame:
        return 0

    env_list: list[float] = []
    i: int
    for i in range(0, y.size - frame + 1, hop):
        chunk: np.ndarray = y[i : i + frame]
        env_list.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    env: np.ndarray = np.asarray(env_list, dtype=np.float32)
    if env.size < 5:
        return 0

    thr: float = float(np.median(env) + 0.6 * np.std(env))
    if thr <= 0:
        return 0

    # Min distance between syllable peaks ~80ms (8 frames at 10ms hop)
    distance: int = max(1, int(round(0.08 / 0.010)))
    peaks: np.ndarray
    peaks, _ = find_peaks(env, height=thr, distance=distance, prominence=(0.1 * thr))
    return int(peaks.size)