from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks


Segment = Tuple[float, float]


@dataclass(frozen=True)
class VoiceMetrics:
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

    def to_dict(self) -> Dict[str, Any]:
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
    audio_path: str,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    rolloff: float = 0.85,
) -> Dict[str, Dict[str, Any]]:
    """
    Оценивает голосовые характеристики по каждому speaker из data['diarization'].

    Входные данные должны соответствовать формату из main.py:52-55:
      data = {'transcription': str, 'diarization': {speaker: {'segments': [(start,end), ...]}}}

    Возвращает:
      {speaker: {числовые метрики...}}
    """
    diar = data.get("diarization")
    if not isinstance(diar, Mapping):
        raise ValueError("data['diarization'] must be a mapping: {speaker: {'segments': [...]}}")

    wav, sr = _load_mono_audio(audio_path)
    results: Dict[str, Dict[str, Any]] = {}
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


def _load_mono_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    return wav, int(sr)


def _extract_segments(payload: Any) -> List[Segment]:
    """
    Accepts:
      - {'segments': [(start, end), ...]} from diarization.get_diarization
      - {'segments': [[start, end], ...]} after JSON roundtrip
    """
    if not isinstance(payload, Mapping):
        raise ValueError("speaker diarization payload must be a mapping like {'segments': [...]} ")
    segs = payload.get("segments", [])
    if not isinstance(segs, Sequence):
        raise ValueError("payload['segments'] must be a sequence")

    out: List[Segment] = []
    for item in segs:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ):
            start = float(item[0])
            end = float(item[1])
            if end > start:
                out.append((start, end))
        else:
            raise ValueError(f"Invalid segment entry: {item!r}. Expected (start,end).")
    return out


def _compute_metrics_for_speaker(
    *,
    wav: np.ndarray,
    sr: int,
    segments: Sequence[Segment],
    frame_ms: float,
    hop_ms: float,
    rolloff: float,
) -> VoiceMetrics:
    eps = 1e-12
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
    total_dur = 0.0
    pitch_all: List[float] = []
    loud_frames_db: List[float] = []
    centroid_vals: List[float] = []
    rolloff_vals: List[float] = []
    flatness_vals: List[float] = []
    syllable_peaks = 0

    for start_s, end_s in segments:
        seg = _slice(wav, sr, start_s, end_s)
        if seg.size < int(0.05 * sr):  # too short
            continue
        dur = seg.size / sr
        total_dur += dur

        p = _estimate_pitch_hz(seg, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        pitch_all.extend(p)

        db_frames = _frame_loudness_dbfs(seg, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        loud_frames_db.extend(db_frames)

        c_mean, r_mean, f_mean = _timbre_stats(seg, sr, rolloff=rolloff)
        if c_mean is not None:
            centroid_vals.append(c_mean)
        if r_mean is not None:
            rolloff_vals.append(r_mean)
        if f_mean is not None:
            flatness_vals.append(f_mean)

        syllable_peaks += _estimate_syllable_peaks(seg, sr)

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

    pitch_hz_median, pitch_hz_p10, pitch_hz_p90, int_rng = _pitch_summaries(pitch_all)

    loud_mean = float(np.mean(loud_frames_db)) if loud_frames_db else None
    loud_p90 = float(np.percentile(loud_frames_db, 90)) if loud_frames_db else None

    centroid_mean = float(np.average(centroid_vals)) if centroid_vals else None
    rolloff_mean = float(np.average(rolloff_vals)) if rolloff_vals else None
    flatness_mean = float(np.average(flatness_vals)) if flatness_vals else None

    syll_per_sec = float(syllable_peaks / total_dur) if total_dur > eps else None
    # Для русской речи усреднённо ~2.7 слога на слово (очень грубо, но даёт масштаб).
    wpm_est = float(syll_per_sec * 60.0 / 2.7) if syll_per_sec is not None else None

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
    n = wav.size
    a = max(0, min(n, int(round(start_s * sr))))
    b = max(0, min(n, int(round(end_s * sr))))
    if b <= a:
        return np.zeros((0,), dtype=np.float32)
    return wav[a:b]


def _estimate_pitch_hz(seg: np.ndarray, sr: int, *, frame_ms: float, hop_ms: float) -> List[float]:
    """
    Pitch estimation using autocorrelation (ACF-based), pure numpy.
    Returns voiced pitch values in Hz.
    """
    frame = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    if frame <= 0 or hop <= 0 or seg.size < frame:
        return []

    # Speech pitch range (Hz)
    f_min = 50.0
    f_max = 500.0
    lag_min = int(sr / f_max)
    lag_max = int(sr / f_min)
    if lag_max <= lag_min or lag_max <= 1:
        return []

    # Window for frames
    win = np.hanning(frame).astype(np.float32)

    # Energy threshold to skip silence
    rms_all = np.sqrt(np.mean(seg * seg) + 1e-12)
    if not np.isfinite(rms_all) or rms_all <= 1e-6:
        return []
    energy_thr = float(rms_all * 0.2)

    pitches: List[float] = []
    for i in range(0, seg.size - frame + 1, hop):
        x = (seg[i : i + frame] * win).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < energy_thr:
            continue

        x = x - float(np.mean(x))
        # Autocorrelation via FFT (fast)
        n = int(2 ** math.ceil(math.log2(x.size * 2)))
        X = np.fft.rfft(x, n=n)
        ac = np.fft.irfft(X * np.conj(X), n=n).astype(np.float32)
        ac = ac[: x.size]  # non-negative lags
        if ac[0] <= 1e-8:
            continue

        ac /= float(ac[0])
        search = ac[lag_min: min(lag_max, ac.size)]
        if search.size == 0:
            continue
        k = int(np.argmax(search)) + lag_min
        if k <= 0:
            continue
        f0 = float(sr / k)
        if f_min <= f0 <= f_max:
            pitches.append(f0)
    return pitches


def _pitch_summaries(pitch_values_hz: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not pitch_values_hz:
        return None, None, None, None
    p = np.asarray(pitch_values_hz, dtype=np.float32)
    p10 = float(np.percentile(p, 10))
    p50 = float(np.percentile(p, 50))
    p90 = float(np.percentile(p, 90))
    if p10 <= 0 or p90 <= 0:
        return p50, p10, p90, None
    # semitone range between p10 and p90
    rng_st = float(12.0 * math.log2(p90 / p10))
    return p50, p10, p90, rng_st


def _frame_loudness_dbfs(seg: np.ndarray, sr: int, *, frame_ms: float, hop_ms: float) -> List[float]:
    frame = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    if frame <= 0 or hop <= 0 or seg.size < frame:
        return []
    eps = 1e-12
    out: List[float] = []
    for i in range(0, seg.size - frame + 1, hop):
        chunk = seg[i : i + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk) + eps))
        db = 20.0 * math.log10(rms + eps)  # dBFS relative to 1.0 full-scale float
        out.append(db)
    # drop extreme silence frames (helps diarization boundaries)
    if out:
        thr = np.percentile(out, 10) - 10.0
        out = [v for v in out if v >= thr]
    return out


def _timbre_stats(seg: np.ndarray, sr: int, *, rolloff: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (spectral_centroid_mean_hz, spectral_rolloff_mean_hz, spectral_flatness_mean).
    """
    if seg.size < int(0.1 * sr):
        return None, None, None
    n_fft = 1024 if sr >= 16000 else 512
    hop = max(1, n_fft // 4)
    win = np.hanning(n_fft).astype(np.float32)
    if seg.size < n_fft:
        return None, None, None

    frames = []
    for i in range(0, seg.size - n_fft + 1, hop):
        frames.append(seg[i : i + n_fft] * win)
    if not frames:
        return None, None, None
    F = np.fft.rfft(np.stack(frames, axis=0), n=n_fft, axis=1)  # (frames, freq_bins)
    mag = np.abs(F).astype(np.float32) + 1e-10
    power = mag * mag

    freqs = np.linspace(0.0, sr / 2.0, mag.shape[1], dtype=np.float32)  # (freq_bins,)
    # centroid per frame
    centroid = (mag * freqs[None, :]).sum(axis=1) / mag.sum(axis=1)
    centroid_mean = float(np.mean(centroid))

    # rolloff per frame
    cumsum = np.cumsum(power, axis=1)
    total = cumsum[:, -1] + 1e-10
    target = total * float(rolloff)
    idx = (cumsum >= target[:, None]).argmax(axis=1)
    rolloff_hz = freqs[idx]
    rolloff_mean = float(np.mean(rolloff_hz))

    # flatness per frame: geometric / arithmetic
    flatness = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    flatness_mean = float(np.mean(flatness))
    return centroid_mean, rolloff_mean, flatness_mean


def _estimate_syllable_peaks(seg: np.ndarray, sr: int) -> int:
    """
    Very rough syllable nuclei estimate:
    - bandpass 300-3400Hz
    - RMS envelope (25ms / 10ms)
    - peak count with basic thresholds
    """
    if seg.size < int(0.2 * sr):
        return 0
    try:
        b, a = butter(N=2, Wn=(300 / (sr / 2), 3400 / (sr / 2)), btype="bandpass")
        y = filtfilt(b, a, seg).astype(np.float32)
    except Exception:
        y = seg

    frame = int(round(sr * 0.025))
    hop = int(round(sr * 0.010))
    if y.size < frame:
        return 0

    env = []
    for i in range(0, y.size - frame + 1, hop):
        chunk = y[i : i + frame]
        env.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    env = np.asarray(env, dtype=np.float32)
    if env.size < 5:
        return 0

    thr = float(np.median(env) + 0.6 * np.std(env))
    if thr <= 0:
        return 0

    # Min distance between syllable peaks ~80ms (8 frames at 10ms hop)
    distance = max(1, int(round(0.08 / 0.010)))
    peaks, _ = find_peaks(env, height=thr, distance=distance, prominence=(0.1 * thr))
    return int(peaks.size)