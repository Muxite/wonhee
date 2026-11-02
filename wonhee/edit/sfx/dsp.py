"""Tiny numpy synth kit for sound effects: oscillators, noise, envelopes, FFT/STFT filters, space and grit.

Why it exists: wonhee's sound effects are synthesized in code, not shipped as samples, so there is no licence to
track and every sound is reproducible from a seed. Everything here is mono float64 at SR (the synthesis rate). The
caller resamples to the requested output rate afterwards (see `retime`). Frequencies and cutoffs may be scalars or
per-sample arrays; time-varying filters run block-wise in the STFT domain, so there are no Python sample loops.
"""
import numpy as np

SR = 44100          # synthesis rate; sounds are built here and retimed to the requested rate at the end
TAU = 2 * np.pi


def n_of(sec: float) -> int:
    return max(1, int(round(sec * SR)))


def times(n: int) -> np.ndarray:
    return np.arange(n) / SR


def curve(a, b, n: int, shape: float = 1.0) -> np.ndarray:
    """a -> b over n samples; shape>1 eases in, <1 eases out. Geometric when both ends are > 0."""
    u = np.linspace(0, 1, n) ** shape
    if a > 0 and b > 0:
        return a * (b / a) ** u
    return a + (b - a) * u


def osc(freq, n: int, shape: str = "sine", phase: float = 0.0) -> np.ndarray:
    """Phase-integrated oscillator: the phase is the running sum of freq, so swept freq arrays glide without clicks."""
    f = np.broadcast_to(np.asarray(freq, float), (n,))
    ph = phase + np.cumsum(f) / SR
    if shape == "sine":
        return np.sin(TAU * ph)
    if shape == "saw":
        return 2 * (ph % 1) - 1
    if shape == "square":
        return np.sign(np.sin(TAU * ph))
    if shape == "tri":
        return 2 * np.abs(2 * (ph % 1) - 1) - 1
    raise ValueError(shape)


def noise(n: int, rng: np.random.Generator, color: str = "white") -> np.ndarray:
    # All randomness comes from the caller's Generator, never the global numpy state, so a seed replays exactly.
    w = rng.standard_normal(n)
    if color == "white":
        return w / 3
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1] if n > 1 else 1
    spec /= np.sqrt(f) if color == "pink" else f        # brown: 1/f amplitude
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-12)


def env(n: int, attack: float = 0.002, decay: float = 0.3, hold: float = 0.0) -> np.ndarray:
    """Linear attack, optional hold, then exponential decay with time constant `decay`."""
    t = times(n)
    a = max(attack, 1e-4)
    e = np.where(t < a, t / a, np.exp(-np.maximum(0, t - a - hold) / decay))
    return e


def fades(x: np.ndarray, fin: float = 0.005, fout: float = 0.02) -> np.ndarray:
    """Short linear ramps at both ends so a cut sound never clicks."""
    x = x.copy()
    i, o = min(len(x), n_of(fin)), min(len(x), n_of(fout))
    x[:i] *= np.linspace(0, 1, i)
    x[len(x) - o:] *= np.linspace(1, 0, o)
    return x


def _mask(f: np.ndarray, lo, hi, order: int = 2, res: float = 0.0) -> np.ndarray:
    m = np.ones_like(f)
    if lo is not None:
        m = m / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    if hi is not None:
        m = m / np.sqrt(1 + (f / hi) ** (2 * order))
        if res:
            m = m + res * np.exp(-0.5 * ((f - hi) / (0.08 * hi)) ** 2)
    return m


def filt(x: np.ndarray, lo=None, hi=None, order: int = 2, res: float = 0.0) -> np.ndarray:
    """Band-limit x. Scalars filter in one FFT; arrays (per-sample cutoffs) sweep via STFT."""
    if np.ndim(lo) == 0 and np.ndim(hi) == 0:
        size = 1 << (2 * len(x) - 1).bit_length()   # zero-pad: no circular wrap of the tail
        spec = np.fft.rfft(x, size)
        spec *= _mask(np.fft.rfftfreq(size, 1 / SR), lo, hi, order, res)
        return np.fft.irfft(spec, size)[:len(x)]
    win, hop = 1024, 256
    w = np.hanning(win)
    pad = np.concatenate([np.zeros(win), x, np.zeros(win)])
    starts = np.arange(0, len(pad) - win, hop)
    frames = np.stack([pad[s:s + win] for s in starts]) * w
    centres = np.clip(starts + win // 2 - win, 0, len(x) - 1)

    def at(c):   # per-sample cutoff -> one value per frame, as a column
        return c if c is None or np.ndim(c) == 0 else np.asarray(c, float)[centres][:, None]
    f = np.fft.rfftfreq(win, 1 / SR)[None, :]
    spec = np.fft.rfft(frames, axis=1) * _mask(f, at(lo), at(hi), order, res)
    out, norm = np.zeros(len(pad)), np.zeros(len(pad))
    for s, fr in zip(starts, np.fft.irfft(spec, win, axis=1) * w):
        out[s:s + win] += fr
        norm[s:s + win] += w * w
    return (out / np.maximum(norm, 1e-6))[win:win + len(x)]


def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:n]


def reverb(x: np.ndarray, rng: np.random.Generator, size: float = 1.2, mix: float = 0.25,
           tone: float = 6000) -> np.ndarray:
    """Exponentially decaying filtered-noise impulse response; output grows by the tail length."""
    ir = filt(noise(n_of(size), rng), None, tone) * np.exp(-times(n_of(size)) / (size / 7))   # -60 dB at the end
    wet = convolve(x, ir)
    wet *= (np.abs(x).max() + 1e-12) / (np.abs(wet).max() + 1e-12)
    out = wet * mix
    out[:len(x)] += x * (1 - mix)
    return out


def crush(x: np.ndarray, bits: int = 6, hold: int = 4) -> np.ndarray:
    """Bit-depth + sample-rate reduction (sample and hold)."""
    y = np.repeat(x[::hold], hold)[:len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def drive(x: np.ndarray, amount: float = 2.0) -> np.ndarray:
    return np.tanh(x * amount) / np.tanh(amount)


def resample(x: np.ndarray, rate) -> np.ndarray:
    """Varispeed playback: rate scalar or per-output-sample array (negative = backwards)."""
    if np.ndim(rate) == 0:
        n = max(1, int(len(x) / abs(rate)))
        pos = np.arange(n) * rate
    else:
        pos = np.cumsum(rate)
    return np.interp(np.mod(pos, len(x) - 1), np.arange(len(x)), x)


def retime(x: np.ndarray, sr: int) -> np.ndarray:
    """Convert synthesized audio from SR to `sr`. Downsampling low-passes first so content above the new Nyquist
    does not fold back as aliasing; linear interpolation can never exceed the input peak."""
    if sr == SR:
        return x
    if sr < SR:
        x = filt(x, None, 0.45 * sr)
    m = max(1, int(round(len(x) * sr / SR)))
    return np.interp(np.arange(m) * (SR / sr), np.arange(len(x)), x)


def place(dst: np.ndarray, src: np.ndarray, at: int, gain: float = 1.0) -> None:
    """Add src into dst starting at sample `at`, clipping both ends."""
    if at < 0:
        src, at = src[-at:], 0
    k = min(len(src), len(dst) - at)
    if k > 0:
        dst[at:at + k] += src[:k] * gain


def normalise(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    x = filt(x, 20, None)   # DC block (subtracting the mean would lift silence off zero)
    return x * (peak / (np.abs(x).max() + 1e-12))
