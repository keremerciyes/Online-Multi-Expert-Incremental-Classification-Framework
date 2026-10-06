"""Deployable feature-extractor backbones ported from the Avalanche comparison.

Two backbone families replace the old raw conv encoders:

- ``BundleEncoderBackbone``: a frozen conv autoencoder encoder loaded from a
  ``.pt`` design bundle (``ae_raw_basicmotions``, ``ae_raw_clap``). Emits a
  ``[B, C, T']`` feature map for conv/linear projector heads.
- ``PrecomputedFeatureBackbone``: an identity backbone over feature vectors that
  the datamodule precomputes from a handcrafted / hybrid bundle
  (``handcrafted16``, ``wdcnn_hybrid``). Emits ``[B, D, 1]``.

Handcrafted descriptors are speed-dependent (order-domain), so those bundles are
precomputed in the datamodule where the shaft speed per window is known.
"""

import re

import numpy as np
import torch
import torch.nn as nn

from motor_anomaly_classification.models.base import BaseBackbone


SAMPLING_RATE_HZ = 26667.0
WINDOW_SECONDS = 0.2

HANDCRAFTED22_FEATURE_NAMES = (
    "log_rms", "crest", "kurtosis", "spectral_entropy_hz", "frequency_centroid_hz",
    "xy_correlation", "xz_correlation", "yz_correlation", "log_axis_rms_ratio",
    "log_xz_rms_ratio", "log_yz_rms_ratio", "hf_kurtosis", "order_2_3_4x_total",
    "order_5_18x_total", "order_18p5_21p5_total",
    "order_23p5_24p5_41p5_42p5_over_23p5_42p5", "hz_2000_2250_over_1200_2250",
    "hz_2200_2400_total", "hz_2400_2700_over_2400_3200",
    "hz_3200_3450_over_3200_3750", "hz_400_1000_total", "hz_5000_nyquist_total",
)


# ---------------------------------------------------------------------------
# Handcrafted physical vibration descriptors (16 features, speed-aware).
# ---------------------------------------------------------------------------
def _kurtosis(sig: np.ndarray) -> np.ndarray:
    centered = sig - sig.mean(axis=1, keepdims=True)
    var = (centered ** 2).mean(axis=1)
    return (centered ** 4).mean(axis=1) / (var ** 2 + 1e-12)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = (a0 * b0).mean(axis=1)
    den = np.sqrt((a0 ** 2).mean(axis=1) * (b0 ** 2).mean(axis=1)) + 1e-12
    return num / den


def _sum_band(ac: np.ndarray, ac_freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    mask = (ac_freqs >= f_lo) & (ac_freqs < f_hi)
    if not mask.any():
        idx = int(np.argmin(np.abs(ac_freqs - 0.5 * (f_lo + f_hi))))
        return ac[:, idx]
    return ac[:, mask].sum(axis=1)


def _abs_ratio(ac: np.ndarray, ac_freqs: np.ndarray, f_lo: float, f_hi: float, total: np.ndarray) -> np.ndarray:
    return _sum_band(ac, ac_freqs, f_lo, f_hi) / total


def _order_ratio(ac: np.ndarray, ac_freqs: np.ndarray, o_lo: float, o_hi: float, f_r: float, df: float, total: np.ndarray) -> np.ndarray:
    f_lo, f_hi = o_lo * f_r, o_hi * f_r
    if f_hi - f_lo < df:
        center = 0.5 * (f_lo + f_hi)
        f_lo, f_hi = center - 0.5 * df, center + 0.5 * df
    return _sum_band(ac, ac_freqs, f_lo, f_hi) / total


def _hf_kurtosis(sig: np.ndarray, freqs: np.ndarray, f_cut: float, t: int) -> np.ndarray:
    spec = np.fft.rfft(sig, axis=1)
    spec[:, freqs < f_cut] = 0.0
    return _kurtosis(np.fft.irfft(spec, n=t, axis=1))


def handcrafted_features(x: np.ndarray, rpm: np.ndarray, fs: float = SAMPLING_RATE_HZ) -> np.ndarray:
    """Compact, physically-interpretable vibration descriptors (16 features).

    x:   (N, T, C) windows (C >= 2, channel 0 = X, channel 1 = Y accelerometer)
    rpm: (N,) shaft speed in rpm for each window
    """
    x = np.asarray(x, dtype=np.float32)
    n, t, c = x.shape
    eps = 1e-8
    ax_raw = x[:, :, 0]
    ay_raw = x[:, :, 1] if c > 1 else x[:, :, 0]
    ax = ax_raw - ax_raw.mean(axis=1, keepdims=True)
    ay = ay_raw - ay_raw.mean(axis=1, keepdims=True)
    magnitude = np.sqrt(ax ** 2 + ay ** 2)

    power = (np.abs(np.fft.rfft(ax, axis=1)) ** 2 + np.abs(np.fft.rfft(ay, axis=1)) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    nyquist = fs / 2.0
    df = fs / t
    ac = power[:, 1:]
    ac_freqs = freqs[1:]
    total = ac.sum(axis=1) + eps

    rms = np.sqrt((magnitude ** 2).mean(axis=1))
    log_rms = np.log10(rms + eps)
    crest = magnitude.max(axis=1) / (rms + eps)
    kurt = 0.5 * (_kurtosis(ax) + _kurtosis(ay))

    e_1x = np.zeros(n); e_2x = np.zeros(n); e_harm = np.zeros(n); e_sub = np.zeros(n)
    for rpm_val in np.unique(rpm):
        sel = rpm == rpm_val
        f_r = max(float(rpm_val), 1.0) / 60.0
        e_1x[sel] = _order_ratio(ac[sel], ac_freqs, 0.8, 1.2, f_r, df, total[sel])
        e_2x[sel] = _order_ratio(ac[sel], ac_freqs, 1.8, 2.2, f_r, df, total[sel])
        e_harm[sel] = _order_ratio(ac[sel], ac_freqs, 2.5, 5.5, f_r, df, total[sel])
        e_sub[sel] = _order_ratio(ac[sel], ac_freqs, 0.1, 0.8, f_r, df, total[sel])

    e_line = _abs_ratio(ac, ac_freqs, 90.0, 130.0, total)
    e_mid = _abs_ratio(ac, ac_freqs, 200.0, 1000.0, total)
    e_high = _abs_ratio(ac, ac_freqs, 1000.0, 5000.0, total)
    e_vhf = _abs_ratio(ac, ac_freqs, 5000.0, nyquist, total)

    prob = ac / total[:, None]
    centroid = (ac_freqs[None, :] * prob).sum(axis=1)
    entropy = -(prob * np.log(prob + eps)).sum(axis=1) / np.log(ac.shape[1])

    xy_corr = _row_corr(ax, ay)
    rms_x = np.sqrt((ax ** 2).mean(axis=1))
    rms_y = np.sqrt((ay ** 2).mean(axis=1))
    axis_ratio = np.log((rms_x + eps) / (rms_y + eps))

    hf_kurt = 0.5 * (_hf_kurtosis(ax, freqs, 1000.0, t) + _hf_kurtosis(ay, freqs, 1000.0, t))

    features = np.stack(
        [
            log_rms, crest, kurt,
            e_1x, e_2x, e_harm, e_sub,
            e_line, e_mid, e_high, e_vhf,
            centroid, entropy,
            xy_corr, axis_ratio,
            hf_kurt,
        ],
        axis=1,
    ).astype(np.float32)
    return np.nan_to_num(features)


# ---------------------------------------------------------------------------
# Robust per-block standardizer (median / IQR + optional per-block L2).
# ---------------------------------------------------------------------------
def handcrafted22_features(windows: np.ndarray, rpm: np.ndarray, fs: float = SAMPLING_RATE_HZ) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[2] < 3:
        raise ValueError("handcrafted22 requires windows with X, Y, Z channels.")
    sample_count, window_length, _ = windows.shape
    rpm = np.asarray(rpm, dtype=np.float32)
    if rpm.shape != (sample_count,) or not np.all(np.isfinite(rpm)) or np.any(rpm <= 0):
        raise ValueError("handcrafted22 requires one finite positive RPM per window.")
    epsilon = 1e-8
    axes = [windows[:, :, index] - windows[:, :, index].mean(axis=1, keepdims=True) for index in range(3)]
    axis_x, axis_y, axis_z = axes
    magnitude = np.sqrt(axis_x ** 2 + axis_y ** 2 + axis_z ** 2)
    taper = np.hanning(window_length).astype(np.float32)[None, :]
    power = sum(np.abs(np.fft.rfft(axis * taper, axis=1)) ** 2 for axis in axes).astype(np.float64)
    frequencies = np.fft.rfftfreq(window_length, d=1.0 / fs)
    ac_power = power[:, 1:]
    ac_frequencies = frequencies[1:]
    total_power = ac_power.sum(axis=1) + epsilon
    rms = np.sqrt((magnitude ** 2).mean(axis=1))
    log_rms = np.log10(rms + epsilon)
    crest = magnitude.max(axis=1) / (rms + epsilon)
    kurtosis = sum(_kurtosis(axis) for axis in axes) / 3.0
    order_2_3_4 = np.zeros(sample_count)
    order_5_18 = np.zeros(sample_count)
    order_18p5_21p5 = np.zeros(sample_count)
    high_order_ratio = np.zeros(sample_count)
    for speed in np.unique(rpm):
        selected = rpm == speed
        shaft_hz = max(float(speed), 1.0) / 60.0
        selected_power = ac_power[selected]
        order_2_3_4[selected] = (
            _sum_band(selected_power, ac_frequencies, 1.75 * shaft_hz, 2.25 * shaft_hz)
            + _sum_band(selected_power, ac_frequencies, 2.75 * shaft_hz, 3.25 * shaft_hz)
            + _sum_band(selected_power, ac_frequencies, 3.75 * shaft_hz, 4.25 * shaft_hz)
        ) / total_power[selected]
        order_5_18[selected] = _sum_band(selected_power, ac_frequencies, 5.0 * shaft_hz, 18.0 * shaft_hz) / total_power[selected]
        order_18p5_21p5[selected] = _sum_band(selected_power, ac_frequencies, 18.5 * shaft_hz, 21.5 * shaft_hz) / total_power[selected]
        context = _sum_band(selected_power, ac_frequencies, 23.5 * shaft_hz, 42.5 * shaft_hz) + epsilon
        high_order_ratio[selected] = (
            _sum_band(selected_power, ac_frequencies, 23.5 * shaft_hz, 24.5 * shaft_hz)
            + _sum_band(selected_power, ac_frequencies, 41.5 * shaft_hz, 42.5 * shaft_hz)
        ) / context
    band_2000 = _sum_band(ac_power, ac_frequencies, 2000.0, 2250.0) / (_sum_band(ac_power, ac_frequencies, 1200.0, 2250.0) + epsilon)
    band_2200 = _abs_ratio(ac_power, ac_frequencies, 2200.0, 2400.0, total_power)
    band_2400 = _sum_band(ac_power, ac_frequencies, 2400.0, 2700.0) / (_sum_band(ac_power, ac_frequencies, 2400.0, 3200.0) + epsilon)
    band_3200 = _sum_band(ac_power, ac_frequencies, 3200.0, 3450.0) / (_sum_band(ac_power, ac_frequencies, 3200.0, 3750.0) + epsilon)
    band_400 = _abs_ratio(ac_power, ac_frequencies, 400.0, 1000.0, total_power)
    band_5000 = _abs_ratio(ac_power, ac_frequencies, 5000.0, fs / 2.0, total_power)
    probability = ac_power / total_power[:, None]
    centroid = (ac_frequencies[None, :] * probability).sum(axis=1)
    entropy = -(probability * np.log(probability + epsilon)).sum(axis=1) / np.log(ac_power.shape[1])
    rms_x, rms_y, rms_z = [np.sqrt((axis ** 2).mean(axis=1)) for axis in axes]
    high_frequency_kurtosis = sum(_hf_kurtosis(axis, frequencies, 1000.0, window_length) for axis in axes) / 3.0
    features = np.stack([
        log_rms, crest, kurtosis, entropy, centroid,
        _row_corr(axis_x, axis_y), _row_corr(axis_x, axis_z), _row_corr(axis_y, axis_z),
        np.log((rms_x + epsilon) / (rms_y + epsilon)),
        np.log((rms_x + epsilon) / (rms_z + epsilon)),
        np.log((rms_y + epsilon) / (rms_z + epsilon)), high_frequency_kurtosis,
        order_2_3_4, order_5_18, order_18p5_21p5, high_order_ratio,
        band_2000, band_2200, band_2400, band_3200, band_400, band_5000,
    ], axis=1).astype(np.float32)
    return np.nan_to_num(features)


class RobustBlockStandardizer:
    def __init__(self, block_sizes: list[int], l2_per_block: bool = True, clip: float = 5.0):
        self.block_sizes = [int(s) for s in block_sizes]
        self.l2_per_block = bool(l2_per_block)
        self.clip = float(clip)
        self.median: np.ndarray | None = None
        self.iqr: np.ndarray | None = None

    @classmethod
    def from_state(cls, state: dict) -> "RobustBlockStandardizer":
        obj = cls(state["block_sizes"], bool(state.get("l2_per_block", True)), float(state.get("clip", 5.0)))
        obj.median = np.asarray(state["median"], dtype=np.float32)
        obj.iqr = np.asarray(state["iqr"], dtype=np.float32)
        return obj

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.median is not None and self.iqr is not None, "standardizer not fitted"
        z = (x - self.median) / self.iqr
        if self.clip > 0:
            z = np.clip(z, -self.clip, self.clip)
        if self.l2_per_block:
            parts: list[np.ndarray] = []
            start = 0
            for size in self.block_sizes:
                block = z[:, start:start + size]
                norm = np.linalg.norm(block, axis=1, keepdims=True) + 1e-8
                parts.append(block / norm)
                start += size
            z = np.concatenate(parts, axis=1)
        return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Autoencoder encoders (rebuilt to load bundle state dicts).
# ---------------------------------------------------------------------------
def _conv1d_ae_encoder(in_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv1d(16, 8, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
    )


def _wdcnn_encoder(in_ch: int, latent_ch: int = 16) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, 16, 64, 16, 24, bias=False),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(16, 32, 8, 4, 2, bias=False),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, latent_ch, 4, 2, 0, bias=False),
        nn.BatchNorm1d(latent_ch),
        nn.ReLU(),
    )


def rebuild_encoder(arch: str, in_ch: int, latent_ch: int = 8) -> nn.Sequential:
    if arch == "conv1d":
        return _conv1d_ae_encoder(in_ch)
    if arch == "wdcnn":
        return _wdcnn_encoder(in_ch, latent_ch)
    raise ValueError(f"Unknown feature-extractor AE arch: {arch}")


def load_bundle(bundle_path: str) -> dict:
    return torch.load(bundle_path, map_location="cpu", weights_only=False)


def speed_label_to_rpm(speed: object) -> float:
    numbers = re.findall(r"[-+]?\d*\.?\d+", str(speed))
    return float(numbers[0]) if numbers else float("nan")


@torch.no_grad()
def pooled_ae_latent(encoder: nn.Module, windows_ntc: np.ndarray, device: torch.device, batch: int = 64) -> np.ndarray:
    """Run an AE encoder over raw windows (N, T, C) -> GAP+GMP pooled latents (N, 2*latent_ch)."""
    signal = torch.from_numpy(np.ascontiguousarray(windows_ntc)).permute(0, 2, 1).contiguous().float()
    gap = nn.AdaptiveAvgPool1d(1)
    gmp = nn.AdaptiveMaxPool1d(1)
    chunks: list[np.ndarray] = []
    for start in range(0, signal.shape[0], batch):
        feature_map = encoder(signal[start:start + batch].to(device))
        pooled = torch.cat([gap(feature_map).flatten(1), gmp(feature_map).flatten(1)], dim=1)
        chunks.append(pooled.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def compute_bundle_features(bundle: dict, windows_ntc: np.ndarray, rpm: np.ndarray, device: torch.device | str = "cpu") -> np.ndarray:
    """Compute the design's standardized feature vector (N, D) for a handcrafted /
    hybrid bundle: [handcrafted, ae-latent] blocks -> robust standardize."""
    device = torch.device(device)
    scaler = RobustBlockStandardizer.from_state(bundle["standardizer"])
    blocks: list[np.ndarray] = []
    if bool(bundle.get("has_handcrafted")):
        fs = float(bundle.get("sampling_rate_hz", SAMPLING_RATE_HZ))
        if int(bundle.get("handcrafted_dim", 16)) == 22 or bundle.get("design") == "handcrafted22":
            if tuple(bundle.get("handcrafted_feature_names", ())) != HANDCRAFTED22_FEATURE_NAMES:
                raise ValueError("Handcrafted22 bundle feature names/order do not match the supported schema.")
            blocks.append(handcrafted22_features(windows_ntc, rpm, fs=fs))
        else:
            blocks.append(handcrafted_features(windows_ntc, rpm, fs=fs))
    ae = bundle.get("ae")
    if ae is not None and bundle.get("encoder_state_dict") is not None:
        encoder = rebuild_encoder(ae["arch"], int(ae["in_ch"]), int(ae["latent_ch"])).to(device)
        encoder.load_state_dict(bundle["encoder_state_dict"])
        encoder.eval()
        blocks.append(pooled_ae_latent(encoder, windows_ntc, device))
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    return scaler.transform(features)


# ---------------------------------------------------------------------------
# Backbones.
# ---------------------------------------------------------------------------
class BundleEncoderBackbone(BaseBackbone):
    """Frozen conv-autoencoder encoder loaded from a design bundle."""

    def __init__(self, bundle_path: str, in_channels: int = 2, load_pretrained: bool = True, freeze: bool = True):
        super().__init__()
        bundle = load_bundle(bundle_path)
        ae = bundle.get("ae")
        if ae is None:
            raise ValueError(f"Bundle '{bundle_path}' has no AE encoder; use PrecomputedFeatureBackbone instead.")
        self.encoder = rebuild_encoder(str(ae["arch"]), int(ae.get("in_ch", in_channels)), int(ae.get("latent_ch", 8)))
        if load_pretrained and bundle.get("encoder_state_dict") is not None:
            self.encoder.load_state_dict(bundle["encoder_state_dict"])
        self.output_dim = int(ae.get("latent_channels", ae.get("latent_ch", 8)))
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class PrecomputedFeatureBackbone(BaseBackbone):
    """Identity backbone over datamodule-precomputed feature vectors (N, D, 1)."""

    def __init__(self, bundle_path: str, in_channels: int | None = None, load_pretrained: bool = True, freeze: bool = True):
        super().__init__()
        bundle = load_bundle(bundle_path)
        self.output_dim = int(bundle["feature_dim"])
        self.pooled_output_dim = self.output_dim
        self.identity = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        if inputs.shape[1] != self.output_dim:
            raise ValueError(
                f"PrecomputedFeatureBackbone expected {self.output_dim} channels but got {inputs.shape[1]}. "
                f"Set datamodule.config.feature_bundle_path to the matching bundle so features are precomputed."
            )
        return self.identity(inputs)
