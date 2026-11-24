"""Pure-PyTorch BS-RoFormer & Mel-Band RoFormer audio source separation.

Inference-only implementation compatible with ZFTurbo / ViperX / aufr33
Music-Source-Separation-Training checkpoints, using only libraries already in
OS Studio's environment (torch, einops, librosa, pyyaml, huggingface_hub).

Architecture based on lucidrains/BS-RoFormer and ZFTurbo/Music-Source-Separation-Training
(MIT License).
"""
from __future__ import annotations

from functools import partial
import hashlib
import os
import threading

from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from librosa import filters
import torch
from torch import einsum, nn
import torch.nn.functional as F
import yaml

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _INFERENCE_SDPA_BACKENDS = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False


# ------------------------------------------------------------------ models
# Default Hugging Face sources (downloaded once into <models_dir>/seed_vc/roformer)
ROFORMER_MODELS = {
    "vocals": {
        "arch": "bs_roformer",
        "repo": "Eddycrack864/Music-Source-Separation-Training",
        "ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "ckpt_sha256": "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
        "config": "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "config_sha256": "2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f",
        "ckpt_mirrors": [
            ("url", "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
            ("hf", "Derur/UVR-models", "!More-URV-models/Best_model_creator/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
        ],
        "config_mirrors": [
            ("hf", "Derur/UVR-models", "!More-URV-models/Best_model_creator/model_bs_roformer_ep_317_sdr_12.9755.yaml"),
        ],
        "folder": "roformer/bs_roformer_vocals_1297",
    },
    "karaoke": {
        "arch": "mel_band_roformer",
        "repo": "shiromiya/audio-separation-models",
        "ckpt": "mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "ckpt_sha256": "1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813",
        "config": "mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
        "config_sha256": "73c47d6209d25e7e598f8d7e90a604f4ce8617e9a9484e277b286ae7e125e1fd",
        "ckpt_mirrors": [
            ("url", "https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
            ("url", "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
        ],
        "config_mirrors": [],
        "folder": "roformer/mel_band_roformer_karaoke",
    },
}


class _TupleSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader that also accepts !!python/tuple used in MSST configs."""


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_TupleSafeLoader)


# ------------------------------------------------------------------ RoPE
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def _apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        dtype = t.dtype
        t_f = t.float()
        freqs_f = freqs.float()
        rot_dim = freqs_f.shape[-1]
        t_middle = t_f[..., :rot_dim]
        t_right = t_f[..., rot_dim:]
        t_transformed = (t_middle * freqs_f.cos()) + (_rotate_half(t_middle) * freqs_f.sin())
        out = torch.cat((t_transformed, t_right), dim=-1) if t_right.numel() else t_transformed
        return out.to(dtype)


class RotaryEmbedding(nn.Module):
    """Drop-in compatible with lucidrains rotary_embedding_torch.RotaryEmbedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        seq = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
        with torch.amp.autocast("cuda", enabled=False):
            freqs = einsum("n, f -> n f", seq, self.freqs.float())
            freqs = repeat(freqs, "n f -> n (f r)", r=2)
        return _apply_rotary_emb(freqs, t)


# ------------------------------------------------------------------ shared blocks
def _pack_one(t, pattern):
    return pack([t], pattern)


def _unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def _l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Attend(nn.Module):
    def __init__(self, dropout: float = 0.0, flash: bool = True, scale: float | None = None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

    def forward(self, q, k, v):
        if self.flash:
            if self.scale is not None:
                q = q * (self.scale / (q.shape[-1] ** -0.5))
            if q.is_cuda and not self.training and _HAS_SDPA_KERNEL:
                with sdpa_kernel(_INFERENCE_SDPA_BACKENDS, set_priority=True):
                    return F.scaled_dot_product_attention(q, k, v)
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: RotaryEmbedding | None = None,
        flash: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        out = self.attend(q, k, v)
        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 32, heads: int = 8, scale: float = 8.0,
                 flash: bool = False, dropout: float = 0.0):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.attend = Attend(scale=scale, dropout=dropout, flash=flash)
        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x)
        q, k = _l2norm(q), _l2norm(k)
        q = q * self.temperature.exp()
        return self.to_out(self.attend(q, k, v))


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: RotaryEmbedding | None = None,
        flash_attn: bool = True,
        linear_attn: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads,
                                       dropout=attn_dropout, flash=flash_attn)
            else:
                attn = Attention(dim=dim, dim_head=dim_head, heads=heads,
                                 dropout=attn_dropout, rotary_embed=rotary_embed,
                                 flash=flash_attn)
            self.layers.append(nn.ModuleList([attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]))
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            for dim_in in dim_inputs
        ])

    def forward(self, x):
        parts = x.split(self.dim_inputs, dim=-1)
        return torch.stack([fn(p) for p, fn in zip(parts, self.to_features)], dim=-2)


def _mlp(dim_in: int, dim_out: int, dim_hidden: int, num_hidden: int) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden,) * num_hidden), dim_out)
    net = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            net.append(nn.Tanh())
    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], num_hidden: int, mlp_expansion_factor: int = 4):
        super().__init__()
        self.dim_inputs = dim_inputs
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList([
            nn.Sequential(_mlp(dim, dim_in * 2, dim_hidden, num_hidden), nn.GLU(dim=-1))
            for dim_in in dim_inputs
        ])

    def forward(self, x):
        bands = x.unbind(dim=-2)
        return torch.cat([fn(b) for b, fn in zip(bands, self.to_freqs)], dim=-1)


# ------------------------------------------------------------------ BSRoformer
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRoformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        linear_transformer_depth: int = 0,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        flash_attn: bool = True,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        zero_dc: bool = True,
        mask_estimator_depth: int = 2,
        mlp_expansion_factor: int = 4,
        skip_connection: bool = False,
        **ignored,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection
        self.zero_dc = zero_dc

        t_kwargs = dict(dim=dim, heads=heads, dim_head=dim_head,
                        attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                        flash_attn=flash_attn, norm_output=False)
        time_rot = RotaryEmbedding(dim=dim_head)
        freq_rot = RotaryEmbedding(dim=dim_head)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            mods = []
            if linear_transformer_depth > 0:
                mods.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **t_kwargs))
            mods.append(Transformer(depth=time_transformer_depth, rotary_embed=time_rot, **t_kwargs))
            mods.append(Transformer(depth=freq_transformer_depth, rotary_embed=freq_rot, **t_kwargs))
            self.layers.append(nn.ModuleList(mods))

        self.final_norm = RMSNorm(dim)
        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length,
                                win_length=stft_win_length, normalized=stft_normalized)
        self.stft_window_fn = partial(torch.hann_window, stft_win_length)
        freqs = stft_n_fft // 2 + 1
        freqs_per_bands = tuple(freqs_per_bands)
        assert sum(freqs_per_bands) == freqs, f"freqs_per_bands sums to {sum(freqs_per_bands)}, expected {freqs}"

        dim_inputs = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        # BSRoformer uses (depth - 1) hidden layers in MaskEstimator MLP
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dim_inputs,
                          num_hidden=max(0, mask_estimator_depth - 1),
                          mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(num_stems)
        ])

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        raw_audio, packed = _pack_one(raw_audio, "* t")
        with torch.amp.autocast("cuda", enabled=False):
            win = self.stft_window_fn(device=device)
            stft_repr = torch.stft(raw_audio.float(), **self.stft_kwargs, window=win, return_complex=True)
            stft_repr = torch.view_as_real(stft_repr)
        stft_repr = _unpack_one(stft_repr, packed, "* f t c")
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")
        x = rearrange(stft_repr, "b f t c -> b t (f c)")
        x = self.band_split(x)

        store = [None] * len(self.layers)
        for i, block in enumerate(self.layers):
            if len(block) == 3:
                lin_t, time_t, freq_t = block
                x, ft_ps = pack([x], "b * d")
                x = lin_t(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_t, freq_t = block
            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_t(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_t(x)
            (x,) = unpack(x, ps, "* f d")
            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)
        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        with torch.amp.autocast("cuda", enabled=False):
            stft_c = torch.view_as_complex(rearrange(stft_repr.float(), "b f t c -> b 1 f t c").contiguous())
            mask_c = torch.view_as_complex(mask.float().contiguous())
            out_stft = rearrange(stft_c * mask_c, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
            if self.zero_dc:
                out_stft[:, 0] = 0.0
            recon = torch.istft(out_stft, **self.stft_kwargs, window=win, return_complex=False,
                                length=raw_audio.shape[-1])
        return rearrange(recon, "(b n s) t -> b n s t", s=self.audio_channels, n=self.num_stems)


# ------------------------------------------------------------------ MelBandRoformer
class MelBandRoformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        linear_transformer_depth: int = 0,
        num_bands: int = 60,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        flash_attn: bool = True,
        dim_freqs_in: int = 1025,
        sample_rate: int = 44100,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        zero_dc: bool = True,
        mask_estimator_depth: int = 1,
        mlp_expansion_factor: int = 4,
        skip_connection: bool = False,
        match_input_audio_length: bool = True,
        **ignored,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection
        self.zero_dc = zero_dc
        self.match_input_audio_length = match_input_audio_length

        # MelBandRoformer uses norm_output=True (default) on each Transformer
        t_kwargs = dict(dim=dim, heads=heads, dim_head=dim_head,
                        attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                        flash_attn=flash_attn, norm_output=True)
        time_rot = RotaryEmbedding(dim=dim_head)
        freq_rot = RotaryEmbedding(dim=dim_head)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            mods = []
            if linear_transformer_depth > 0:
                mods.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **t_kwargs))
            mods.append(Transformer(depth=time_transformer_depth, rotary_embed=time_rot, **t_kwargs))
            mods.append(Transformer(depth=freq_transformer_depth, rotary_embed=freq_rot, **t_kwargs))
            self.layers.append(nn.ModuleList(mods))

        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length,
                                win_length=stft_win_length, normalized=stft_normalized)
        self.stft_window_fn = partial(torch.hann_window, stft_win_length)
        freqs = stft_n_fft // 2 + 1

        mel_fb = torch.from_numpy(filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands))
        mel_fb[0, 0] = 1.0
        mel_fb[-1, -1] = 1.0
        freqs_per_band = mel_fb > 0

        repeated = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated[freqs_per_band]
        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2) * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)
        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")
        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        dim_inputs = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())
        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        # MelBandRoformer uses `mask_estimator_depth` hidden layers in MaskEstimator MLP
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dim_inputs,
                          num_hidden=mask_estimator_depth,
                          mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(num_stems)
        ])

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        batch, channels, raw_len = raw_audio.shape
        raw_audio, packed = _pack_one(raw_audio, "* t")
        with torch.amp.autocast("cuda", enabled=False):
            win = self.stft_window_fn(device=device)
            stft_repr = torch.stft(raw_audio.float(), **self.stft_kwargs, window=win, return_complex=True)
            stft_repr = torch.view_as_real(stft_repr)
        stft_repr = _unpack_one(stft_repr, packed, "* f t c")
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        batch_arange = torch.arange(batch, device=device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = rearrange(x, "b f t c -> b t (f c)")
        x = self.band_split(x)

        store = [None] * len(self.layers)
        for i, block in enumerate(self.layers):
            if len(block) == 3:
                lin_t, time_t, freq_t = block
                x, ft_ps = pack([x], "b * d")
                x = lin_t(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_t, freq_t = block
            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_t(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_t(x)
            (x,) = unpack(x, ps, "* f d")
            if self.skip_connection:
                store[i] = x

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        with torch.amp.autocast("cuda", enabled=False):
            stft_c = torch.view_as_complex(rearrange(stft_repr.float(), "b f t c -> b 1 f t c").contiguous())
            masks_c = torch.view_as_complex(masks.float().contiguous()).type(stft_c.dtype)
            scatter_idx = repeat(self.freq_indices, "f -> b n f t", b=batch, n=self.num_stems, t=stft_c.shape[-1])
            expanded = repeat(stft_c, "b 1 ... -> b n ...", n=self.num_stems)
            summed = torch.zeros_like(expanded).scatter_add_(2, scatter_idx, masks_c)
            denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels).float()
            averaged = summed / denom.clamp(min=1e-8)
            out_stft = rearrange(stft_c * averaged, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
            if self.zero_dc:
                out_stft[:, 0] = 0.0
            recon = torch.istft(out_stft, **self.stft_kwargs, window=win, return_complex=False,
                                length=raw_len if self.match_input_audio_length else None)
        recon = rearrange(recon, "(b n s) t -> b n s t", b=batch, s=self.audio_channels, n=self.num_stems)
        return recon


# ------------------------------------------------------------------ loader & overlap-add inference
_LOAD_LOCK = threading.Lock()


def build_roformer(arch: str, config: dict) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    if arch == "bs_roformer":
        return BSRoformer(**model_cfg)
    if arch == "mel_band_roformer":
        return MelBandRoformer(**model_cfg)
    raise ValueError(f"Unknown RoFormer architecture: {arch}")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_file(path: str, expected_sha256: str | None) -> tuple[bool, str]:
    if not os.path.isfile(path) or os.path.getsize(path) <= 100:
        return False, "missing or empty"
    if not expected_sha256:
        return True, ""
    actual = _file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        return False, f"SHA-256 mismatch (got {actual[:12]}…, expected {expected_sha256[:12]}…)"
    return True, ""


def _download_url(url: str, dest: str, expected_sha256: str | None = None) -> None:
    import urllib.request
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "OS-Studio/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        ok, reason = _verify_file(tmp, expected_sha256)
        if not ok:
            raise RuntimeError(f"Mirror file {os.path.basename(dest)} failed verification: {reason}")
        os.replace(tmp, dest)
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _ensure_file(repo: str, remote: str, local: str, target_dir: str, mirrors: list,
                 expected_sha256: str | None = None) -> None:
    force_redownload = False
    if os.path.isfile(local):
        ok, reason = _verify_file(local, expected_sha256)
        if ok:
            return
        print(f"[SongVoice] cached {os.path.basename(local)} is corrupt ({reason}); removing and re-downloading",
              flush=True)
        try:
            os.remove(local)
        except OSError:
            pass
        force_redownload = True

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import hf_hub_download
    attempts = [("hf", repo, remote)] + list(mirrors or [])
    last_err = None
    for entry in attempts:
        try:
            if entry[0] == "hf":
                _, r_id, r_file = entry
                print(f"[SongVoice] downloading {r_id}/{r_file}", flush=True)
                downloaded = hf_hub_download(r_id, r_file, local_dir=target_dir,
                                             force_download=force_redownload)
                ok, reason = _verify_file(downloaded, expected_sha256)
                if not ok:
                    try:
                        os.remove(downloaded)
                    except OSError:
                        pass
                    raise RuntimeError(f"Downloaded {os.path.basename(local)} failed verification: {reason}")
                if os.path.abspath(downloaded) != os.path.abspath(local) and os.path.isfile(downloaded):
                    os.replace(downloaded, local)
                return
            else:
                _, url = entry
                print(f"[SongVoice] downloading {os.path.basename(local)} from mirror", flush=True)
                _download_url(url, local, expected_sha256=expected_sha256)
                return
        except Exception as exc:
            last_err = exc
            print(f"[SongVoice] download attempt failed ({exc}); trying fallback...", flush=True)
    raise RuntimeError(f"Could not download verified {os.path.basename(local)}: {last_err}")


def load_roformer_model(kind: str, model_dir: str, device: torch.device) -> tuple[nn.Module, dict]:
    """Download (once, SHA-256 verified) and load a RoFormer separator ('vocals' or 'karaoke')."""
    spec = ROFORMER_MODELS[kind]
    target_dir = os.path.join(model_dir, spec["folder"])
    os.makedirs(target_dir, exist_ok=True)
    ckpt_path = os.path.join(target_dir, os.path.basename(spec["ckpt"]))
    cfg_path = os.path.join(target_dir, os.path.basename(spec["config"]))

    with _LOAD_LOCK:
        _ensure_file(spec["repo"], spec["config"], cfg_path, target_dir,
                     spec.get("config_mirrors", []), expected_sha256=spec.get("config_sha256"))
        _ensure_file(spec["repo"], spec["ckpt"], ckpt_path, target_dir,
                     spec.get("ckpt_mirrors", []), expected_sha256=spec.get("ckpt_sha256"))

    config = load_yaml_config(cfg_path)
    model = build_roformer(spec["arch"], config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state" in state and isinstance(state["state"], dict):
        state = state["state"]
    elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # Strip optional DataParallel 'module.' prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, config


def _windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """Linear fade-in / fade-out window for seamless overlap-add (MSST recipe)."""
    fade_size = max(1, min(fade_size, window_size // 2))
    fadein = torch.linspace(0.0, 1.0, fade_size)
    fadeout = torch.linspace(1.0, 0.0, fade_size)
    window = torch.ones(window_size)
    window[:fade_size] = fadein
    window[-fade_size:] = fadeout
    return window


@torch.no_grad()
def separate_roformer(
    model: nn.Module,
    config: dict,
    mix_stereo_44k: torch.Tensor,
    device: torch.device,
    num_overlap: int = 2,
    pbar=None,
    check_cancel=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run overlap-add RoFormer separation on stereo 44.1 kHz tensor [2, T].

    Returns (target_stem [2, T], residual [2, T]) on CPU as float32.
    For 'vocals': (all_vocals, pure_instrumental).
    For 'karaoke': (lead_vocals, backing_vocals).
    """
    audio_cfg = config.get("audio", {})
    chunk_size = int(audio_cfg.get("chunk_size", 352800))
    num_overlap = max(2, int(num_overlap))
    step = chunk_size // num_overlap
    fade_size = chunk_size // 10
    border = chunk_size - step

    mix = mix_stereo_44k.float().cpu()
    orig_len = mix.shape[-1]

    # Reflection-pad boundaries so chunk edges never see abrupt silence
    if orig_len > 2 * border and border > 0:
        padded = F.pad(mix.unsqueeze(0), (border, border), mode="reflect").squeeze(0)
    else:
        border = 0
        padded = mix

    total_len = padded.shape[-1]
    result = torch.zeros_like(padded)
    counter = torch.zeros_like(padded)
    window = _windowing_array(chunk_size, fade_size)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = device.type == "cuda"

    starts = list(range(0, total_len, step))
    # Avoid trailing tiny sliver already covered by the previous full chunk
    if len(starts) > 1 and starts[-1] + step >= total_len and starts[-2] + chunk_size >= total_len:
        starts.pop()

    for idx, i in enumerate(starts):
        if check_cancel is not None:
            check_cancel()
        part = padded[:, i : i + chunk_size]
        part_len = part.shape[-1]
        if part_len < chunk_size:
            pad_mode = "reflect" if part_len > chunk_size // 2 + 1 else "constant"
            part_in = F.pad(part.unsqueeze(0), (0, chunk_size - part_len), mode=pad_mode).squeeze(0)
        else:
            part_in = part

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            est = model(part_in.unsqueeze(0).to(device))
        # Both BSRoformer and MelBandRoformer return [B, N, C, T]
        est = est[0, 0, :, :part_len].float().cpu()

        w = window[:part_len].clone()
        if i == 0:
            w[:fade_size] = 1.0
        if i + chunk_size >= total_len:
            w[-fade_size:] = 1.0
        result[:, i : i + part_len] += est * w
        counter[:, i : i + part_len] += w

        if pbar is not None:
            pbar.update_absolute(idx + 1, len(starts))

    target = result / counter.clamp(min=1e-8)
    if border > 0:
        target = target[:, border : border + orig_len]
    else:
        target = target[:, :orig_len]

    residual = mix - target
    return target, residual
