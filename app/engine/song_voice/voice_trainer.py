"""Fine-tune Seed-VC's singing model on one person's voice.

Port of seed-vc/train.py (the upstream few-shot recipe), trimmed for use
inside ComfyUI: it reuses the frozen encoders that inference already loaded,
reads models from app/models/seed_vc instead of re-downloading them, and
works on clips handed over in memory.

Each step: OpenVoice re-voices your clip as a random stranger, and the DiT
learns to rebuild *your* mel spectrogram from the stranger's content plus a
slice of your real audio as the prompt. That is exactly the job it does at
song time, so a few hundred steps on a talking clip noticeably sharpen the
likeness.
"""
import os
import random

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def chop_clip(wave, sr, min_s=3.0, max_s=10.0, top_db=40):
    """Split a talking clip on its pauses into 3-10 s pieces with the long silences removed."""
    import librosa
    import numpy as np
    w = wave.numpy()
    spans = librosa.effects.split(w, top_db=top_db, frame_length=2048, hop_length=512)
    pad = int(0.08 * sr)
    pieces, current = [], []
    length = 0
    for a, b in spans:
        a, b = max(0, a - pad), min(len(w), b + pad)
        part = w[a:b]
        while len(part) > max_s * sr:          # one long breathless run: cut it
            pieces.append(part[:int(max_s * sr)])
            part = part[int(max_s * sr):]
        if length + len(part) > max_s * sr and length >= min_s * sr:
            pieces.append(np.concatenate(current))
            current, length = [], 0
        current.append(part)
        length += len(part)
    if current:
        tail = np.concatenate(current)
        if len(tail) >= min_s * sr or not pieces:
            pieces.append(tail)
        else:
            pieces[-1] = np.concatenate([pieces[-1], tail])
    return [torch.from_numpy(p).float() for p in pieces if len(p) >= 1.0 * sr]


class VoiceTrainer:
    def __init__(self, seed_vc, base_ckpt, openvoice_dir, se_db_path):
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from modules.openvoice.api import ToneColorConverter
        from optimizers import build_optimizer

        self.svc = seed_vc
        self.device = seed_vc.device
        self.model = build_model(seed_vc.params, stage="DiT")
        self.model, _, _, _ = load_checkpoint(self.model, None, base_ckpt, load_only_params=True,
                                              ignore_modules=[], is_distributed=False)
        for key in self.model:
            self.model[key].to(self.device)
        self.optimizer = build_optimizer({k: self.model[k] for k in self.model}, lr=1e-5)

        self.converter = ToneColorConverter(os.path.join(openvoice_dir, "config.json"), device=str(self.device))
        self.converter.load_ckpt(os.path.join(openvoice_dir, "checkpoint.pth"))
        self.converter.model.eval()
        self.se_db = torch.load(se_db_path, map_location="cpu")

    def _batch(self, pieces, batch_size, replay=None):
        chosen = random.sample(pieces, batch_size) if len(pieces) >= batch_size else \
            [random.choice(pieces) for _ in range(batch_size)]
        if replay and batch_size > 1:
            chosen[-1] = random.choice(replay)
        mels = [self.svc.to_mel(p[None].to(self.device))[0] for p in chosen]
        n = len(chosen)
        max_w = max(p.numel() for p in chosen)
        max_m = max(m.size(1) for m in mels)
        waves = torch.zeros(n, max_w, device=self.device)
        mel = torch.full((n, mels[0].size(0), max_m), -10.0, device=self.device)
        wave_len = torch.zeros(n, dtype=torch.long, device=self.device)
        mel_len = torch.zeros(n, dtype=torch.long, device=self.device)
        for i, (p, m) in enumerate(zip(chosen, mels)):
            waves[i, :p.numel()] = p.to(self.device)
            mel[i, :, :m.size(1)] = m
            wave_len[i], mel_len[i] = p.numel(), m.size(1)
        return waves, mel, wave_len, mel_len

    def _semantic(self, waves_16k):
        svc = self.svc
        inputs = svc.whisper_features([w.cpu().numpy() for w in waves_16k], return_tensors="pt",
                                      return_attention_mask=True, sampling_rate=16000)
        feats = svc.whisper._mask_input_features(inputs.input_features,
                                                 attention_mask=inputs.attention_mask).to(self.device)
        with torch.no_grad():
            out = svc.whisper.encoder(feats.to(svc.whisper.encoder.dtype), return_dict=True)
        return out.last_hidden_state.float()[:, :waves_16k.size(-1) // 320 + 1]

    def step(self, waves, mels, wave_len, mel_len):
        sr, B = self.svc.sr, waves.size(0)
        with torch.no_grad():
            w22 = torchaudio.functional.resample(waves, sr, 22050)
            l22 = (wave_len.float() * 22050 / sr).long()
            se = self.converter.extract_se(w22, l22)
            target_se = self.se_db[torch.randint(0, len(self.se_db), (B,))].to(self.device)
            stranger = self.converter.convert(w22, l22, se, target_se).squeeze(1)
            stranger = torchaudio.functional.resample(stranger, 22050, sr)

            w16 = torchaudio.functional.resample(waves, sr, 16000)
            l16 = (wave_len.float() * 16000 / sr).long()
            s16 = torchaudio.functional.resample(stranger, sr, 16000)
            s_ori, s_alt = self._semantic(w16), self._semantic(s16)
            f0 = self.svc.rmvpe.infer_from_audio_batch(w16).float().to(self.device)

            style = []
            for i in range(B):
                fb = kaldi.fbank(w16[i:i + 1, :l16[i]], num_mel_bins=80, dither=0, sample_frequency=16000)
                style.append(self.svc.campplus((fb - fb.mean(dim=0, keepdim=True))[None]))
            style = torch.cat(style, dim=0)

        alt_cond = self.model.length_regulator(s_alt, ylens=mel_len, f0=f0)[0]
        ori_cond = self.model.length_regulator(s_ori, ylens=mel_len, f0=f0)[0]

        # A random leading slice is the "prompt" (your real audio), the rest must be rebuilt.
        prompt_len = (torch.rand(B, device=self.device) * (mel_len - 1)).floor().long()
        prompt_len[torch.rand(B, device=self.device) < 0.1] = 0
        cond = alt_cond.clone()
        for i in range(B):
            cond[i, :prompt_len[i]] = ori_cond[i, :prompt_len[i]]
        common = min(mels.size(2), cond.size(1))
        loss, _ = self.model.cfm(mels[:, :, :common], mel_len.clamp(max=common), prompt_len,
                                 cond[:, :common], style)

        self.optimizer.zero_grad()
        loss.backward()
        keys = ("cfm",) if self.freeze_regulator else ("cfm", "length_regulator")
        for key in keys:
            torch.nn.utils.clip_grad_norm_(self.model[key].parameters(), 10.0)
            self.optimizer.step(key)
            self.optimizer.scheduler(key=key)
        return loss.item()

    def train(self, pieces, steps, batch_size=2, on_step=None, replay=None, freeze_regulator=True):
        """replay: clips of other voices mixed into every batch so the model keeps
        handling general content. freeze_regulator: leave the words->content
        mapping untouched and learn only timbre (the DiT)."""
        self.freeze_regulator = freeze_regulator
        self.model.length_regulator.requires_grad_(not freeze_regulator)
        for key in self.model:
            self.model[key].train()
        if freeze_regulator:
            self.model.length_regulator.eval()
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
        ema = None
        for i in range(steps):
            loss = self.step(*self._batch(pieces, batch_size, replay))
            ema = loss if ema is None else ema * 0.95 + loss * 0.05
            if on_step is not None:
                on_step(i + 1, ema)
        for key in self.model:
            self.model[key].eval()
        return ema

    def save(self, path):
        # fp16 halves the file (~400 MB); load_state_dict casts back to fp32.
        net = {k: {n: t.half() if t.is_floating_point() else t for n, t in self.model[k].state_dict().items()}
               for k in self.model}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({"net": net}, tmp)
        os.replace(tmp, path)
