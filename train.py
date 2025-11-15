#!/usr/bin/env python3
"""
Low-Rank Transformer Language Model Trainer
Main training script with fixed syntax and enhanced functionality
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
import os
import sys
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, logging as hf_log
from tqdm.auto import tqdm

from __future__ import annotations
import argparse, json, math, pathlib, random, time, os, sys
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, logging as hf_log
from tqdm.auto import tqdm

# ───────────────────────── Globals ─────────────────────────
hf_log.set_verbosity_error()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ───────────────────────── Determinism ─────────────────────────
def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass

# Tokenizer (default DeepSeek V3.2 Exp)
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "deepseek-ai/DeepSeek-V3.2-Exp")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
VOCAB = max(tok.get_vocab().values()) + 1
BLANK = tok.pad_token_id
EOS = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

PRESETS: Dict[str, Dict[str, int]] = {
    "small":   dict(d=512, layers=8,  heads=16, rank=64),
    "smallx2": dict(d=512, layers=16, heads=16, rank=64),
    "base":    dict(d=768, layers=12, heads=24, rank=96),
    # requested: base version with 17 layers
    "base17":  dict(d=768, layers=17, heads=24, rank=96),
}

DEFAULT_BLOCK = 576
LR_CORE, LR_HEAD = 5e-5, 2e-4
DEFAULT_SAVE_SEC = 24 * 3600
CKDIR = pathlib.Path("ckpts_joint")

# Defaults for automatic after-SFT if user only sets --after_sft_steps
DEFAULT_AFTER_SFT_SOURCES = "mlabonne/opc-sft-stage2-chat,HuggingFaceH4/ultrachat_200k"
DEFAULT_AFTER_SFT_BLOCK = 1120

# New: default pretrain sources (replaces SlimPajama/C4)
DEFAULT_PRETRAIN_SOURCES = "HuggingFaceFW/fineweb-edu,togethercomputer/RedPajama-Data-1T,oscar-corpus/OSCAR-2201:en"

# ───────────────────────── Utilities ─────────────────────────
def rng_state():
    if DEV.type == "cuda":
        try:
            return torch.cuda.get_rng_state(DEV)
        except TypeError:
            return torch.cuda.get_rng_state()
    return torch.get_rng_state()

def _is_probably_ckpt(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and path.suffix == ".pt" and not path.name.endswith(".pt.tmp") and path.stat().st_size > (1<<20)
    except Exception:
        return False

def _resolve_ckpt(path: pathlib.Path) -> pathlib.Path | None:
    try:
        if path.is_dir():
            cands = sorted([p for p in path.glob("*.pt") if _is_probably_ckpt(p)],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None
        if path.suffix == ".tmp":
            solid = path.with_suffix("")
            return solid if _is_probably_ckpt(solid) else _resolve_ckpt(path.parent)
        return path if _is_probably_ckpt(path) else _resolve_ckpt(path.parent)
    except Exception:
        return None

def _try_load(path: pathlib.Path, map_location="cpu"):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ckpt-skip] {path} not usable: {e}")
        return None

# ───────────────────────── AMP helper ─────────────────────────
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _supports_fp8() -> bool:
    return hasattr(torch, "float8_e4m3fn")

def _auto_amp_dtype(prefer_fp8: bool = False):
    if DEV.type != "cuda":
        return torch.float32
    if prefer_fp8 and _supports_fp8():
        return torch.float8_e4m3fn
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    except Exception:
        return torch.float16

def amp(enabled: bool, prefer_fp8: bool = False):
    if not (enabled and DEV.type == "cuda"):
        return nullcontext()
    return _ac(device_type="cuda", dtype=_auto_amp_dtype(prefer_fp8=prefer_fp8))

# ───────────────────────── Chat helpers ─────────────────────────
def _coerce_role(r: str) -> str:
    r = (r or "").lower()
    if r in {"user", "human", "customer", "questioner"}:
        return "user"
    if r in {"assistant", "gpt", "bot", "agent", "answerer"}:
        return "assistant"
    if r in {"system", "context", "instruction"}:
        return "system"
    return r or "user"

def _render_chat_text_from_ex(ex: dict, messages_key: str, add_generation_prompt: bool) -> Optional[str]:
    msgs = ex.get(messages_key)
    if msgs is None:
        for alt in ("conversations", "dialog", "turns"):
            if isinstance(ex.get(alt), list):
                msgs = ex[alt]
                break
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        try:
            norm = []
            for m in msgs:
                role = _coerce_role(m.get("role", "")); content = m.get("content", m.get("text", ""))
                if not isinstance(content, str):
                    continue
                norm.append({"role": role, "content": content})
            if not norm:
                return None
            return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            return None
    for a, b in (("prompt", "response"), ("instruction", "output"), ("question", "answer")):
        if isinstance(ex.get(a), str) and isinstance(ex.get(b), str):
            return f"User: {ex[a]}\nAssistant: {ex[b]}"
    return None

# ───────────────────────── Robust streaming data ─────────────────────────
def _open_stream_one(ds_name: str, seed: int):
    if ":" in ds_name:
        base, config = ds_name.split(":", 1)
    else:
        base, config = ds_name, None
    dc = DownloadConfig(max_retries=5, use_etag=True, resume_download=True)
    if base == "json":
        if not config:
            raise ValueError("Use 'json:/path/to/file.jsonl' or glob like 'json:/data/*.jsonl'")
        data_files = {"train": config}
        ds = load_dataset("json", data_files=data_files, split="train", streaming=True, download_config=dc)
    else:
        if config:
            ds = load_dataset(base, config, split="train", streaming=True, download_config=dc)
        else:
            ds = load_dataset(base, split="train", streaming=True, download_config=dc)
    ds = ds.shuffle(buffer_size=10_000, seed=seed)
    return iter(ds)

def token_stream(args, target: int, seed: int = 42, max_retries: int = 999, *,
                 source: Optional[str] = None, chat: Optional[bool] = None,
                 chat_messages_key: Optional[str] = None, sft_add_generation_prompt: Optional[bool] = None,
                 dataset_field_text: Optional[str] = None):
    ds_names = source if source is not None else args.source
    sources = [s.strip() for s in ds_names.split(",") if s.strip()]
    if not sources:
        # Default replaced: use the three stable sources by default
        sources = [s.strip() for s in DEFAULT_PRETRAIN_SOURCES.split(",") if s.strip()]
    use_chat = args.chat if chat is None else chat
    msg_key  = args.chat_messages_key if chat_messages_key is None else chat_messages_key
    add_gen  = args.sft_add_generation_prompt if sft_add_generation_prompt is None else sft_add_generation_prompt
    text_key = args.dataset_field_text if dataset_field_text is None else dataset_field_text

    src_idx = 0; emitted = 0; it = None; attempts = 0; backoff_base = 2.0
    while emitted < target:
        try:
            if it is None:
                it = _open_stream_one(sources[src_idx], seed)
            ex = next(it)
            text = None
            if isinstance(ex, dict):
                if use_chat:
                    text = _render_chat_text_from_ex(ex, msg_key, add_gen)
                if text is None:
                    if text_key and isinstance(ex.get(text_key), str):
                        text = ex[text_key]
                    elif isinstance(ex.get("text"), str):
                        text = ex["text"]
            if not isinstance(text, str):
                attempts = 0; continue
            enc = tok.encode(text)
            if EOS is not None and (len(enc) == 0 or enc[-1] != EOS):
                enc.append(EOS)
            for t in enc:
                yield t; emitted += 1
                if emitted >= target:
                    return
            attempts = 0
        except StopIteration:
            it = None; src_idx = (src_idx + 1) % len(sources)
        except Exception as e:
            attempts += 1
            sleep_s = min(60.0, backoff_base ** min(attempts, 6))
            print(f"[stream-retry] source={sources[src_idx]} attempts={attempts} sleep={sleep_s:.1f}s reason={type(e).__name__}", flush=True)
            time.sleep(sleep_s); it = None
            if attempts % 5 == 0 and len(sources) > 1:
                src_idx = (src_idx + 1) % len(sources)
            if attempts > max_retries:
                raise

# ───────────────────────── Relative positional bias (ALiBi) ─────────────────────────
def _alibi_slopes(n_heads: int):
    import math
    def pow2slopes(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        vals = pow2slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        vals = pow2slopes(closest)
        extra = pow2slopes(2 * closest)
        vals += extra[0::2][: n_heads - closest]
    return torch.tensor(vals, device=DEV).view(1, n_heads, 1, 1)

def alibi_bias(n_heads: int, n_tokens: int):
    i = torch.arange(n_tokens, device=DEV).view(1, 1, n_tokens, 1)
    j = torch.arange(n_tokens, device=DEV).view(1, 1, 1, n_tokens)
    dist = (j - i).clamp_min(0)
    slopes = _alibi_slopes(n_heads)
    return -slopes * dist

# ───────────────────────── Model components ─────────────────────────
class LowRankMHA(nn.Module):
    def __init__(self, d: int, h: int, r: int, use_relpos: bool = True):
        super().__init__()
        assert d % h == 0, "d must be divisible by number of heads"
        self.h, self.dk = h, d // h
        self.use_relpos = use_relpos
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.U = nn.Parameter(torch.randn(self.dk, r))
        nn.init.orthogonal_(self.U)
        self.proj = nn.Linear(h * r, d, bias=False)
        self.drop = nn.Dropout(0.1)

    def _proj(self, x):
        B, N, _ = x.shape
        return (x.view(B, N, self.h, self.dk).transpose(1, 2) @ self.U)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rel_bias_tokens: Optional[int] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        q = self._proj(self.q(x))
        k_new = self._proj(self.k(x))
        v_new = self._proj(self.v(x))

        if kv_cache is None:
            k, v = k_new, v_new
        else:
            k, v = kv_cache
            if use_cache:
                k = torch.cat([k, k_new], dim=2)
                v = torch.cat([v, v_new], dim=2)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)

        if q.size(2) == k.size(2):
            if self.use_relpos and rel_bias_tokens is not None:
                att = att + alibi_bias(self.h, rel_bias_tokens)
            if mask is not None:
                att = att + mask

        z = (att.softmax(-1) @ v).transpose(1, 2)
        z = z.reshape(x.size(0), x.size(1), -1)
        out = self.drop(self.proj(z))
        return (out, (k, v)) if use_cache else out

class Block(nn.Module):
    def __init__(self, d: int, h: int, r: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = LowRankMHA(d, h, r, use_relpos=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        n = x.size(1)
        if use_cache:
            y, new_kv = self.mha(self.ln1(x), mask, rel_bias_tokens=n if mask is not None else None, kv_cache=kv, use_cache=True)
            x = x + y
            x = x + self.ff(self.ln2(x))
            return x, new_kv
        else:
            x = x + self.mha(self.ln1(x), mask, rel_bias_tokens=n)
            return x + self.ff(self.ln2(x))

class Encoder(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        d, l, h, r = cfg["d"], cfg["layers"], cfg["heads"], cfg["rank"]
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, h, r) for _ in range(l)])
        self.ln = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor],
                kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False):
        x = self.emb(ids)
        if not use_cache:
            for blk in self.blocks:
                x = blk(x, mask)
            return self.ln(x)
        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if (kv_caches is not None) else None
            x, kv_out = blk(x, mask, kv, use_cache=True)
            new_kvs.append(kv_out)
        return self.ln(x), new_kvs

class ARHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, VOCAB)
    def forward(self, h): return self.proj(h)

# ───────────────────────── Masks ─────────────────────────
def causal_mask(n):
    m = torch.full((1, 1, n, n), float("-inf"), device=DEV)
    return torch.triu(m, 1)

# ───────────────────────── Checkpoint helpers ─────────────────────────
def save_ckpt(path: pathlib.Path, core: nn.Module, ar_h: nn.Module,
              opt: torch.optim.Optimizer, scaler: GradScaler, meta: Dict[str, Any]):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {
        "core": core.state_dict(),
        "ar": ar_h.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": meta.get("cfg"),
        "tokenizer_id": TOKENIZER_ID,
        **{k: v for k, v in meta.items() if k not in {"cfg"}},
    }
    torch.save(state, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)
    (path.parent / "latest.json").write_text(json.dumps({"path": str(path), "step": meta["step"]}))
    print(f"\n✓ saved checkpoint {path.name}")

def load_ckpt(path: pathlib.Path, core: nn.Module, ar_h: nn.Module,
              opt: torch.optim.Optimizer, scaler: GradScaler):
    p = _resolve_ckpt(path) or path
    ck = _try_load(p, map_location="cpu")
    if ck is None:
        raise FileNotFoundError(f"No valid checkpoint at {p}")
    core.load_state_dict(ck["core"])
    if "ar" in ck:
        ar_h.load_state_dict(ck["ar"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("seen_tok", 0), ck.get("wall_time", time.time())

def _safe_load_any(path: pathlib.Path, tgt: nn.Module, key: str | None = None, rename: str | None = None):
    p = _resolve_ckpt(path) or path
    if not p or not p.exists(): return 0
    ck = _try_load(p, map_location="cpu")
    if ck is None: return 0
    sd = ck.get(key, ck) if key else ck
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if rename:
        sd = {k.replace(rename, "proj."): v for k, v in sd.items() if rename in k}
    tgt_sd = tgt.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt_sd and v.shape == tgt_sd[k].shape}
    if filt:
        tgt.load_state_dict(filt, strict=False)
    return len(filt)

def infer_cfg_from_ckpt(path: pathlib.Path):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return None
    sd = _try_load(p, map_location="cpu")
    if sd is None: return None
    if isinstance(sd, dict) and "cfg" in sd and isinstance(sd["cfg"], dict):
        return dict(sd["cfg"])
    core = sd.get("core")
    if core is None: return None
    emb_w = core.get("emb.weight")
    if emb_w is None: return None
    d = emb_w.shape[1]
    layer_ids = []
    for k in core.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                layer_ids.append(int(parts[1]))
    layers = (max(layer_ids) + 1) if layer_ids else None
    U = core.get("blocks.0.mha.U")
    heads = rank = None
    if U is not None:
        dk, r = U.shape
        rank = r
        heads = d // dk if dk > 0 else None
    out = {"d": d}
    if layers is not None: out["layers"] = layers
    if heads is not None:  out["heads"] = heads
    if rank is not None:   out["rank"] = rank
    return out

# ───────────────────────── Train loop helpers ─────────────────────────
def _parse_grow_plan(s: str) -> List[int]:
    steps = []
    for part in s.split(","):
        part = part.strip()
        if part:
            v = int(part)
            if v >= 128:
                steps.append(v)
    return sorted(set(steps))

def _init_save_timers(resume_wall_time: float | None, interval_sec: int) -> Tuple[float, float]:
    now_wall = time.time()
    now_mono = time.monotonic()
    if resume_wall_time is None:
        return now_wall, now_mono
    elapsed_wall = max(0.0, now_wall - resume_wall_time)
    elapsed_clamped = min(float(interval_sec), elapsed_wall)
    return now_wall, now_mono - elapsed_clamped

def _count_enabled_params(*modules: Optional[nn.Module]) -> int:
    total = 0
    for m in modules:
        if m is not None:
            total += sum(p.numel() for p in m.parameters())
    return total

def _make_optimizer(core, ar_h, lr_core: float, lr_head: float):
    return torch.optim.AdamW([
        {"params": [p for p in core.parameters() if p.requires_grad], "lr": lr_core},
        {"params": ar_h.parameters(), "lr": lr_head},
    ])

def _phase_freeze(core: nn.Module, *, freeze_core: bool, unfreeze_ln: bool, train_emb: bool):
    for p in core.parameters():
        p.requires_grad = not freeze_core
    if freeze_core:
        if unfreeze_ln:
            for blk in core.blocks:
                for p in blk.ln1.parameters(): p.requires_grad = True
                for p in blk.ln2.parameters(): p.requires_grad = True
            for p in core.ln.parameters(): p.requires_grad = True
        if train_emb:
            for p in core.emb.parameters(): p.requires_grad = True

def _train_phase(
    args,
    *,
    core: nn.Module,
    ar_h: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    start_step: int,
    seen_tok: int,
    resume_wall_time: Optional[float],
    ce_tok,
    cfg: Dict[str,int],
    source: str,
    steps: Optional[int],
    block: int,
    save_dir: str,
    save_every_sec: int,
    save_every_steps: int,
    auto_grow: bool,
    grow_plan_s: str,
    grow_every_steps: int,
    chat: bool,
    chat_messages_key: str,
    dataset_field_text: str,
    sft_add_generation_prompt: bool,
    amp_flag: bool,
    fp8_only_flag: bool,
    fp8_fallback_flag: bool,
    target_tokens_override: Optional[int] = None,
    phase_name: str = "phase"
):
    BLOCK = block
    pbar = None

    if target_tokens_override is not None:
        target_tokens = target_tokens_override
    else:
        enabled_param_count = _count_enabled_params(core, ar_h)
        target_tokens = int(25 * enabled_param_count)

    new_tokens_needed = target_tokens - seen_tok
    if steps:
        new_tokens_needed = steps * BLOCK

    total_tokens_needed = seen_tok + max(0, new_tokens_needed)
    if new_tokens_needed <= 0:
        print(f"[{phase_name}] target already reached – skipping.")
        return start_step, seen_tok, resume_wall_time

    print(f"[{phase_name}] [auto-steps] {new_tokens_needed // BLOCK:,} steps (@ {BLOCK} tokens/step)")
    grow_plan = _parse_grow_plan(grow_plan_s) if auto_grow else []

    stream = token_stream(args, target_tokens, seed=42,
                          source=source, chat=chat, chat_messages_key=chat_messages_key,
                          sft_add_generation_prompt=sft_add_generation_prompt, dataset_field_text=dataset_field_text)
    buf: list[int] = []
    if pbar is None:
        pbar = tqdm(total=total_tokens_needed, initial=seen_tok, unit="tok")

    last_save_wall, last_save_mono = _init_save_timers(resume_wall_time, save_every_sec)
    step = start_step; steps_since_last_grow = 0

    while seen_tok < total_tokens_needed:
        try:
            while len(buf) < BLOCK:
                buf.append(next(stream))
        except StopIteration:
            break
        ids = torch.tensor(buf[:BLOCK], device=DEV).unsqueeze(0)
        buf = buf[BLOCK:]
        tgt_ar = ids.clone()

        try:
            with amp(amp_flag or fp8_only_flag, prefer_fp8=fp8_only_flag and (_supports_fp8() or fp8_fallback_flag)):
                h_ar = core(ids, causal_mask(ids.size(1)))
                logits_ar = ar_h(h_ar)[:, :-1]
                loss = ce_tok(logits_ar.reshape(-1, VOCAB), tgt_ar[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                new_block = max(128, BLOCK // 2)
                if new_block < BLOCK:
                    print(f"\n[{phase_name}][OOM] reducing block from {BLOCK} -> {new_block}")
                    BLOCK = new_block
                    if DEV.type == "cuda":
                        torch.cuda.empty_cache()
                    buf = ids[0].tolist() + buf
                    steps_since_last_grow = 0
                    continue
            raise

        step += 1; seen_tok += BLOCK
        pbar.update(BLOCK)
        pbar.set_postfix_str(f"{phase_name} loss={loss.item():.3f} block={BLOCK}")

        if save_every_sec > 0:
            now_mono = time.monotonic()
            if now_mono - last_save_mono >= save_every_sec:
                ck_name = f"{phase_name}_step{step:08d}.pt"
                save_ckpt(pathlib.Path(save_dir) / ck_name, core, ar_h, opt, scaler,
                          meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(),
                                "py_state": random.getstate(), "torch_state": rng_state(), "fp8_only": fp8_only_flag})
                last_save_mono = now_mono

        if save_every_steps > 0 and step > 0 and (step % save_every_steps == 0):
            ck_name = f"{phase_name}_step{step:08d}.pt"
            save_ckpt(pathlib.Path(save_dir) / ck_name, core, ar_h, opt, scaler,
                      meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(),
                            "py_state": random.getstate(), "torch_state": rng_state(), "fp8_only": fp8_only_flag})

        if auto_grow:
            steps_since_last_grow += 1
            if steps_since_last_grow >= grow_every_steps:
                steps_since_last_grow = 0
                try:
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        candidate = grow_plan[idx + 1]
                        print(f"[{phase_name}][auto-grow] {BLOCK} -> {candidate}")
                        BLOCK = candidate
                        if DEV.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        print(f"[{phase_name}][auto-grow] at max planned block.")
                except ValueError:
                    grow_plan = sorted(set(grow_plan + [BLOCK]))
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        candidate = grow_plan[idx + 1]
                        print(f"[{phase_name}][auto-grow] moving to planned BLOCK {candidate}")
                        BLOCK = candidate
                        if DEV.type == "cuda":
                            torch.cuda.empty_cache()

    if pbar is not None:
        pbar.close()

    save_ckpt(pathlib.Path(save_dir) / f"{phase_name}_final.pt", core, ar_h, opt, scaler,
              meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(),
                    "py_state": random.getstate(), "torch_state": rng_state(), "fp8_only": args.fp8_only})
    print(f"🎉 {phase_name} complete")
    return step, seen_tok, time.time()

# ───────────────────────── Top-level Train orchestrator ─────────────────────────
def train(args):
    cfg = PRESETS[args.preset].copy()

    # probe unless --fresh
    if not args.fresh:
        src_probe = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        prev_cfg = infer_cfg_from_ckpt(src_probe)
    else:
        prev_cfg = None

    if prev_cfg and not args.fresh:
        cfg["d"] = prev_cfg.get("d", cfg["d"])
        if prev_cfg.get("heads"): cfg["heads"] = prev_cfg["heads"]
        if args.rank is None and prev_cfg.get("rank"): cfg["rank"] = prev_cfg["rank"]
        if prev_cfg.get("layers"): cfg["layers"] = prev_cfg["layers"]
        if args.x2 and prev_cfg.get("layers"): cfg["layers"] = max(cfg["layers"], prev_cfg["layers"] * 2)
    if args.rank: cfg["rank"] = args.rank
    if args.x2 and not prev_cfg: cfg["layers"] *= 2

    BLOCK = args.block or DEFAULT_BLOCK

    core = Encoder(cfg).to(DEV)
    ar_h = ARHead(cfg["d"]).to(DEV)

    # shape-safe warm-start even in --fresh
    loaded = 0; src = None
    if args.warmstart_from:
        src = _resolve_ckpt(pathlib.Path(args.warmstart_from)) or pathlib.Path(args.warmstart_from)
    else:
        maybe = _resolve_ckpt(pathlib.Path(args.save_dir) / "final.pt")
        if maybe and not args.fresh:
            src = maybe
    if src:
        loaded += _safe_load_any(src, core, key="core")
        loaded += _safe_load_any(src, ar_h,   key="ar")
        if loaded:
            print(f"Warm-start: loaded {loaded} matching tensors from {src}")

    _phase_freeze(core, freeze_core=args.freeze_core, unfreeze_ln=args.unfreeze_ln, train_emb=args.train_emb)
    opt = _make_optimizer(core, ar_h, args.lr_core, args.lr_head)
    scaler = GradScaler(enabled=((args.amp or args.fp8_only) and DEV.type == "cuda"))
    ce_tok = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_step, seen_tok = 0, 0
    last_save_wall = None
    if args.resume and not args.fresh:
        start_step, seen_tok, last_save_wall = load_ckpt(pathlib.Path(args.resume), core, ar_h, opt, scaler)
        print(f"✓ resumed from step {start_step:,}, seen_tokens={seen_tok:,}")

    # Phase A: pretrain
    step, seen_tok, last_save_wall = _train_phase(
        args,
        core=core, ar_h=ar_h, opt=opt, scaler=scaler,
        start_step=start_step, seen_tok=seen_tok, resume_wall_time=last_save_wall,
        ce_tok=ce_tok, cfg=cfg,
        source=args.source, steps=args.steps, block=BLOCK,
        save_dir=args.save_dir, save_every_sec=args.save_every_sec, save_every_steps=args.save_every_steps,
        auto_grow=args.auto_grow, grow_plan_s=args.grow_plan, grow_every_steps=args.grow_every_steps,
        chat=args.chat, chat_messages_key=args.chat_messages_key, dataset_field_text=args.dataset_field_text,
        sft_add_generation_prompt=args.sft_add_generation_prompt,
        amp_flag=args.amp, fp8_only_flag=args.fp8_only, fp8_fallback_flag=args.fp8_fallback,
        target_tokens_override=(args.target_tokens if args.target_tokens else None),
        phase_name="pretrain"
    )

    # Auto-wire Phase B defaults if steps provided but no source
    if (not args.after_sft_source) and (args.after_sft_steps and args.after_sft_steps > 0):
        args.after_sft_source = DEFAULT_AFTER_SFT_SOURCES
        args.after_sft_chat = True
        if args.after_sft_add_generation_prompt is None:
            args.after_sft_add_generation_prompt = True
        if not args.after_sft_block or args.after_sft_block <= 0:
            args.after_sft_block = DEFAULT_AFTER_SFT_BLOCK

    if args.after_sft_source and args.after_sft_steps and args.after_sft_steps > 0:
        print("\n[after-sft] starting automatic post-pretraining chat SFT phase")
        _phase_freeze(core,
                      freeze_core=args.after_sft_freeze_core,
                      unfreeze_ln=args.after_sft_unfreeze_ln,
                      train_emb=args.after_sft_train_emb)
        opt = _make_optimizer(core, ar_h,
                              args.after_sft_lr_core or args.lr_core,
                              args.after_sft_lr_head or args.lr_head)

        step, seen_tok, last_save_wall = _train_phase(
            args,
            core=core, ar_h=ar_h, opt=opt, scaler=scaler,
            start_step=step, seen_tok=seen_tok, resume_wall_time=last_save_wall,
            ce_tok=ce_tok, cfg=cfg,
            source=args.after_sft_source, steps=args.after_sft_steps,
            block=args.after_sft_block or DEFAULT_AFTER_SFT_BLOCK,
            save_dir=args.save_dir, save_every_sec=args.save_every_sec, save_every_steps=args.save_every_steps,
            auto_grow=args.after_sft_auto_grow, grow_plan_s=(args.after_sft_grow_plan or args.grow_plan),
            grow_every_steps=(args.after_sft_grow_every_steps or args.grow_every_steps),
            chat=args.after_sft_chat, chat_messages_key=args.after_sft_chat_messages_key,
            dataset_field_text=args.after_sft_dataset_field_text,
            sft_add_generation_prompt=(args.after_sft_add_generation_prompt
                                       if args.after_sft_add_generation_prompt is not None
                                       else args.sft_add_generation_prompt),
            amp_flag=args.amp, fp8_only_flag=args.fp8_only, fp8_fallback_flag=args.fp8_fallback,
            target_tokens_override=None,
            phase_name="sft"
        )

    save_ckpt(pathlib.Path(args.save_dir) / "final.pt", core, ar_h, opt, scaler,
              meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(),
                    "py_state": random.getstate(), "torch_state": rng_state(), "fp8_only": args.fp8_only})
    print("🎉 training complete")

# ───────────────────────── Sampling utils ─────────────────────────
def _apply_no_repeat_ngram(logits: torch.Tensor, ids: torch.Tensor, n: int):
    if n <= 0 or ids.size(1) < n - 1:
        return logits
    prefix = ids[0, - (n - 1):].tolist()
    banned = []
    tokens = ids[0].tolist()
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n - 1] == prefix:
            banned.append(tokens[i + n - 1])
    if banned:
        banned_idx = torch.tensor(banned, device=logits.device, dtype=torch.long)
        logits[..., banned_idx] = float("-inf")
    return logits

def _apply_rep_presence_frequency(
    logits: torch.Tensor, ids: torch.Tensor, last_n: int,
    repetition_penalty: float, presence_penalty: float, frequency_penalty: float
):
    if ids.numel() == 0:
        return logits
    hist = ids[0, -last_n:].to(torch.long) if last_n > 0 else ids[0].to(torch.long)
    if hist.numel() == 0:
        return logits
    uniq, counts = torch.unique(hist, return_counts=True)
    if presence_penalty != 0.0 or frequency_penalty != 0.0:
        adjust = presence_penalty + frequency_penalty * counts.to(logits.dtype)
        logits[..., uniq] = logits[..., uniq] - adjust
    if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6:
        sel = logits[..., uniq]
        sel = torch.where(sel > 0, sel / repetition_penalty, sel * repetition_penalty)
        logits[..., uniq] = sel
    return logits

def _filter_top_k_top_p_min_p(
    logits: torch.Tensor, top_k: int, top_p: float, min_p: float, temperature: float
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-8)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    V = probs.size(-1)
    if top_k and top_k < V:
        vals, idx = torch.topk(probs, top_k, dim=-1)
        mask = torch.full_like(probs, 0.0)
        mask.scatter_((1 if probs.dim() == 2 else -1), idx, 1.0)
        probs = probs * mask
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        mask = torch.zeros_like(probs)
        mask.scatter_(1, sorted_idx, keep.to(mask.dtype))
        probs = probs * mask
    if min_p > 0.0:
        probs = torch.where(probs >= min_p, probs, torch.zeros_like(probs))
    sums = probs.sum(-1, keepdim=True)
    empty = (sums == 0)
    if empty.any():
        fallback_idx = logits.argmax(-1, keepdim=True)
        probs = torch.where(empty, torch.zeros_like(probs), probs)
        probs.scatter_(-1, fallback_idx, torch.where(empty, torch.ones_like(sums), torch.zeros_like(sums)))
    probs = probs / probs.sum(-1, keepdim=True)
    return probs

# ───────────────────────── Inference helpers ─────────────────────────
def load_joint(ckpt: str, preset: str):
    path = _resolve_ckpt(pathlib.Path(ckpt)) or pathlib.Path(ckpt)
    sd = _try_load(path, map_location="cpu")
    if sd is None:
        raise FileNotFoundError(f"No valid checkpoint at {path}")
    cfg = sd["cfg"] if "cfg" in sd and isinstance(sd["cfg"], dict) else (infer_cfg_from_ckpt(path) or PRESETS[preset])
    core = Encoder(cfg).to(DEV)
    ar_h = ARHead(cfg["d"]).to(DEV)
    core.load_state_dict(sd["core"])
    if "ar" in sd:
        ar_h.load_state_dict(sd["ar"])
    return core, ar_h

def _warn_tokenizer_mismatch(sd_tokenizer_id: str | None):
    if not sd_tokenizer_id:
        return
    if sd_tokenizer_id != TOKENIZER_ID:
        print(f"[warn] tokenizer mismatch: ckpt used '{sd_tokenizer_id}', runtime is '{TOKENIZER_ID}'. Expect degraded outputs.", file=sys.stderr)

DECODE_PRESETS = {
    "det": dict(greedy=True, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                repetition_penalty=1.05, presence_penalty=0.0, frequency_penalty=0.0,
                penalty_last_n=128, no_repeat_ngram_size=3),
    "balanced": dict(greedy=False, temperature=0.7, top_k=40, top_p=0.9, min_p=0.0,
                     repetition_penalty=1.1, presence_penalty=0.3, frequency_penalty=0.3,
                     penalty_last_n=256, no_repeat_ngram_size=3),
    "creative": dict(greedy=False, temperature=0.85, top_k=80, top_p=0.95, min_p=0.0,
                     repetition_penalty=1.05, presence_penalty=0.2, frequency_penalty=0.2,
                     penalty_last_n=256, no_repeat_ngram_size=3),
}

@torch.no_grad()
def ar_decode(core, ar_h, prompt: str, max_new: int, T: float,
              greedy: bool, top_k: int, top_p: float, min_p: float,
              repetition_penalty: float, presence_penalty: float,
              frequency_penalty: float, penalty_last_n: int,
              no_repeat_ngram_size: int,
              use_fp8: bool, fp8_fallback: bool):
    prompt_ids = tok.encode(prompt)
    if len(prompt_ids) == 0:
        ids = torch.tensor([[EOS] if EOS is not None else [0]], device=DEV); prompt_len = 0
    else:
        ids = torch.tensor([prompt_ids], device=DEV); prompt_len = ids.size(1)

    t0 = time.time()
    with amp(use_fp8 or False, prefer_fp8=use_fp8 and (_supports_fp8() or fp8_fallback)):
        h_full, kvs = core(ids, causal_mask(ids.size(1)), use_cache=True)
        for _ in range(max_new):
            logits = ar_h(h_full)[:, -1]
            logits = _apply_no_repeat_ngram(logits, ids, no_repeat_ngram_size)
            logits = _apply_rep_presence_frequency(logits, ids, penalty_last_n,
                                                   repetition_penalty, presence_penalty, frequency_penalty)
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = _filter_top_k_top_p_min_p(logits.squeeze(0), top_k, top_p, min_p, T)
                nxt = probs.multinomial(1)
            ids = torch.cat([ids, nxt.unsqueeze(0) if nxt.dim()==1 else nxt], 1)
            x = ids[:, -1:]; h_full, kvs = core(x, None, kv_caches=kvs, use_cache=True)

    full_ids = ids[0].tolist()
    prompt_text = tok.decode(full_ids[:prompt_len], skip_special_tokens=True)
    gen_text    = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)

    if sys.stdout.isatty():
        sys.stdout.write("\x1b[90m"); sys.stdout.write(prompt_text); sys.stdout.write("\x1b[0m"); sys.stdout.write(gen_text + "\n")
    else:
        sys.stdout.write(prompt_text + gen_text + "\n")

    print(f"[{len(full_ids) - prompt_len} tok in {time.time() - t0:.2f}s]")

# ───────────────────────── CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--preset", choices=PRESETS, default="base17")
    tr.add_argument("--rank", type=int)
    tr.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    tr.add_argument("--source", default=DEFAULT_PRETRAIN_SOURCES,
                    help="Comma-separated datasets (optionally dataset:config), or json:/path.jsonl")
    tr.add_argument("--target_tokens", type=int)
    tr.add_argument("--steps", type=int)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--save_every_sec", type=int, default=DEFAULT_SAVE_SEC)
    tr.add_argument("--save_every_steps", type=int, default=0)
    tr.add_argument("--save_dir", default=str(CKDIR))
    tr.add_argument("--resume", type=str)
    tr.add_argument("--x2", action="store_true")
    tr.add_argument("--warmstart_from", type=str, default=None)
    tr.add_argument("--fresh", action="store_true")

    # FP8 control
    tr.add_argument("--fp8-only", action="store_true", dest="fp8_only")
    tr.add_argument("--fp8-fallback", action="store_true", dest="fp8_fallback")

    # Progressive block growth
    tr.add_argument("--auto_grow", action="store_true")
    tr.add_argument("--grow_plan", type=str, default="576,768,1024")
    tr.add_argument("--grow_every_steps", type=int, default=50000)

    # Chat / dataset fields
    tr.add_argument("--chat", action="store_true")
    tr.add_argument("--chat_messages_key", type=str, default="messages")
    tr.add_argument("--dataset_field_text", type=str, default="text")
    tr.add_argument("--sft_add_generation_prompt", action="store_true")

    # Phase A freezing / LRs
    tr.add_argument("--freeze_core", action="store_true")
    tr.add_argument("--unfreeze_ln", action="store_true")
    tr.add_argument("--train_emb", action="store_true")
    tr.add_argument("--lr_core", type=float, default=LR_CORE)
    tr.add_argument("--lr_head", type=float, default=LR_HEAD)

    # Phase B: automatic SFT
    tr.add_argument("--after_sft_source", type=str, default="")
    tr.add_argument("--after_sft_steps", type=int, default=0)
    tr.add_argument("--after_sft_chat", action="store_true")
    tr.add_argument("--after_sft_chat_messages_key", type=str, default="messages")
    tr.add_argument("--after_sft_dataset_field_text", type=str, default="text")
    tr.add_argument("--after_sft_add_generation_prompt", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None)
    tr.add_argument("--after_sft_block", type=int, default=0)
    tr.add_argument("--after_sft_auto_grow", action="store_true")
    tr.add_argument("--after_sft_grow_plan", type=str, default="")
    tr.add_argument("--after_sft_grow_every_steps", type=int, default=0)
    tr.add_argument("--after_sft_freeze_core", action="store_true")
    tr.add_argument("--after_sft_unfreeze_ln", action="store_true")
    tr.add_argument("--after_sft_train_emb", action="store_true")
    tr.add_argument("--after_sft_lr_core", type=float, default=0.0)
    tr.add_argument("--after_sft_lr_head", type=float, default=0.0)

    inf = sub.add_parser("infer")
    inf.add_argument("--mode", choices=["ar"], required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--preset", default="base17")
    inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=256)
    inf.add_argument("--seed", type=int, default=1234)
    inf.add_argument("--greedy", action="store_true")
    inf.add_argument("--temperature", type=float, default=0.7)
    inf.add_argument("--top_k", type=int, default=40)
    inf.add_argument("--top_p", type=float, default=0.9)
    inf.add_argument("--min_p", type=float, default=0.0)
    inf.add_argument("--repetition_penalty", type=float, default=1.1)
    inf.add_argument("--presence_penalty", type=float, default=0.3)
    inf.add_argument("--frequency_penalty", type=float, default=0.3)
    inf.add_argument("--penalty_last_n", type=int, default=256)
    inf.add_argument("--no_repeat_ngram_size", type=int, default=3)
    inf.add_argument("--fp8-only", action="store_true", dest="fp8_only")
    inf.add_argument("--fp8-fallback", action="store_true", default=False, dest="fp8_fallback")
    inf.add_argument("--decode_preset", choices=["det","balanced","creative"], default="balanced")

    args = ap.parse_args()
    if args.cmd == "train":
        if args.fp8_only:
            print("[init] FP8-only requested. If FP8 kernels are missing, use --fp8-fallback to continue with bf16.")
        train(args)
    else:
        core, ar_h = load_joint(args.ckpt, args.preset)
        try:
            p = _resolve_ckpt(pathlib.Path(args.ckpt)) or pathlib.Path(args.ckpt)
            _sd = _try_load(p, map_location="cpu")
            _warn_tokenizer_mismatch(_sd.get("tokenizer_id") if isinstance(_sd, dict) else None)
        except Exception:
            pass
        set_seed(args.seed)
        dp = DECODE_PRESETS.get(args.decode_preset, {})
        g  = dp.get("greedy", args.greedy)
        T  = dp.get("temperature", args.temperature)
        k  = dp.get("top_k", args.top_k)
        p_ = dp.get("top_p", args.top_p)
        mp = dp.get("min_p", args.min_p)
        rp = dp.get("repetition_penalty", args.repetition_penalty)
        pp = dp.get("presence_penalty", args.presence_penalty)
        fp = dp.get("frequency_penalty", args.frequency_penalty)
        ln = dp.get("penalty_last_n", args.penalty_last_n)
        ng = dp.get("no_repeat_ngram_size", args.no_repeat_ngram_size)

        ar_decode(core, ar_h, args.prompt, args.max_new, T,
                  g, k, p_, mp, rp, pp, fp, ln, ng,
                  use_fp8=args.fp8_only, fp8_fallback=args.fp8_fallback if hasattr(args, "fp8_fallback") else False)

if __name__ == "__main__":
    main()
