"""
SAKURA multi-hop audio reasoning — Stage 0 baseline.

Confirms the open gap: a strong 2026 base (Kimi-Audio) should score HIGH on single-hop
(perceive the audio attribute) but LOW on multi-hop (chain it) — even though it perceives
correctly. MCQ-style answers -> deterministic scoring, NO LLM judge (no leakage).
"""
import modal, os, re

app = modal.App("vanitas-sakura")
vol = modal.Volume.from_name("vanitas-sakura", create_if_missing=True)

# Kimi-Audio image: clone the official repo + install (custom inference library).
kimi_img = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .run_commands(
        "git clone https://github.com/MoonshotAI/Kimi-Audio.git /root/Kimi-Audio",
        "cd /root/Kimi-Audio && git submodule update --init --recursive || true",
        # build prereqs FIRST (flash-attn's setup.py imports packaging at build time)
        "pip install packaging wheel setuptools ninja",
        # install Kimi's requirements WITHOUT flash_attn (prebuilt wheel below). This pins torch==2.6.0.
        "cd /root/Kimi-Audio && grep -vi 'flash' requirements.txt > req_noflash.txt && "
        "pip install -r req_noflash.txt",
        # prebuilt flash-attn matching torch 2.6 / cu12 / cp310 (no source compile; build box has no nvcc)
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        # the kimia_infer package itself (--no-deps: don't re-trigger flash-attn build)
        "cd /root/Kimi-Audio && pip install -e . --no-deps",
        # transformers version is BRACKETED: GLM4 speech tokenizer needs EncoderDecoderCache
        # (added in 4.43.0) -> >=4.43; Kimi's attention uses the pre-refactor rotary that 4.44
        # broke (28-vs-156 mismatch) -> <4.44. So the working window is 4.43.x.
        "pip install 'transformers==4.43.4' 'huggingface_hub==0.24.6' 'datasets==2.21.0'",
        # Kimi's whisper-large-v3 safetensors has NO metadata header -> metadata is None, which
        # 4.44.2 rejects (it requires format in [pt,tf,flax,mlx]). The file IS a pytorch
        # safetensors, just missing the tag -> default missing metadata to {"format":"pt"} in the
        # if-CHECK only (double-quoted). The single-quoted copy is inside an f-string error msg
        # that never executes once the check passes -- patching it would break f-string syntax.
        "sed -i 's/metadata\\.get(\"format\")/(metadata or {\"format\":\"pt\"}).get(\"format\")/g' "
        "/usr/local/lib/python3.10/site-packages/transformers/modeling_utils.py",
        # image-build mtime normalization can leave a STALE .pyc (old bytecode runs despite
        # the patched source) -> delete compiled bytecode so it recompiles from the patch.
        "find /usr/local/lib/python3.10/site-packages/transformers -name '*.pyc' -delete",
    )
    # HF cache on the volume so the ~20GB Kimi weights persist across runs (no re-download).
    .env({"PYTHONPATH": "/root/Kimi-Audio", "HF_HUB_ENABLE_HF_TRANSFER": "0",
          "HF_HOME": "/data/hf"})
)

_STOP = {"the","a","an","of","to","in","is","and","or","for","by","with","be","are","it",
         "its","that","this","as","on","at","from","which","what"}
def _norm(s): return re.sub(r"[^a-z0-9 ]"," ",(s or "").lower())
def _match(pred, ans):
    """Lenient legacy matcher (kept for the Kimi fn): content-word recall >= 0.6."""
    p = _norm(pred); cw = [w for w in _norm(ans).split() if len(w)>2 and w not in _STOP]
    if not cw: return _norm(ans) in p
    return sum(1 for w in cw if w in p)/len(cw) >= 0.6


# ---- strict deterministic MCQ scorer (no LLM judge -> no leakage) -------------------
def _parse_options(instruction):
    """Pull '(a) text' options from the SAKURA instruction -> {letter: text}."""
    s = (instruction or "").replace("\n", " ")
    opts = {}
    for m in re.finditer(r"\(([a-dA-D])\)\s*(.+?)(?=\s*\([a-dA-D]\)|$)", s):
        opts[m.group(1).lower()] = m.group(2).strip().rstrip(".").strip()
    return opts

def _gold_letter_text(gold):
    m = re.match(r"\s*\(([a-dA-D])\)\s*(.*)", gold or "")
    if m: return m.group(1).lower(), m.group(2).strip()
    return None, (gold or "").strip()

def _word_in(text, phrase):
    """Case-insensitive, word-boundary containment (so 'male' is NOT found in 'female')."""
    if not phrase: return False
    return re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text.lower()) is not None

def mcq_correct(pred, gold, instruction):
    """Did the model SELECT the gold option? Strict: chosen letter, else exact option text."""
    pred_l = (pred or "").lower()
    gl, gt = _gold_letter_text(gold)
    opts = _parse_options(instruction)
    # 1) explicit letter selection in the prediction, e.g. '(b)' or 'answer is b'
    letters = set(re.findall(r"\(([a-d])\)", pred_l))
    if not letters:
        for m in re.finditer(r"\b(?:option|answer|choice)\b[:\s]*\(?([a-d])\)?\b", pred_l):
            letters.add(m.group(1))
    if letters:
        return gl in letters and len(letters) == 1   # ambiguous multi-pick = wrong
    # 2) option-text matching (word-boundary), correct only if ONLY the gold option appears
    if opts and gl in opts:
        matched = {L for L, t in opts.items() if _word_in(pred_l, t)}
        if matched:
            return matched == {gl}
    # 3) fallback: gold option text appears
    return _word_in(pred_l, gt)


def pick_option(pred, instruction):
    """Extract the SELECTED option letter from a prediction (for majority vote). None if unclear."""
    import re as _re
    pred_l = (pred or "").lower()
    letters = _re.findall(r"\(([a-d])\)", pred_l)
    if not letters:
        for m in _re.finditer(r"\b(?:option|answer|choice)\b[:\s]*\(?([a-d])\)?\b", pred_l):
            letters.append(m.group(1))
    if letters:
        from collections import Counter
        c = Counter(letters).most_common()
        if len(c) == 1 or (len(c) > 1 and c[0][1] > c[1][1]):
            return c[0][0]            # dominant letter
        return None                   # tie -> unclear
    opts = _parse_options(instruction)
    matched = [L for L, t in opts.items() if _word_in(pred_l, t)]
    return matched[0] if len(matched) == 1 else None


@app.function(image=kimi_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def sakura_baseline(track: str = "GenderQA", n: int = 40):
    """Baseline Kimi-Audio on a SAKURA track: single-hop vs multi-hop accuracy."""
    import soundfile as sf, numpy as np, tempfile, json
    from datasets import load_dataset
    from kimia_infer.api.kimia import KimiAudio

    print(f"[sakura] loading Kimi-Audio...", flush=True)
    model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=False)
    vol.commit()  # persist the ~20GB HF download to the volume so reruns skip it
    sampling = {"audio_temperature": 0.0, "audio_top_k": 5, "text_temperature": 0.0, "text_top_k": 5}
    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    ds = ds.select(range(min(n, len(ds))))

    def ask(audio_arr, sr, instruction):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_arr, sr); path = f.name
        messages = [{"role": "user", "message_type": "text", "content": instruction},
                    {"role": "user", "message_type": "audio", "content": path}]
        _, text = model.generate(messages, **sampling, output_type="text")
        os.unlink(path)
        return text

    s_c = 0; m_c = 0; nn = 0
    for ex in ds:
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        sp = ask(a, sr, ex["single_instruction"]); mp = ask(a, sr, ex["multi_instruction"])
        s_c += int(_match(sp, ex["single_answer"])); m_c += int(_match(mp, ex["multi_answer"])); nn += 1
        if nn <= 3:
            print(f"\n[ex {nn}] SINGLE gold={ex['single_answer']!r} pred={sp[:80]!r} ok={_match(sp,ex['single_answer'])}", flush=True)
            print(f"        MULTI  gold={ex['multi_answer']!r} pred={mp[:80]!r} ok={_match(mp,ex['multi_answer'])}", flush=True)
        if nn % 10 == 0: print(f"[sakura] {nn}/{len(ds)}  single={s_c/nn:.2f} multi={m_c/nn:.2f}", flush=True)
    print(f"\n[sakura] {track}  N={nn}  SINGLE-hop={s_c/nn:.3f}  MULTI-hop={m_c/nn:.3f}  GAP={(s_c-m_c)/nn:.3f}", flush=True)
    print(f"[sakura] {'GAP CONFIRMED (multi << single) -> open problem real for 2026 base' if (s_c-m_c)/nn>0.15 else 'small gap -> reasoning more solved than expected'}", flush=True)
    return {"track": track, "single": s_c/nn, "multi": m_c/nn, "n": nn}


# ----------------------------------------------------------------------------------
# Stage 0b / Stage 1 base: Qwen2-Audio-7B-Instruct on CLEAN plain-transformers inference.
# (Kimi-Audio abandoned: irreconcilable transformers version conflict in its custom stack.)
# SAKURA published gap (Table 2): Qwen2-Audio Gender 88.0->47.2, Lang 83.8->48.0, Animal
# 88.8->61.4 -> 24-41pt drops. We reproduce on our own pipeline = the base we build Stage 1 on.
# ----------------------------------------------------------------------------------
qwen_img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0", "torchaudio==2.4.0",
        # datasets>=3 decodes the Audio feature via torchcodec; pin to the soundfile-based 2.21.0
        "transformers==4.48.0", "accelerate", "datasets==2.21.0", "soundfile", "librosa",
        "huggingface_hub", "hf_transfer",
    )
    .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


_TRACKS = ["GenderQA", "LanguageQA", "EmotionQA", "AnimalQA"]


def _run_sakura(ask_fn, resample_to, tracks, n, tag):
    """Shared SAKURA eval loop with STRICT MCQ scoring. ask_fn(audio16k, instruction)->str."""
    import numpy as np, librosa
    from datasets import load_dataset
    results = {}
    for track in tracks:
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        ds = ds.select(range(min(n, len(ds))))
        s_c = m_c = nn = 0
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != resample_to: a = librosa.resample(a, orig_sr=sr, target_sr=resample_to)
            sp = ask_fn(a, ex["single_instruction"]); mp = ask_fn(a, ex["multi_instruction"])
            s_ok = mcq_correct(sp, ex["single_answer"], ex["single_instruction"])
            m_ok = mcq_correct(mp, ex["multi_answer"], ex["multi_instruction"])
            s_c += int(s_ok); m_c += int(m_ok); nn += 1
            if nn <= 2:
                print(f"\n[{tag}:{track} ex{nn}] S gold={ex['single_answer']!r} pred={sp[:70]!r} ok={s_ok}", flush=True)
                print(f"        M gold={ex['multi_answer']!r} pred={mp[:70]!r} ok={m_ok}", flush=True)
        single, multi = s_c/nn, m_c/nn
        results[track] = {"single": single, "multi": multi, "gap": single-multi, "n": nn}
        print(f"[{tag}] {track}  N={nn}  SINGLE={single:.3f}  MULTI={multi:.3f}  GAP={single-multi:.3f}", flush=True)
    avg_s = sum(r["single"] for r in results.values())/len(results)
    avg_m = sum(r["multi"] for r in results.values())/len(results)
    print(f"\n[{tag}] ===== AVG over {len(results)} tracks  SINGLE={avg_s:.3f}  MULTI={avg_m:.3f}  GAP={avg_s-avg_m:.3f} =====", flush=True)
    return {"per_track": results, "avg_single": avg_s, "avg_multi": avg_m, "avg_gap": avg_s-avg_m}


@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_baseline(tracks: str = ",".join(_TRACKS), n: int = 50):
    """Qwen2-Audio on SAKURA, all tracks, STRICT MCQ scoring (one model load)."""
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[qwen] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate  # 16000

    def ask(audio16k, instruction):
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "x.wav"},
            {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    return _run_sakura(ask, SR, tracks.split(","), n, "qwen")


# ----------------------------------------------------------------------------------
# Stretch / 2nd base: Audio Flamingo 3 (NVIDIA) — native transformers class, strongest
# perception (speech+sound+music) + built-in CoT. Tests: does the multi-hop gap survive SOTA?
# ----------------------------------------------------------------------------------
af3_img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0", "torchaudio==2.6.0",
        # AudioFlamingo3ForConditionalGeneration landed in transformers 5.x (2025-11)
        "transformers==5.0.0", "accelerate", "datasets==2.21.0", "soundfile", "librosa",
        "huggingface_hub", "hf_transfer",
    )
    .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=af3_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def af3_baseline(tracks: str = ",".join(_TRACKS), n: int = 50):
    """Audio Flamingo 3 on SAKURA, all tracks, STRICT MCQ scoring (one model load)."""
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    MODEL = "nvidia/audio-flamingo-3-hf"
    print(f"[af3] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    # load fp32 first to dodge the known dtype bug (hf#42259), keep on GPU
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate

    import soundfile as sf, tempfile, os as _os
    def ask(audio16k, instruction):
        # AF3's documented API loads audio from a path -> write a temp wav at its SR
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio16k, SR); path = f.name
        conv = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "path": path}]}]
        inputs = processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True, return_dict=True)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        _os.unlink(path)
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    return _run_sakura(ask, SR, tracks.split(","), n, "af3")


# ----------------------------------------------------------------------------------
# Clean non-Qwen 2026 frontier: Audio Flamingo Next (NVIDIA, arXiv 2604.10905, Apr 2026).
# Native AudioFlamingoNextForConditionalGeneration class. Tests: does the gap survive the
# 2026 frontier? Instruct variant = direct answers (apples-to-apples vs Qwen2-A/AF3).
# ----------------------------------------------------------------------------------
afnext_img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0", "torchaudio==2.6.0",
        # AF-Next is Apr-2026; its native class may be newer than any pinned release -> git main
        "git+https://github.com/huggingface/transformers", "accelerate",
        "datasets==2.21.0", "soundfile", "librosa", "huggingface_hub", "hf_transfer",
    )
    .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=afnext_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def afnext_baseline(tracks: str = ",".join(_TRACKS), n: int = 50, think: bool = False):
    """Audio Flamingo Next on SAKURA, all tracks, STRICT MCQ. think=True -> -think-hf variant."""
    import torch, soundfile as sf, tempfile, os as _os
    import transformers
    from transformers import AutoProcessor, AutoConfig

    MODEL = "nvidia/audio-flamingo-next-think-hf" if think else "nvidia/audio-flamingo-next-hf"
    print(f"[afnext] loading {MODEL} (think={think})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    # read the exact generation class from the config (AutoModel loads the base, no .generate)
    arch = AutoConfig.from_pretrained(MODEL).architectures[0]
    print(f"[afnext] architecture = {arch}", flush=True)
    AFNext = getattr(transformers, arch)
    # fp32 to dodge the AF3/AF-Next audio-conv dtype bug (hf#42259): float input vs bf16 bias
    model = AFNext.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate
    max_tok = 512 if think else 256

    def ask(audio16k, instruction):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio16k, SR); path = f.name
        conv = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "path": path}]}]
        inputs = processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True, return_dict=True)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        _os.unlink(path)
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    return _run_sakura(ask, SR, tracks.split(","), n, "afnext")


# ==================================================================================
# STAGE 1 — the reasoning layer (over the FROZEN Qwen2-Audio base, no training).
# Thesis: the model PERCEIVES the attribute (high single-hop) but fails to INTEGRATE it
# into the chained question (low multi-hop). Ablation:
#   A (base)  : single-turn multi_instruction with audio          (= our baseline)
#   B (chain) : 2-turn extract->chain — turn1 extracts the attribute (single_instruction),
#               turn2 answers multi_instruction WITH the model's own extracted answer in
#               context. Forces the audio-perceived fact explicitly into the reasoning chain.
# Per-track multi-hop accuracy A vs B (+ lift) on the SAME items = the Stage-1 result.
# ==================================================================================
@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_stage1(tracks: str = ",".join(_TRACKS), n: int = 50):
    """Qwen2-Audio Stage 1: base multi-hop vs extract->chain multi-hop (strict MCQ)."""
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[stage1] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate

    def _gen(conv, audio16k):
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def base_multi(audio16k, single_instr, multi_instr):
        # A: single-turn, audio + multi question (the baseline path)
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "x.wav"}, {"type": "text", "text": multi_instr}]}]
        return _gen(conv, audio16k)

    def chain_multi(audio16k, single_instr, multi_instr):
        # B: turn1 extract the attribute, turn2 chain with the extracted answer in context
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "x.wav"}, {"type": "text", "text": single_instr}]}]
        a1 = _gen(conv, audio16k)
        conv.append({"role": "assistant", "content": [{"type": "text", "text": a1}]})
        conv.append({"role": "user", "content": [{"type": "text", "text": multi_instr}]})
        a2 = _gen(conv, audio16k)
        return a1, a2

    results = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        ds = ds.select(range(min(n, len(ds))))
        a_c = b_c = nn = 0
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, mi = ex["single_instruction"], ex["multi_instruction"]
            base = base_multi(a, si, mi)
            ext, chain = chain_multi(a, si, mi)
            a_ok = mcq_correct(base, ex["multi_answer"], mi)
            b_ok = mcq_correct(chain, ex["multi_answer"], mi)
            a_c += int(a_ok); b_c += int(b_ok); nn += 1
            if nn <= 2:
                print(f"\n[stage1:{track} ex{nn}] gold={ex['multi_answer']!r}", flush=True)
                print(f"   A base  pred={base[:70]!r} ok={a_ok}", flush=True)
                print(f"   extract={ext[:50]!r}", flush=True)
                print(f"   B chain pred={chain[:70]!r} ok={b_ok}", flush=True)
        A, B = a_c/nn, b_c/nn
        results[track] = {"base_multi": A, "chain_multi": B, "lift": B-A, "n": nn}
        print(f"[stage1] {track}  N={nn}  A_base={A:.3f}  B_chain={B:.3f}  LIFT={B-A:+.3f}", flush=True)
    avg_a = sum(r["base_multi"] for r in results.values())/len(results)
    avg_b = sum(r["chain_multi"] for r in results.values())/len(results)
    print(f"\n[stage1] ===== AVG  A_base={avg_a:.3f}  B_chain={avg_b:.3f}  LIFT={avg_b-avg_a:+.3f} =====", flush=True)
    return {"per_track": results, "avg_base": avg_a, "avg_chain": avg_b, "avg_lift": avg_b-avg_a}


@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_stage1_sc(tracks: str = ",".join(_TRACKS), n: int = 50, kchains: int = 5):
    """Stage 1 ablation C: A base / B extract->chain / C self-consistency (K batched chains).
    Targets the CHAINING residual (e.g. Gender): same extracted attribute, vote over K chains.
    Reports multi-hop acc AND sequences/item (the efficiency/Pareto axis)."""
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[sc] loading {MODEL} (K={kchains})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate

    def _gen(conv, audio16k, k=1):
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            if k == 1:
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            else:  # K sampled chains in ONE batched call (~1x latency self-consistency)
                out = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, top_p=0.9, num_return_sequences=k)
        gen = out[:, inputs["input_ids"].shape[1]:]
        dec = processor.batch_decode(gen, skip_special_tokens=True)
        return [d.strip() for d in dec]

    def run_chain(audio16k, single_instr, multi_instr, k=1):
        # turn1 extract (greedy), turn2 chain (greedy if k==1 else K sampled)
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "x.wav"}, {"type": "text", "text": single_instr}]}]
        ext = _gen(conv, audio16k, 1)[0]
        conv.append({"role": "assistant", "content": [{"type": "text", "text": ext}]})
        conv.append({"role": "user", "content": [{"type": "text", "text": multi_instr}]})
        return ext, _gen(conv, audio16k, k)

    results = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test").select(range(min(n, 9999)))
        ds = ds.select(range(min(n, len(ds))))
        a_c = b_c = c_c = nn = 0
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            # A base
            convA = [{"role": "user", "content": [
                {"type": "audio", "audio_url": "x.wav"}, {"type": "text", "text": mi}]}]
            base = _gen(convA, a, 1)[0]
            # B chain (greedy)
            ext, chains1 = run_chain(a, si, mi, 1)
            # C self-consistency: K sampled chains, majority vote the picked option
            _, chainsK = run_chain(a, si, mi, kchains)
            picks = [pick_option(c, mi) for c in chainsK]
            picks = [p for p in picks if p]
            vote = Counter(picks).most_common(1)[0][0] if picks else None
            c_ok = bool(vote) and mcq_correct(f"({vote})", gold, mi)
            a_ok = mcq_correct(base, gold, mi)
            b_ok = mcq_correct(chains1[0], gold, mi)
            a_c += int(a_ok); b_c += int(b_ok); c_c += int(c_ok); nn += 1
            if nn <= 2:
                print(f"\n[sc:{track} ex{nn}] gold={gold!r} ext={ext[:40]!r}", flush=True)
                print(f"   A={a_ok} B={b_ok} | C votes={Counter(picks)} -> {vote} ok={c_ok}", flush=True)
        A, B, C = a_c/nn, b_c/nn, c_c/nn
        results[track] = {"A_base": A, "B_chain": B, "C_selfcons": C, "n": nn}
        print(f"[sc] {track}  N={nn}  A={A:.3f}  B={B:.3f}  C={C:.3f}  (C-B={C-B:+.3f}, C-A={C-A:+.3f})", flush=True)
    avgA = sum(r["A_base"] for r in results.values())/len(results)
    avgB = sum(r["B_chain"] for r in results.values())/len(results)
    avgC = sum(r["C_selfcons"] for r in results.values())/len(results)
    # compute/item (generated sequences): A=1, B=2, C=1(extract)+1(B chain)+K(sampled)=2+K
    print(f"\n[sc] ===== AVG  A={avgA:.3f}(1seq)  B={avgB:.3f}(2seq)  C={avgC:.3f}({2+kchains}seq) =====", flush=True)
    print(f"[sc] lift C over A = {avgC-avgA:+.3f}, C over B = {avgC-avgB:+.3f}", flush=True)
    return {"per_track": results, "avgA": avgA, "avgB": avgB, "avgC": avgC, "kchains": kchains}


# ==================================================================================
# PAMR on AUDIO FLAMINGO NEXT (2026 frontier = the HEADLINE). Full closed loop:
# perceive->anchor -> anchored self-consistency chain -> verify (vote agreement) ->
# RE-ANCHOR on low agreement -> commit. vs base. Held-out slice [start, start+n).
# ==================================================================================
@app.function(image=afnext_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def afnext_pamr(tracks: str = ",".join(_TRACKS), start: int = 50, n: int = 50, kchains: int = 5):
    """PAMR on AF-Next vs base, strict MCQ, on a held-out example range."""
    import numpy as np, torch, librosa, soundfile as sf, tempfile, os as _os
    from collections import Counter
    import transformers
    from transformers import AutoProcessor, AutoConfig
    from datasets import load_dataset

    MODEL = "nvidia/audio-flamingo-next-hf"
    print(f"[pamr] loading {MODEL} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    arch = AutoConfig.from_pretrained(MODEL).architectures[0]
    model = getattr(transformers, arch).from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate

    def _gen(conv, k=1):
        inputs = processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True, return_dict=True)
        inputs = {kk: (v.to(model.device) if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            if k == 1:
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, top_p=0.9, num_return_sequences=k)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [d.strip() for d in processor.batch_decode(gen, skip_special_tokens=True)]

    def U(text, path=None):
        c = [{"type": "text", "text": text}]
        if path: c.append({"type": "audio", "path": path})
        return {"role": "user", "content": c}
    def Asst(text): return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    def vote_of(chains, mi):
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        cnt = Counter(picks)
        if not cnt: return None, 0.0
        top, k = cnt.most_common(1)[0]
        return top, k / max(1, len(chains))

    results = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds))
        ds = ds.select(range(lo, hi))
        base_c = pamr_c = re_n = nn = 0
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, a, SR); path = f.name
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            base = _gen([U(mi, path)], 1)[0]
            anchor = _gen([U(si, path)], 1)[0]
            chains = _gen([U(si, path), Asst(anchor), U(mi)], kchains)
            vote, agree = vote_of(chains, mi)
            if agree < 0.6:  # verify failed -> RE-ANCHOR (listen again) and re-chain
                re_n += 1
                anchor2 = _gen([U("Listen to the audio again carefully and re-identify the key attribute. " + si, path)], 1)[0]
                chains2 = _gen([U(si, path), Asst(anchor2), U(mi)], kchains)
                v2, _ = vote_of(chains2, mi)
                vote = v2 or vote
            _os.unlink(path)
            base_ok = mcq_correct(base, gold, mi)
            pamr_ok = bool(vote) and mcq_correct(f"({vote})", gold, mi)
            base_c += int(base_ok); pamr_c += int(pamr_ok); nn += 1
            if nn <= 2:
                print(f"\n[pamr:{track} ex{nn}] gold={gold!r} base={base_ok} anchor={anchor[:30]!r} vote={vote} reanchored={agree<0.6} PAMR={pamr_ok}", flush=True)
        B_, P_ = base_c / nn, pamr_c / nn
        results[track] = {"base": B_, "pamr": P_, "lift": P_ - B_, "reanchor_rate": re_n / nn, "n": nn}
        print(f"[pamr] {track}  [{lo}:{hi}]  base={B_:.3f}  PAMR={P_:.3f}  LIFT={P_-B_:+.3f}  reanchor={re_n/nn:.2f}", flush=True)
    avgB = sum(r["base"] for r in results.values()) / len(results)
    avgP = sum(r["pamr"] for r in results.values()) / len(results)
    print(f"\n[pamr] ===== AF-Next [{start}:{start+n}]  AVG base={avgB:.3f}  PAMR={avgP:.3f}  LIFT={avgP-avgB:+.3f} =====", flush=True)
    return {"per_track": results, "avg_base": avgB, "avg_pamr": avgP, "start": start, "n": n}


# ==================================================================================
# MATA + ANCHOR (the COMBINE): white-box attention steering toward audio (MATA) composed
# with the prompt-level anchor. MATA = upweight last-token->audio attention by (1+alpha)
# in layers [LO,HI] pre-softmax (arXiv 2509.18816). Targets the ROOT cause (text-driven
# reasoning under-attends audio). Ablation in one run: base / MATA / anchor / MATA+anchor.
# ==================================================================================
_MATA = {"on": False, "span": None, "alpha": 0.1, "lo": 10, "hi": 20}

@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_mata(tracks: str = "GenderQA,AnimalQA", start: int = 0, n: int = 20, alpha: float = 0.1):
    """MATA attention-steering + anchor on Qwen2-Audio (SAKURA multi-hop). 4-way ablation."""
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    import transformers.models.qwen2.modeling_qwen2 as q2

    _MATA["alpha"] = alpha
    # ---- patch Qwen2 eager attention to inject MATA scaling on the LAST query row ----
    def mata_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = q2.repeat_kv(key, module.num_key_value_groups)
        vs = q2.repeat_kv(value, module.num_key_value_groups)
        aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None:
            aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
        li = getattr(module, "layer_idx", None)
        if _MATA["on"] and _MATA["span"] is not None and li is not None and _MATA["lo"] <= li <= _MATA["hi"]:
            a_s, a_e = _MATA["span"]
            if a_e >= a_s and a_e < aw.shape[-1]:
                aw[:, :, -1, a_s:a_e + 1] = aw[:, :, -1, a_s:a_e + 1] * (1.0 + _MATA["alpha"])
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
        aw = torch.nn.functional.dropout(aw, p=dropout, training=module.training)
        out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
        return out, aw
    q2.eager_attention_forward = mata_eager

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[mata] loading {MODEL} (alpha={alpha}, layers {_MATA['lo']}-{_MATA['hi']})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate
    AUD = getattr(model.config, "audio_token_index", None)

    def _gen(conv, audio16k, mata_on):
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        ids = inputs["input_ids"][0]
        if AUD is not None:
            pos = (ids == AUD).nonzero(as_tuple=True)[0]
            _MATA["span"] = (int(pos.min()), int(pos.max())) if len(pos) else None
        _MATA["on"] = mata_on and _MATA["span"] is not None
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        _MATA["on"] = False
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def base(a, si, mi, mata):
        return _gen([{"role": "user", "content": [
            {"type": "audio", "audio_url": "x"}, {"type": "text", "text": mi}]}], a, mata)
    def anchor_chain(a, si, mi, mata):
        c = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": si}]}]
        a1 = _gen(c, a, mata)
        c += [{"role": "assistant", "content": [{"type": "text", "text": a1}]},
              {"role": "user", "content": [{"type": "text", "text": mi}]}]
        return _gen(c, a, mata)

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds = ds.select(range(lo, hi))
        c = {"base": 0, "mata": 0, "anchor": 0, "both": 0}; nn = 0
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, mi, g = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            c["base"]   += int(mcq_correct(base(a, si, mi, False), g, mi))
            c["mata"]   += int(mcq_correct(base(a, si, mi, True),  g, mi))
            c["anchor"] += int(mcq_correct(anchor_chain(a, si, mi, False), g, mi))
            c["both"]   += int(mcq_correct(anchor_chain(a, si, mi, True),  g, mi))
            nn += 1
        res[track] = {k: v / nn for k, v in c.items()}
        r = res[track]
        print(f"[mata] {track} [{lo}:{hi}] base={r['base']:.3f} MATA={r['mata']:.3f} anchor={r['anchor']:.3f} BOTH={r['both']:.3f}", flush=True)
    avg = {k: sum(res[t][k] for t in res) / len(res) for k in ["base", "mata", "anchor", "both"]}
    print(f"\n[mata] ===== AVG base={avg['base']:.3f} MATA={avg['mata']:.3f} anchor={avg['anchor']:.3f} BOTH={avg['both']:.3f} =====", flush=True)
    print(f"[mata] MATA lift={avg['mata']-avg['base']:+.3f} | anchor lift={avg['anchor']-avg['base']:+.3f} | BOTH lift={avg['both']-avg['base']:+.3f}", flush=True)
    return {"per_track": res, "avg": avg}


# ==================================================================================
# DISTILLATION step 1: rejection-sampling (STaR) trace generation on a SAKURA TRAIN slice.
# Keep chains that reach the correct answer -> distillation data. Yield per track = the
# diagnostic (low Gender yield => residual needs an EXTERNAL teacher, not self-distill).
# ==================================================================================
@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_distill_data(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 150, kchains: int = 8):
    """Generate correct (audio,question->reasoning) traces via rejection sampling. Save JSONL."""
    import numpy as np, torch, librosa, json
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[distill-data] loading {MODEL} (K={kchains}, train [{start}:{start+n}])...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate

    def _gen(conv, audio16k, k):
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            if k == 1:
                out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=300, do_sample=True,
                                     temperature=0.8, top_p=0.95, num_return_sequences=k)
        return [d.strip() for d in processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

    import os as _os
    _os.makedirs("/data/distill", exist_ok=True)
    yields = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        kept = 0; nn = 0
        path = f"/data/distill/{track}_{lo}_{hi}.jsonl"
        with open(path, "w") as fout:
            for idx_in, ex in enumerate(ds_sel):
                a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
                if isinstance(a, list): a = np.array(a, dtype=np.float32)
                a = a.astype(np.float32)
                if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
                si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
                # one-shot CoT chains directly on the multi question (what we distill: direct perceive->chain)
                prompt = mi + "\nFirst state the audio attribute, then reason step by step, then give the final answer."
                chains = _gen([{"role": "user", "content": [
                    {"type": "audio", "audio_url": "x"}, {"type": "text", "text": prompt}]}], a, kchains)
                correct = [c for c in chains if mcq_correct(c, gold, mi)]
                nn += 1
                if correct:
                    kept += 1
                    correct.sort(key=len)  # shortest correct trace (compress)
                    fout.write(json.dumps({"track": track, "ds_index": lo + idx_in,
                                           "instruction": prompt, "trace": correct[0], "gold": gold}) + "\n")
                if nn % 25 == 0: print(f"[distill-data] {track} {nn}/{len(ds_sel)} kept={kept} yield={kept/nn:.2f}", flush=True)
        yields[track] = {"kept": kept, "n": nn, "yield": kept / nn, "path": path}
        print(f"[distill-data] {track} YIELD={kept}/{nn}={kept/nn:.2f}  -> {path}", flush=True)
    vol.commit()
    print(f"\n[distill-data] yields: " + " ".join(f"{t}={y['yield']:.2f}" for t, y in yields.items()), flush=True)
    return yields


# ==================================================================================
# IMPROVED BACKTRACK: verify = PERCEPTION self-consistency (extract attribute K times via
# the MCQ single-hop prompt; if the picks DISAGREE -> perception shaky -> BACKTRACK: re-perceive
# with a focused prompt, re-chain). Targets the perception-error part of the residual that
# chain-agreement misses. vs base. Reports lift + backtrack rate + did-backtrack-help.
# ==================================================================================
@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_backtrack(tracks: str = "GenderQA,EmotionQA,AnimalQA", start: int = 300, n: int = 30,
                   kperc: int = 5, thr: float = 0.6, max_rounds: int = 3):
    """Perception-self-consistency-gated LOOP/backtrack on Qwen2-Audio SAKURA multi-hop.
    max_rounds=1 => backtrack (one correction); >1 => looping (iterate until perception converges)."""
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[bt] loading {MODEL} (kperc={kperc}, thr={thr})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vol.commit()
    SR = processor.feature_extractor.sampling_rate

    def _gen(conv, audio16k, k=1, sample=False):
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            if not sample:
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                     temperature=0.8, top_p=0.95, num_return_sequences=k)
        return [d.strip() for d in processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

    def U(t, path=False): return {"role": "user", "content": ([{"type": "audio", "audio_url": "x"}] if path else []) + [{"type": "text", "text": t}]}
    def Asst(t): return {"role": "assistant", "content": [{"type": "text", "text": t}]}

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        base_c = m_c = bt_n = bt_help = total_rounds = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            base = _gen([U(mi, True)], a, 1)[0]
            # VERIFY = perception self-consistency: extract attribute K times, pick option each
            percepts = _gen([{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": si}]}], a, kperc, sample=True)
            picks = [p for p in (pick_option(c, si) for c in percepts) if p]
            agree = (Counter(picks).most_common(1)[0][1] / len(picks)) if picks else 0.0
            anchor = percepts[0]; acc = list(picks); rounds = 0
            while agree < thr and rounds < max_rounds:  # LOOP: re-perceive until perception converges
                rounds += 1
                more = _gen([{"role": "user", "content": [{"type": "audio", "audio_url": "x"},
                    {"type": "text", "text": "Listen very carefully and focus only on this. " + si}]}], a, kperc, sample=True)
                mp = [p for p in (pick_option(c, si) for c in more) if p]
                acc += mp
                if acc:
                    agree = Counter(acc).most_common(1)[0][1] / len(acc)
                anchor = more[0]
            if rounds > 0: bt_n += 1; total_rounds += rounds
            ans = _gen([U(si, True), Asst(anchor), U(mi)], a, 1)[0]
            base_ok = mcq_correct(base, gold, mi); m_ok = mcq_correct(ans, gold, mi)
            base_c += int(base_ok); m_c += int(m_ok); nn += 1
            if agree < thr and m_ok and not base_ok: bt_help += 1
        res[track] = {"base": base_c/nn, "method": m_c/nn, "lift": (m_c-base_c)/nn, "bt_rate": bt_n/nn, "bt_fixed": bt_help, "n": nn}
        r = res[track]
        print(f"[bt] {track} [{lo}:{hi}] base={r['base']:.3f} method={r['method']:.3f} lift={r['lift']:+.3f} loop_rate={r['bt_rate']:.2f} avg_rounds={(total_rounds/max(1,bt_n)):.1f} fixed={bt_help}", flush=True)
    avgb = sum(r["base"] for r in res.values())/len(res); avgm = sum(r["method"] for r in res.values())/len(res)
    print(f"\n[bt] ===== AVG base={avgb:.3f} method={avgm:.3f} lift={avgm-avgb:+.3f} =====", flush=True)
    return {"per_track": res, "avg_base": avgb, "avg_method": avgm}


# ==================================================================================
# DISTILLATION step 1b: GROUNDING FILTER + inspect. Keep only traces that explicitly cite
# the CORRECT perceived attribute (MGRD-style) -> removes lucky-guess / text-surrogate traces.
# Prints surviving count + sample traces so we SEE quality before training. CPU (cheap).
# ==================================================================================
@app.function(image=qwen_img, volumes={"/data": vol}, timeout=3600)
def distill_filter(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 150):
    """Grounding-filter the saved STaR traces; report real yield + samples."""
    import json, os, re as _re
    from datasets import load_dataset

    kept_all = 0; raw_all = 0
    for track in tracks.split(","):
        path = f"/data/distill/{track}_{start}_{start+n}.jsonl"
        if not os.path.exists(path):
            print(f"[filter] MISSING {path}", flush=True); continue
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        # map ds_index -> correct single-hop attribute text
        rows = [json.loads(l) for l in open(path)]
        out = f"/data/distill/{track}_{start}_{start+n}_grounded.jsonl"
        kept = 0; shown = 0
        with open(out, "w") as fo:
            for r in rows:
                ex = ds[r["ds_index"]]
                _, attr = _gold_letter_text(ex["single_answer"])   # e.g. "Female","German"
                attr = (attr or "").strip()
                grounded = bool(attr) and _re.search(r"\b" + _re.escape(attr.lower()) + r"\b", r["trace"].lower())
                if grounded:
                    kept += 1; fo.write(json.dumps(r) + "\n")
                    if shown < 2:
                        shown += 1
                        print(f"\n[sample {track}] attr={attr!r} gold={r['gold']!r}\n  trace={r['trace'][:220]!r}", flush=True)
        raw_all += len(rows); kept_all += kept
        print(f"[filter] {track}: raw={len(rows)} grounded={kept} ({kept/max(1,len(rows)):.2f}) -> {out}", flush=True)
    vol.commit()
    print(f"\n[filter] TOTAL grounded={kept_all}/{raw_all} ({kept_all/max(1,raw_all):.2f})", flush=True)
    return {"kept": kept_all, "raw": raw_all}


# training image = qwen_img + peft for LoRA SFT
qwen_train_img = qwen_img.pip_install("peft==0.14.0")

# ==================================================================================
# DISTILLATION step 2: LoRA SFT (self-distillation/STaR) on grounded traces. Cogito config
# (LoRA, 1 epoch, lr 1e-5). max_steps>0 = quick sanity. Saves adapter to /data/distill/lora.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_distill_train(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 150,
                       epochs: int = 1, lr: float = 1e-5, max_steps: int = -1, suffix: str = "grounded"):
    """LoRA SFT Qwen2-Audio on grounded STaR traces; loss only on the assistant trace."""
    import json, os, torch, numpy as np, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[sft] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    SR = processor.feature_extractor.sampling_rate

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg); model.print_trainable_parameters()
    model.config.use_cache = False

    # load grounded traces + reload audio by ds_index
    examples = []
    ds_cache = {}
    for track in tracks.split(","):
        path = f"/data/distill/{track}_{start}_{start+n}_{suffix}.jsonl"
        if not os.path.exists(path): print(f"[sft] missing {path}", flush=True); continue
        if track not in ds_cache: ds_cache[track] = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        for l in open(path):
            r = json.loads(l); r["track"] = track; examples.append(r)
    print(f"[sft] {len(examples)} grounded training traces", flush=True)

    def get_audio(r):
        ex = ds_cache[r["track"]][r["ds_index"]]
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32)
        if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
        return a

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(examples)
        def __getitem__(self, i):
            r = examples[i]; a = get_audio(r)
            user = {"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": r["instruction"]}]}
            asst = {"role": "assistant", "content": [{"type": "text", "text": r["trace"]}]}
            ptext = processor.apply_chat_template([user], add_generation_prompt=True, tokenize=False)
            ftext = processor.apply_chat_template([user, asst], tokenize=False)
            full = processor(text=ftext, audios=[a], sampling_rate=SR, return_tensors="pt")
            plen = processor(text=ptext, audios=[a], sampling_rate=SR, return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone(); labels[:, :plen] = -100
            full["labels"] = labels
            return {k: v[0] for k, v in full.items()}

    def collate(b):  # batch_size 1 -> just stack the single example
        return {k: (torch.stack([x[k] for x in b]) if x_has(b, k) else b[0][k]) for k in b[0]}
    def x_has(b, k): return all(k in x for x in b)

    args = TrainingArguments(output_dir="/data/distill/lora", per_device_train_batch_size=1,
                             gradient_accumulation_steps=8, num_train_epochs=epochs, learning_rate=lr,
                             logging_steps=2, save_strategy="no", bf16=True, max_steps=max_steps,
                             report_to=[], remove_unused_columns=False, dataloader_num_workers=2)
    trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
    print(f"[sft] training (max_steps={max_steps}, epochs={epochs})...", flush=True)
    trainer.train()
    if max_steps <= 0:
        model.save_pretrained("/data/distill/lora_adapter"); vol.commit()
        print("[sft] saved adapter -> /data/distill/lora_adapter", flush=True)
    return {"n_train": len(examples), "max_steps": max_steps}


# ==================================================================================
# DISTILLATION step 3: eval base vs SFT-LoRA on HELD-OUT slice [50:100]. Same prompt as
# training. Strict MCQ. Tells us if self-distillation closed what inference couldn't.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_distill_eval(tracks: str = ",".join(_TRACKS), start: int = 50, n: int = 50):
    """Eval base vs LoRA-SFT on held-out SAKURA multi-hop."""
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[eval] loading base {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print("[eval] loading LoRA adapter...", flush=True)
    sft_model = PeftModel.from_pretrained(
        Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda"),
        "/data/distill/lora_adapter").eval()
    SR = processor.feature_extractor.sampling_rate
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."

    def gen(model, audio16k, mi):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": mi + PROMPT}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    from collections import Counter
    def gen_sc(model, audio16k, mi, k=5):  # SFT + self-consistency (the COMBO)
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": mi + PROMPT}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                 top_p=0.9, num_return_sequences=k)
        chains = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        bc = sc = cc = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            mi, gold = ex["multi_instruction"], ex["multi_answer"]
            bc += int(mcq_correct(gen(base_model, a, mi), gold, mi))
            sc += int(mcq_correct(gen(sft_model, a, mi), gold, mi))
            cc += int(mcq_correct(gen_sc(sft_model, a, mi), gold, mi))  # SFT + self-consistency
            nn += 1
        res[track] = {"base": bc/nn, "sft": sc/nn, "combo": cc/nn, "n": nn}
        r = res[track]
        print(f"[eval] {track} [{lo}:{hi}] base={r['base']:.3f} SFT={r['sft']:.3f} SFT+SC={r['combo']:.3f}", flush=True)
    avgb = sum(r["base"] for r in res.values())/len(res)
    avgs = sum(r["sft"] for r in res.values())/len(res)
    avgc = sum(r["combo"] for r in res.values())/len(res)
    print(f"\n[eval] ===== HELD-OUT [{start}:{start+n}]  base={avgb:.3f}  SFT={avgs:.3f}  SFT+SC(combo)={avgc:.3f} =====", flush=True)
    print(f"[eval] vs AF-Next frontier 0.523 -> SFT beats? {avgs>0.523} | COMBO beats? {avgc>0.523}", flush=True)
    return {"per_track": res, "avg_base": avgb, "avg_sft": avgs, "avg_combo": avgc}


_ATTR_NAME = {"GenderQA": "speaker's gender", "LanguageQA": "spoken language",
              "EmotionQA": "speaker's emotion", "AnimalQA": "animal making the sound"}

# ==================================================================================
# CROSS-MODAL self-distillation data: TEXT-TEACHER. Generate the chain in TEXT-ONLY mode with
# the GOLD attribute injected -> the model's strong text reasoning produces a correct, grounded
# chain even for the residual. Higher ceiling than audio-STaR. Saves to _grounded.jsonl.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def xmodal_distill_data(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 150, kchains: int = 4):
    """Text-teacher (gold-attribute-injected, no audio) -> correct grounded chains for SFT."""
    import json, os, numpy as np, torch
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[xmodal] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    tok = processor.tokenizer
    def _gen_text(prompt, k):
        # text-only: use the TOKENIZER directly (the audio processor requires audio and chokes)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=(k > 1),
                                 temperature=0.7, top_p=0.95, num_return_sequences=k)
        return [d.strip() for d in tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

    os.makedirs("/data/distill", exist_ok=True)
    tot_kept = 0
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        attr_name = _ATTR_NAME[track]
        out = f"/data/distill/{track}_{lo}_{hi}_grounded.jsonl"
        kept = 0; nn = 0
        with open(out, "w") as fo:
            for i_in, ex in enumerate(ds_sel):
                _, attr = _gold_letter_text(ex["single_answer"])
                mi, gold = ex["multi_instruction"], ex["multi_answer"]
                # the SFT *input* prompt (audio-time, no gold) — what the student sees:
                student_prompt = mi + "\nFirst state the audio attribute, then reason step by step, then give the final answer."
                # the TEACHER prompt (text-only, gold injected) — produces the target chain:
                teacher_prompt = (f"The {attr_name} is {attr}. Using ONLY this fact, answer the question. "
                                  f"Start by stating '{attr}', then reason step by step, then give the final answer.\n{mi}")
                chains = _gen_text(teacher_prompt, kchains)
                # keep correct chains that are grounded (state the attribute)
                good = [c for c in chains if mcq_correct(c, gold, mi) and _word_in(c, attr)]
                nn += 1
                if good:
                    good.sort(key=len); kept += 1
                    fo.write(json.dumps({"track": track, "ds_index": lo + i_in,
                                         "instruction": student_prompt, "trace": good[0], "gold": gold}) + "\n")
                if nn % 30 == 0: print(f"[xmodal] {track} {nn}/{len(ds_sel)} kept={kept}", flush=True)
        tot_kept += kept
        print(f"[xmodal] {track} kept={kept}/{nn} -> {out}", flush=True)
    vol.commit()
    print(f"\n[xmodal] TOTAL teacher traces = {tot_kept}", flush=True)
    return {"kept": tot_kept}


# ==================================================================================
# Pure-inference SC on the BASE model (no SFT) — apples-to-apples vs SFT+SC. Settles
# "does training help beyond inference?" Same slice/prompt/K as qwen_distill_eval.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_base_sc(tracks: str = ",".join(_TRACKS), start: int = 50, n: int = 50, kchains: int = 5):
    """Base Qwen2-Audio + self-consistency (NO SFT) on held-out slice, strict MCQ."""
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[base+sc] loading {MODEL} (K={kchains})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."

    def gen_sc(audio16k, mi, k):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": mi + PROMPT}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                 top_p=0.9, num_return_sequences=k)
        chains = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        c = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            mi, gold = ex["multi_instruction"], ex["multi_answer"]
            c += int(mcq_correct(gen_sc(a, mi, kchains), gold, mi)); nn += 1
        res[track] = c/nn
        print(f"[base+sc] {track} [{lo}:{hi}] base+SC={c/nn:.3f}", flush=True)
    avg = sum(res.values())/len(res)
    print(f"\n[base+sc] ===== HELD-OUT [{start}:{start+n}] base+SC={avg:.3f} (vs SFT+SC=0.585, frontier=0.523) =====", flush=True)
    return {"per_track": res, "avg": avg}


# ==================================================================================
# TRUE distillation from a STRONG external teacher (DeepSeek-R1-Distill-Qwen-7B). Text-only,
# gold-attribute injected -> high-quality grounded chains. Kept SHORT (long CoT hurts audio).
# Saves _deepseek.jsonl. (vs our self-distillation which used Qwen's own text side.)
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def deepseek_distill_data(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 150):
    """R1-Distill-Qwen-7B text-teacher -> correct grounded SHORT chains for audio distillation."""
    import json, os, re as _re, torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"[deepseek] loading {MODEL}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vol.commit()

    def gen(prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    os.makedirs("/data/distill", exist_ok=True)
    tot = 0
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        attr_name = _ATTR_NAME[track]
        out = f"/data/distill/{track}_{lo}_{hi}_deepseek.jsonl"
        kept = nn = 0
        with open(out, "w") as fo:
            for i_in, ex in enumerate(ds_sel):
                _, attr = _gold_letter_text(ex["single_answer"])
                mi, gold = ex["multi_instruction"], ex["multi_answer"]
                student_prompt = mi + "\nFirst state the audio attribute, then reason step by step, then give the final answer."
                teacher = (f"The {attr_name} is {attr}. Using only this, answer the question in 2-3 short sentences: "
                           f"first state '{attr}', then one reasoning step, then 'Answer: (letter)'.\n{mi}")
                raw = gen(teacher)
                # strip <think>...</think>, keep concise; require correct + grounded
                chain = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
                if not chain: chain = raw.strip()
                chain = chain[:400]  # keep SHORT
                nn += 1
                if mcq_correct(chain, gold, mi) and _word_in(chain, attr):
                    kept += 1
                    fo.write(json.dumps({"track": track, "ds_index": lo + i_in,
                                         "instruction": student_prompt, "trace": chain, "gold": gold}) + "\n")
                if nn % 30 == 0: print(f"[deepseek] {track} {nn}/{len(ds_sel)} kept={kept}", flush=True)
        tot += kept
        print(f"[deepseek] {track} kept={kept}/{nn} -> {out}", flush=True)
    vol.commit()
    print(f"\n[deepseek] TOTAL R1-teacher traces = {tot}", flush=True)
    return {"kept": tot}


# ==================================================================================
# NO-AUDIO control: self-consistency on the QUESTION TEXT ONLY (no audio). If this ≈ audio
# base+SC (0.600), the result is text-prior skew (inflated). If << 0.600, the gain is genuinely
# audio-grounded. The decisive "is it real audio reasoning?" test.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_noaudio_sc(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 500, kchains: int = 5):
    """Text-only self-consistency (NO audio) — control for text-prior skew."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # hf_transfer is flaky
    import numpy as np, torch
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[noaudio] loading {MODEL} (K={kchains}, TEXT-ONLY)...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."

    def sc_text(mi, k):
        text = tok.apply_chat_template([{"role": "user", "content": mi + PROMPT}],
                                       add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                 top_p=0.9, num_return_sequences=k)
        chains = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        c = nn = 0
        for ex in ds_sel:
            mi, gold = ex["multi_instruction"], ex["multi_answer"]
            c += int(mcq_correct(sc_text(mi, kchains), gold, mi)); nn += 1
            if nn % 30 == 0: print(f"[noaudio] {track} {nn}/{len(ds_sel)} running={c/nn:.3f}", flush=True)
        res[track] = c/nn
        print(f"[noaudio] {track} [{lo}:{hi}] noaudio+SC={c/nn:.3f}", flush=True)
    avg = sum(res.values())/len(res)
    print(f"\n[noaudio] ===== TEXT-ONLY [{start}:{start+n}] noaudio+SC={avg:.3f} (vs audio base+SC=0.600) =====", flush=True)
    print(f"[noaudio] {'TEXT-PRIOR SKEW (result inflated!)' if avg > 0.55 else 'audio-grounded (gain is real)'}", flush=True)
    return {"per_track": res, "avg": avg}


# ==================================================================================
# AF-Next + self-consistency — the FAIR cross-model comparison (same CoT+vote as Qwen base+SC).
# Turns "a 7B beats the frontier" into "method works ON the frontier too".
# ==================================================================================
@app.function(image=afnext_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def afnext_base_sc(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 150, kchains: int = 5):
    """AF-Next + self-consistency on SAKURA multi-hop, strict MCQ."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, soundfile as sf, tempfile
    from collections import Counter
    import transformers
    from transformers import AutoProcessor, AutoConfig
    from datasets import load_dataset

    MODEL = "nvidia/audio-flamingo-next-hf"
    print(f"[afnext-sc] loading {MODEL} (K={kchains})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    arch = AutoConfig.from_pretrained(MODEL).architectures[0]
    model = getattr(transformers, arch).from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."

    def sc(audio16k, mi, k):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio16k, SR); path = f.name
        conv = [{"role": "user", "content": [{"type": "text", "text": mi + PROMPT}, {"type": "audio", "path": path}]}]
        inputs = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True)
        inputs = {kk: (v.to(model.device) if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                 top_p=0.9, num_return_sequences=k)
        chains = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        _os.unlink(path)
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        c = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            mi, gold = ex["multi_instruction"], ex["multi_answer"]
            c += int(mcq_correct(sc(a, mi, kchains), gold, mi)); nn += 1
            if nn % 25 == 0: print(f"[afnext-sc] {track} {nn}/{len(ds_sel)} running={c/nn:.3f}", flush=True)
        res[track] = c/nn
        print(f"[afnext-sc] {track} [{lo}:{hi}] AFNext+SC={c/nn:.3f}", flush=True)
    avg = sum(res.values())/len(res)
    print(f"\n[afnext-sc] ===== [{start}:{start+n}] AFNext+SC={avg:.3f} (AF-Next base 0.523, Qwen+SC 0.600) =====", flush=True)
    return {"per_track": res, "avg": avg}


# ==================================================================================
# AudioLens-style perception fix: save early-layer hidden state (attribute-rich), inject a
# scaled copy into a deeper layer (recover the info that degrades by the final layer).
# Tests SINGLE-HOP perception (the ceiling) base vs injected. White-box forward hooks.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_audiolens(tracks: str = "GenderQA,EmotionQA", n: int = 50,
                   l_early: int = 14, l_deep: int = 24, alpha: float = 4.0):
    """AudioLens early->deep layer injection; single-hop perception base vs injected."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[audiolens] loading {MODEL} (inject L{l_early}->L{l_deep}, alpha={alpha})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate

    # locate the LLM decoder layers
    llm = model.language_model
    layers = llm.model.layers if hasattr(llm, "model") else llm.layers
    print(f"[audiolens] {len(layers)} LLM layers", flush=True)

    _HS = {}; _ON = {"v": False}
    def save_hook(mod, inp, out):
        if _ON["v"]: _HS["e"] = (out[0] if isinstance(out, tuple) else out).detach()
    def inject_hook(mod, inp, out):
        if _ON["v"] and "e" in _HS:
            h = out[0] if isinstance(out, tuple) else out
            if _HS["e"].shape == h.shape:
                h = h + alpha * _HS["e"]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out
    layers[l_early].register_forward_hook(save_hook)
    layers[l_deep].register_forward_hook(inject_hook)

    def ask(audio16k, instruction):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test").select(range(min(n, 99999)))
        ds = ds.select(range(min(n, len(ds))))
        b = inj = nn = 0
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, gold = ex["single_instruction"], ex["single_answer"]
            _ON["v"] = False; base = ask(a, si)              # base (no injection)
            _ON["v"] = True;  injd = ask(a, si)              # with injection
            b += int(mcq_correct(base, gold, si)); inj += int(mcq_correct(injd, gold, si)); nn += 1
        res[track] = {"base": b/nn, "injected": inj/nn, "lift": (inj-b)/nn}
        r = res[track]
        print(f"[audiolens] {track} single-hop base={r['base']:.3f} injected={r['injected']:.3f} lift={r['lift']:+.3f}", flush=True)
    avgb = sum(r["base"] for r in res.values())/len(res); avgi = sum(r["injected"] for r in res.values())/len(res)
    print(f"\n[audiolens] ===== single-hop base={avgb:.3f} injected={avgi:.3f} lift={avgi-avgb:+.3f} (L{l_early}->L{l_deep} a={alpha}) =====", flush=True)
    return {"per_track": res, "avg_base": avgb, "avg_injected": avgi}


# ==================================================================================
# AudioLens sweep — HONEST protocol: tune (l_early, l_deep, alpha) on a DEV split, pick ONE
# config, eval on a HELD-OUT TEST split. Single uniform config across tracks (no per-track
# hacking). Generalization to unseen test = proof it's a general fix, not a benchmark hack.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_audiolens_sweep(tracks: str = "GenderQA,EmotionQA",
                         dev_start: int = 350, dev_n: int = 60,
                         test_start: int = 0, test_n: int = 100):
    """Dev-sweep AudioLens injection configs, pick best, eval on held-out test. Single-hop."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[sweep] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate
    llm = model.language_model
    layers = llm.model.layers if hasattr(llm, "model") else llm.layers

    # dynamic hooks: every layer saves its output; the configured l_deep injects scaled l_early
    _HS = {}; _CFG = {"on": False, "le": 14, "ld": 24, "a": 4.0}
    def make_hook(idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            _HS[idx] = h.detach()
            if _CFG["on"] and idx == _CFG["ld"] and _CFG["le"] in _HS:
                e = _HS[_CFG["le"]]
                if e.shape == h.shape:
                    h = h + _CFG["a"] * e
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        return hook
    for i, L in enumerate(layers): L.register_forward_hook(make_hook(i))

    def ask(audio16k, instruction):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def eval_split(start, n):  # returns dict: config_label -> avg single-hop acc
        items = []
        for track in tracks.split(","):
            ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
            lo, hi = start, min(start + n, len(ds))
            for ex in ds.select(range(lo, hi)):
                a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
                if isinstance(a, list): a = np.array(a, dtype=np.float32)
                a = a.astype(np.float32)
                if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
                items.append((a, ex["single_instruction"], ex["single_answer"]))
        return items

    GRID = [(le, ld, al) for le in (10, 14, 18) for ld in (26,) for al in (4.0, 8.0)]

    # ---- DEV sweep ----
    dev = eval_split(dev_start, dev_n)
    def score(items):
        c = sum(int(mcq_correct(ask(a, si), g, si)) for a, si, g in items)
        return c / len(items)
    _CFG["on"] = False; dev_base = score(dev)
    print(f"[sweep] DEV base single-hop = {dev_base:.3f}", flush=True)
    results = {}
    for (le, ld, al) in GRID:
        _CFG.update(on=True, le=le, ld=ld, a=al)
        acc = score(dev); results[(le, ld, al)] = acc
        print(f"[sweep] DEV inject L{le}->L{ld} a={al}: {acc:.3f} (lift {acc-dev_base:+.3f})", flush=True)
    best = max(results, key=results.get)
    print(f"[sweep] BEST dev config = L{best[0]}->L{best[1]} a={best[2]} (dev {results[best]:.3f} vs base {dev_base:.3f})", flush=True)

    # ---- TEST the single best config on held-out split ----
    test = eval_split(test_start, test_n)
    _CFG["on"] = False; test_base = score(test)
    _CFG.update(on=True, le=best[0], ld=best[1], a=best[2]); test_inj = score(test)
    print(f"\n[sweep] ===== HELD-OUT TEST [{test_start}:{test_start+test_n}] base={test_base:.3f} "
          f"best-config-injected={test_inj:.3f} lift={test_inj-test_base:+.3f} =====", flush=True)
    print(f"[sweep] {'GENERALIZES (real fix)' if test_inj > test_base else 'does NOT generalize (dev-overfit)'}", flush=True)
    return {"dev_base": dev_base, "dev": {str(k): v for k, v in results.items()}, "best": str(best),
            "test_base": test_base, "test_injected": test_inj}


# ==================================================================================
# FAITHFUL AudioLens: (1) PROBE via logit-lens to find the layer where the attribute is most
# decodable (l_early); (2) inject scaled l_early -> l_deep at the LAST (answer) position only,
# with SMALL alpha. Dev-probe+alpha-sweep, then held-out test. (Fixes the blind/too-strong v1.)
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_audiolens_faithful(tracks: str = "GenderQA,EmotionQA", dev_start: int = 350, dev_n: int = 40,
                            test_start: int = 0, test_n: int = 100, l_deep: int = 24):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[faithful] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL); tok = processor.tokenizer
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate
    llm = model.language_model
    layers = llm.model.layers if hasattr(llm, "model") else llm.layers
    norm = llm.model.norm if hasattr(llm, "model") else llm.norm
    lm_head = llm.lm_head

    def prep(audio16k, instruction):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    def load_items(start, n):
        items = []
        for track in tracks.split(","):
            ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
            lo, hi = start, min(start + n, len(ds))
            for ex in ds.select(range(lo, hi)):
                a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
                if isinstance(a, list): a = np.array(a, dtype=np.float32)
                a = a.astype(np.float32)
                if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
                _, attr = _gold_letter_text(ex["single_answer"])
                items.append((a, ex["single_instruction"], ex["single_answer"], attr))
        return items

    # ---- (1) PROBE: logit-lens info score per layer (which layer best decodes the attribute) ----
    dev = load_items(dev_start, dev_n)
    nL = len(layers)
    rr = torch.zeros(nL + 1)  # reciprocal-rank of the gold attribute token, per layer
    for (a, si, _, attr) in dev:
        gt = tok(attr, add_special_tokens=False).input_ids
        if not gt: continue
        gt = gt[0]
        inputs = prep(a, si)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for li, h in enumerate(out.hidden_states):
            logits = lm_head(norm(h[:, -1]))[0].float()
            rank = int((logits > logits[gt]).sum().item())
            rr[li] += 1.0 / (rank + 1)
    rr /= len(dev)
    # restrict to EARLY-to-mid layers before l_deep (the final layer trivially wins logit-lens
    # and is useless for injection; we need an earlier layer to inject INTO a deeper one)
    cand = list(range(6, l_deep))
    l_early = int(max(cand, key=lambda i: rr[i].item()))
    print(f"[faithful] PROBE: best EARLY attribute-info layer l_early={l_early} (score {rr[l_early]:.4f}, "
          f"final-layer {rr[-1]:.4f}); l_deep={l_deep}", flush=True)

    # ---- (2) inject at LAST position only, small alpha ----
    _HS = {}; _CFG = {"on": False, "a": 1.0}
    def save_hook(mod, inp, out):
        _HS["e"] = (out[0] if isinstance(out, tuple) else out).detach()
    def inject_hook(mod, inp, out):
        if _CFG["on"] and "e" in _HS:
            h = out[0] if isinstance(out, tuple) else out
            if _HS["e"].shape == h.shape:
                h = h.clone(); h[:, -1, :] = h[:, -1, :] + _CFG["a"] * _HS["e"][:, -1, :]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out
    layers[l_early].register_forward_hook(save_hook)
    layers[l_deep].register_forward_hook(inject_hook)

    def ask(a, si):
        inputs = prep(a, si)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    def score(items):
        return sum(int(mcq_correct(ask(a, si), g, si)) for a, si, g, _ in items) / len(items)

    # ---- alpha sweep on dev (SMALL alphas this time) ----
    _CFG["on"] = False; dev_base = score(dev)
    print(f"[faithful] DEV base = {dev_base:.3f}", flush=True)
    best_a, best_acc = None, -1
    for al in (0.5, 1.0, 2.0):
        _CFG.update(on=True, a=al); acc = score(dev)
        print(f"[faithful] DEV alpha={al} (L{l_early}->L{l_deep}): {acc:.3f} (lift {acc-dev_base:+.3f})", flush=True)
        if acc > best_acc: best_acc, best_a = acc, al

    # ---- held-out test with best (l_early from probe, best alpha) ----
    test = load_items(test_start, test_n)
    _CFG["on"] = False; tb = score(test)
    _CFG.update(on=True, a=best_a); ti = score(test)
    print(f"\n[faithful] ===== TEST [{test_start}:{test_start+test_n}] L{l_early}->L{l_deep} a={best_a}  base={tb:.3f} injected={ti:.3f} lift={ti-tb:+.3f} =====", flush=True)
    print(f"[faithful] {'GENERALIZES (dev+test both positive)' if (best_acc>dev_base and ti>tb) else 'NULL / noise (not both positive)'}", flush=True)
    return {"l_early": l_early, "best_alpha": best_a, "dev_base": dev_base, "test_base": tb, "test_injected": ti}


# ==================================================================================
# Audio-Specialist-Heads steering (2603.06854; +8pp on Qwen2-Audio/MMAU). Core mechanism:
# s(x) = mean over specialist layers of (h_audio - h_silence) at the last position = "what the
# audio added". Steer: h_final += beta * s. Amplifies the audio's OWN contribution (not a blind
# early layer). Dev beta-sweep -> held-out test. Single-hop perception.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_specialist_steer(tracks: str = "GenderQA,EmotionQA", dev_start: int = 350, dev_n: int = 40,
                          test_start: int = 0, test_n: int = 100, lo: int = 4, hi: int = 11):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[steer] loading {MODEL} (specialist layers {lo}-{hi})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate
    llm = model.language_model
    layers = llm.model.layers if hasattr(llm, "model") else llm.layers

    def prep(audio16k, instruction):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio16k], sampling_rate=SR, return_tensors="pt")
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    # steering vector + hook on the FINAL layer (adds beta*s to last position each step)
    _S = {"v": None, "beta": 0.0, "on": False}
    def steer_hook(mod, inp, out):
        if _S["on"] and _S["v"] is not None:
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone(); h[:, -1, :] = h[:, -1, :] + _S["beta"] * _S["v"].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out
    layers[-1].register_forward_hook(steer_hook)

    def compute_s(audio16k, instruction):
        # audio pass
        ia = prep(audio16k, instruction)
        with torch.no_grad():
            oa = model(**ia, output_hidden_states=True)
        ha = oa.hidden_states
        # matched-duration SILENCE pass
        sil = np.zeros_like(audio16k)
        isl = prep(sil, instruction)
        with torch.no_grad():
            os_ = model(**isl, output_hidden_states=True)
        hs = os_.hidden_states
        # s = mean over specialist layers of (audio - silence) at last position
        L = min(len(ha), len(hs))
        deltas = [(ha[l][:, -1, :] - hs[l][:, -1, :]) for l in range(lo, min(hi + 1, L))]
        return torch.stack(deltas, 0).mean(0).squeeze(0).detach()

    def ask(audio16k, instruction, beta):
        _S["v"] = compute_s(audio16k, instruction) if beta > 0 else None
        _S["beta"] = beta; _S["on"] = beta > 0
        inputs = prep(audio16k, instruction)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        _S["on"] = False
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def load_items(start, n):
        items = []
        for track in tracks.split(","):
            ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
            for ex in ds.select(range(start, min(start + n, len(ds)))):
                a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
                if isinstance(a, list): a = np.array(a, dtype=np.float32)
                a = a.astype(np.float32)
                if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
                items.append((a, ex["single_instruction"], ex["single_answer"]))
        return items
    def score(items, beta):
        return sum(int(mcq_correct(ask(a, si, beta), g, si)) for a, si, g in items) / len(items)

    dev = load_items(dev_start, dev_n)
    dev_base = score(dev, 0.0); print(f"[steer] DEV base = {dev_base:.3f}", flush=True)
    best_b, best_acc = 0.0, dev_base
    for b in (0.5, 1.0, 1.5):
        acc = score(dev, b); print(f"[steer] DEV beta={b}: {acc:.3f} (lift {acc-dev_base:+.3f})", flush=True)
        if acc > best_acc: best_acc, best_b = acc, b
    test = load_items(test_start, test_n)
    tb = score(test, 0.0); ti = score(test, best_b) if best_b > 0 else tb
    print(f"\n[steer] ===== TEST [{test_start}:{test_start+test_n}] beta={best_b} base={tb:.3f} steered={ti:.3f} lift={ti-tb:+.3f} =====", flush=True)
    print(f"[steer] {'GENERALIZES' if (best_b>0 and best_acc>dev_base and ti>tb) else 'NULL/noise'}", flush=True)
    return {"best_beta": best_b, "dev_base": dev_base, "test_base": tb, "test_steered": ti}


# ==================================================================================
# TOOL-AUGMENTED PERCEPTION (Audio-Maestro / AudioToolAgent family). Route each attribute
# to a DEDICATED specialist classifier that sees ONLY the audio (never the gold), then inject
# its prediction into Qwen for multi-hop chaining + self-consistency. General, not a benchmark
# hack: real voice agents route perception to specialists. Two phases with a hard gate:
#   Phase A = specialist single-hop accuracy (must BEAT Qwen single-hop, else dead).
#   Phase B = tool+SC multi-hop vs base+SC (0.600).
# Specialists are transformers-native (no dep hell): MMS-LID (language), AST/AudioSet (animal).
# ==================================================================================
qwen_tool_img = qwen_train_img.pip_install("langcodes", "language_data")

_SPECIALIST = {
    "LanguageQA": ("facebook/mms-lid-256", "lid"),
    "AnimalQA":   ("MIT/ast-finetuned-audioset-10-10-0.4593", "tag"),
    "GenderQA":   ("alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech", "cls"),
}

@app.function(image=qwen_tool_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_tool_perceive(track: str = "LanguageQA", start: int = 0, n: int = 100, kchains: int = 5):
    """Specialist perceives attribute (audio-only) -> inject into Qwen multi-hop + SC.
    Phase A = specialist single-hop (the gate); Phase B = tool+SC multi-hop vs base+SC."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, re as _re
    from collections import Counter
    from datasets import load_dataset
    from transformers import (Qwen2AudioForConditionalGeneration, AutoProcessor,
                              AutoFeatureExtractor, AutoModelForAudioClassification)

    spec_id, kind = _SPECIALIST[track]
    print(f"[tool] track={track} specialist={spec_id} ({kind}) K={kchains}", flush=True)

    QM = "Qwen/Qwen2-Audio-7B-Instruct"
    qproc = AutoProcessor.from_pretrained(QM)
    qwen = Qwen2AudioForConditionalGeneration.from_pretrained(
        QM, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    QSR = qproc.feature_extractor.sampling_rate

    sfe = AutoFeatureExtractor.from_pretrained(spec_id)
    smodel = AutoModelForAudioClassification.from_pretrained(spec_id).to("cuda").eval()
    SSR = sfe.sampling_rate
    id2label = smodel.config.id2label

    def iso_to_name(code):
        try:
            import langcodes
            return langcodes.Language.get(code).display_name()
        except Exception:
            return code

    def specialist_labels(a16, topk=5):
        inp = sfe(a16, sampling_rate=SSR, return_tensors="pt")
        inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            logits = smodel(**inp).logits[0]
        ids = logits.topk(min(topk, logits.shape[-1])).indices.tolist()
        labs = []
        for i in ids:
            lab = id2label[i]
            if kind == "lid": lab = iso_to_name(lab)
            labs.append(str(lab))
        return labs

    def match_option(labels, opts):
        """First specialist label whose word overlaps an option -> (letter, canonical option text)."""
        def toks(s): return [w for w in _re.findall(r"[a-z]+", s.lower()) if len(w) > 2]
        for lab in labels:
            lw = toks(lab)
            for L, t in opts.items():
                tw = toks(t)
                if any(w in tw for w in lw) or any(w in lw for w in tw):
                    return L, t
        return None, (labels[0] if labels else "")

    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."
    def qwen_sc(a16q, mi, detected, k):
        hint = f" [A specialist audio classifier detected the {_ATTR_NAME[track]}: {detected}. Use this.]"
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"},
                                             {"type": "text", "text": mi + hint + PROMPT}]}]
        text = qproc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = qproc(text=text, audios=[a16q], sampling_rate=QSR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            out = qwen.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                top_p=0.9, num_return_sequences=k)
        chains = qproc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
    spec_c = tool_c = nn = 0
    for ex in ds_sel:
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32)
        a_spec = librosa.resample(a, orig_sr=sr, target_sr=SSR) if sr != SSR else a
        a_q = librosa.resample(a, orig_sr=sr, target_sr=QSR) if sr != QSR else a
        labels = specialist_labels(a_spec)
        s_opts = _parse_options(ex["single_instruction"])
        sL, sText = match_option(labels, s_opts)
        sgl, _ = _gold_letter_text(ex["single_answer"])
        spec_ok = (sL == sgl); spec_c += int(spec_ok)
        mi, gold = ex["multi_instruction"], ex["multi_answer"]
        detected = sText if sText else (labels[0] if labels else "unknown")
        tool_c += int(mcq_correct(qwen_sc(a_q, mi, detected, kchains), gold, mi)); nn += 1
        if nn <= 5:
            print(f"[tool ex{nn}] spec_top={labels[:3]} -> det={detected!r} | single gold={ex['single_answer']!r} "
                  f"spec_ok={spec_ok} | opts={list(s_opts.values())}", flush=True)
        if nn % 25 == 0:
            print(f"[tool] {nn}/{len(ds_sel)} specialist_single={spec_c/nn:.3f} tool+SC_multi={tool_c/nn:.3f}", flush=True)
    print(f"\n[tool] ===== {track} [{lo}:{hi}] N={nn} =====", flush=True)
    print(f"[tool] PHASE A specialist single-hop = {spec_c/nn:.3f}", flush=True)
    print(f"[tool] PHASE B tool+SC multi-hop     = {tool_c/nn:.3f}  (base+SC ref = 0.600)", flush=True)
    return {"track": track, "specialist_single": spec_c/nn, "tool_sc_multi": tool_c/nn, "n": nn}


# ==================================================================================
# BINDING-LOCALIZATION PROBE (read-only diagnosis; CREME-style logit-lens). Does Qwen2-Audio's
# multi-hop failure LOCALIZE? At the answer position, per layer, track logit-lens prob of the
# INTERMEDIATE attribute token (e.g. "German") vs the FINAL answer token (e.g. the country).
# Split by CORRECT vs WRONG (greedy multi-hop). CREME signature: in WRONG items the intermediate
# stays dominant in upper layers / the final never rises -> a localizable binding site to patch.
# NOTHING is modified (no weight edit, no steering) -> pure diagnosis for our novel layer's design.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_binding_probe(track: str = "AnimalQA", start: int = 0, n: int = 50):
    """Logit-lens per-layer: intermediate-attr token vs final-answer token at the answer position,
    split CORRECT/WRONG. Find whether multi-hop binding localizes to a layer band."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, re as _re
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    QM = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[bind] loading {QM} track={track}...", flush=True)
    proc = AutoProcessor.from_pretrained(QM)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        QM, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate
    lm = model.language_model
    head = lm.lm_head
    fnorm = lm.model.norm
    tok = proc.tokenizer

    def first_tok(word):
        ids = tok.encode(" " + word.strip(), add_special_tokens=False)
        return ids[0] if ids else None

    PROMPT = "\nAnswer with the correct option letter and the answer in a few words."
    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))

    L = model.config.text_config.num_hidden_layers
    # accumulators: per-layer mean prob of intermediate vs final, for correct/wrong
    acc = {"correct": {"intt": np.zeros(L+1), "fin": np.zeros(L+1), "k": 0},
           "wrong":   {"intt": np.zeros(L+1), "fin": np.zeros(L+1), "k": 0}}
    printed = 0
    for ex in ds_sel:
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32)
        if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
        mi, mgold = ex["multi_instruction"], ex["multi_answer"]
        _, attr_text = _gold_letter_text(ex["single_answer"])   # intermediate (e.g. "German")
        _, fin_text = _gold_letter_text(mgold)                  # final answer text
        it_id, fin_id = first_tok(attr_text.split()[0]), first_tok(fin_text.split()[0])
        if it_id is None or fin_id is None: continue
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"},
                                             {"type": "text", "text": mi + PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[a], sampling_rate=SR, return_tensors="pt")
        inputs = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        pred = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ok = mcq_correct(pred, mgold, mi)
        hs = out.hidden_states  # tuple len L+1, each [1, T, H]
        bucket = acc["correct"] if ok else acc["wrong"]
        for li in range(L+1):
            h = hs[li][0, -1, :]                 # answer position
            logits = head(fnorm(h)).float()
            p = torch.softmax(logits, dim=-1)
            bucket["intt"][li] += p[it_id].item()
            bucket["fin"][li]  += p[fin_id].item()
        bucket["k"] += 1
        if printed < 4:
            print(f"[bind ex] ok={ok} attr={attr_text!r}(int) fin={fin_text!r} pred={pred[:50]!r}", flush=True)
            printed += 1

    print(f"\n[bind] ===== {track} [{lo}:{hi}] correct={acc['correct']['k']} wrong={acc['wrong']['k']} =====", flush=True)
    print(f"[bind] layer |  CORRECT(int->fin)   |   WRONG(int->fin)", flush=True)
    for li in range(L+1):
        cc, ww = acc["correct"], acc["wrong"]
        ci = cc["intt"][li]/max(cc["k"],1); cf = cc["fin"][li]/max(cc["k"],1)
        wi = ww["intt"][li]/max(ww["k"],1); wf = ww["fin"][li]/max(ww["k"],1)
        mark = " <" if li in (L//2, L*3//4, L) else ""
        print(f"[bind] L{li:2d}  |  {ci:.4f} -> {cf:.4f}  |  {wi:.4f} -> {wf:.4f}{mark}", flush=True)
    # localization signal: is there a layer band where WRONG keeps intermediate >> final but CORRECT flips?
    def band(d, key, a0, a1): return float(np.mean(d[key][a0:a1+1]))/max(d["k"],1)
    a0, a1 = L//2, L  # upper half
    c_gap = band(acc["correct"], "fin", a0, a1) - band(acc["correct"], "intt", a0, a1)
    w_gap = band(acc["wrong"], "fin", a0, a1) - band(acc["wrong"], "intt", a0, a1)
    print(f"\n[bind] upper-half (L{a0}-L{a1}) final-minus-intermediate:  CORRECT={c_gap:+.4f}  WRONG={w_gap:+.4f}", flush=True)
    print(f"[bind] {'LOCALIZES (correct binds final, wrong stuck on intermediate) -> patch viable' if (c_gap > w_gap + 0.01) else 'NO clean localization in upper half -> RL fallback'}", flush=True)
    return {"track": track, "c_gap": c_gap, "w_gap": w_gap,
            "correct": acc["correct"]["k"], "wrong": acc["wrong"]["k"]}


# ==================================================================================
# BINDING-LOCALIZATION PROBE v2 (fixes v1's position bug). v1 read logit-lens at the FIRST
# answer token = the option LETTER "(d)", where content tokens sit at ~0 prob -> uninterpretable.
# v2: greedy-generate, find the position that emits the answer CONTENT word (skip letters/stop),
# then logit-lens THERE across all layers for intermediate-attr vs gold-final token, split
# CORRECT/WRONG. Read-only. If correct items show a staged intermediate(mid)->final(late) rise
# while wrong items stall -> localizable binding site for a frozen-base plug-in layer.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_binding_probe2(track: str = "AnimalQA", start: int = 0, n: int = 50):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    QM = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[bind2] loading {QM} track={track}...", flush=True)
    proc = AutoProcessor.from_pretrained(QM)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        QM, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate
    lm = model.language_model; head = lm.lm_head; fnorm = lm.model.norm; tok = proc.tokenizer
    _ST = {"the","a","an","is","it","of","to","and","in","this","that","sound","answer","option","best","match","correct"}

    def first_tok(word):
        ids = tok.encode(" " + word.strip(), add_special_tokens=False); return ids[0] if ids else None

    PROMPT = "\nAnswer with the correct option letter and the answer in a few words."
    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
    L = model.config.text_config.num_hidden_layers
    acc = {"correct": {"intt": np.zeros(L+1), "fin": np.zeros(L+1), "k": 0},
           "wrong":   {"intt": np.zeros(L+1), "fin": np.zeros(L+1), "k": 0}}
    printed = 0
    for ex in ds_sel:
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32)
        if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
        mi, mgold = ex["multi_instruction"], ex["multi_answer"]
        _, attr_text = _gold_letter_text(ex["single_answer"])
        _, fin_text = _gold_letter_text(mgold)
        it_id, fin_id = first_tok(attr_text.split()[0]), first_tok(fin_text.split()[0])
        if it_id is None or fin_id is None: continue
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":mi+PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[a], sampling_rate=SR, return_tensors="pt")
        inputs = {kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inputs.items()}
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        gen_ids = gen[0, plen:]
        pred = proc.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
        ok = mcq_correct(pred, mgold, mi)
        # find the position emitting the first CONTENT word (skip letters/punct/stopwords)
        coff = None
        for j, tid in enumerate(gen_ids.tolist()):
            piece = tok.decode([tid]).strip().lower()
            if len(piece) >= 3 and piece.isalpha() and piece not in _ST:
                coff = j; break
        if coff is None: coff = max(len(gen_ids)-1, 0)
        # forward over prompt+generation, probe the position that PREDICTS the content token
        full_ids = torch.cat([inputs["input_ids"], gen_ids.unsqueeze(0)], dim=1)
        fwd = {"input_ids": full_ids,
               "attention_mask": torch.ones_like(full_ids),
               "input_features": inputs.get("input_features"),
               "feature_attention_mask": inputs.get("feature_attention_mask")}
        fwd = {k:v for k,v in fwd.items() if v is not None}
        with torch.no_grad():
            out = model(**fwd, output_hidden_states=True)
        probe_pos = plen + coff - 1
        hs = out.hidden_states
        bucket = acc["correct"] if ok else acc["wrong"]
        for li in range(L+1):
            h = hs[li][0, probe_pos, :]
            p = torch.softmax(head(fnorm(h)).float(), dim=-1)
            bucket["intt"][li] += p[it_id].item(); bucket["fin"][li] += p[fin_id].item()
        bucket["k"] += 1
        if printed < 4:
            print(f"[bind2 ex] ok={ok} int={attr_text!r} fin={fin_text!r} content_tok={tok.decode([gen_ids[coff].item()])!r} pred={pred[:50]!r}", flush=True)
            printed += 1

    cc, ww = acc["correct"], acc["wrong"]
    print(f"\n[bind2] ===== {track} [{lo}:{hi}] correct={cc['k']} wrong={ww['k']} =====", flush=True)
    print(f"[bind2] layer |  CORRECT int->fin   |   WRONG int->fin", flush=True)
    for li in range(L+1):
        ci=cc['intt'][li]/max(cc['k'],1); cf=cc['fin'][li]/max(cc['k'],1)
        wi=ww['intt'][li]/max(ww['k'],1); wf=ww['fin'][li]/max(ww['k'],1)
        print(f"[bind2] L{li:2d}  |  {ci:.4f} -> {cf:.4f}  |  {wi:.4f} -> {wf:.4f}", flush=True)
    def pk(d,key): 
        arr=d[key]/max(d['k'],1); return int(np.argmax(arr)), float(np.max(arr))
    print(f"\n[bind2] CORRECT final peaks at L{pk(cc,'fin')[0]} (p={pk(cc,'fin')[1]:.3f}); intermediate peaks at L{pk(cc,'intt')[0]} (p={pk(cc,'intt')[1]:.3f})", flush=True)
    # localization: in correct items does final rise above intermediate in upper layers, and is the
    # correct-vs-wrong final gap concentrated in a layer band?
    cf_top=float(np.mean(cc['fin'][L//2:])/max(cc['k'],1)); wf_top=float(np.mean(ww['fin'][L//2:])/max(ww['k'],1))
    print(f"[bind2] upper-half mean GOLD-final prob: CORRECT={cf_top:.4f} WRONG={wf_top:.4f} (gap={cf_top-wf_top:+.4f})", flush=True)
    print(f"[bind2] {'LOCALIZES (final-binding signal separates correct/wrong) -> patch viable' if (cf_top - wf_top > 0.01) else 'still weak separation -> binding diffuse, RL the honest lever'}", flush=True)
    return {"track": track, "cf_top": cf_top, "wf_top": wf_top, "correct": cc['k'], "wrong": ww['k']}


# ==================================================================================
# BINDING-CORRECTION LAYER (frozen-base plug-in; Qwen weights UNTOUCHED). Probe v2 localized the
# multi-hop binding to the L24->L31 band (intermediate peaks ~L24, final ~L32; wrong items never
# develop the final). This is a forward-hook that AMPLIFIES the band's own residual update at the
# answer position: h_out := h_in + lambda*(h_out - h_in) for layers in [lo,hi). Gives the binding
# computation more gain. Honest dev->test lambda-sweep, greedy multi-hop (isolates binding, no SC).
# NOT weight editing, NOT training: a separable novel layer on a frozen model.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_binding_boost(track: str = "AnimalQA", dev_start: int = 350, dev_n: int = 40,
                       test_start: int = 0, test_n: int = 100, lo: int = 24, hi: int = 32):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    QM = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[boost] loading {QM} band L{lo}-L{hi-1} track={track}...", flush=True)
    proc = AutoProcessor.from_pretrained(QM)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        QM, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate
    layers = model.language_model.model.layers
    _B = {"on": False, "lam": 1.0}

    def make_hook():
        def hook(mod, inp, out):
            if not _B["on"]: return out
            h_out = out[0] if isinstance(out, tuple) else out
            h_in = inp[0]
            upd = h_out[:, -1:, :] - h_in[:, -1:, :]
            h_out = h_out.clone()
            h_out[:, -1:, :] = h_in[:, -1:, :] + _B["lam"] * upd
            return (h_out,) + tuple(out[1:]) if isinstance(out, tuple) else h_out
        return hook
    for li in range(lo, min(hi, len(layers))):
        layers[li].register_forward_hook(make_hook())

    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."
    def gen_greedy(a16, mi):
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":mi+PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inputs = {kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def load_items(s, k):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        ds = ds.select(range(s, min(s+k, len(ds))))
        items = []
        for ex in ds:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            items.append((a, ex["multi_instruction"], ex["multi_answer"]))
        return items
    def score(items, lam):
        _B["on"] = (lam != 1.0); _B["lam"] = lam
        c = sum(int(mcq_correct(gen_greedy(a, mi), g, mi)) for a, mi, g in items)
        _B["on"] = False
        return c/len(items)

    dev = load_items(dev_start, dev_n)
    dev_base = score(dev, 1.0); print(f"[boost] DEV base = {dev_base:.3f}", flush=True)
    best_l, best_acc = 1.0, dev_base
    for lam in (1.3, 1.6, 2.0):
        acc = score(dev, lam); print(f"[boost] DEV lambda={lam}: {acc:.3f} (lift {acc-dev_base:+.3f})", flush=True)
        if acc > best_acc: best_acc, best_l = acc, lam
    test = load_items(test_start, test_n)
    tb = score(test, 1.0); ti = score(test, best_l) if best_l != 1.0 else tb
    print(f"\n[boost] ===== TEST [{test_start}:{test_start+test_n}] lambda={best_l} base={tb:.3f} boosted={ti:.3f} lift={ti-tb:+.3f} =====", flush=True)
    print(f"[boost] {'WORKS (binding layer lifts multi-hop, dev+test both positive)' if (best_l!=1.0 and best_acc>dev_base and ti>tb) else 'NULL (binding does not yield to a frozen-hook boost -> RL/adapter the honest lever)'}", flush=True)
    return {"best_lambda": best_l, "dev_base": dev_base, "test_base": tb, "test_boost": ti}


# ==================================================================================
# BINDING-BOOST VERIFICATION (the skeptic's run). The boost showed dev +0.05 / test +0.25 — the
# asymmetry + magnitude demand proof it's real, not slice-luck or a scorer artifact. This run:
# (1) prints base-vs-boosted GENERATIONS side by side (coherent reasoning, or garbage that scores?),
# (2) re-measures base vs boost (fixed lambda) on a FRESH slice, (3) parametrized for a 2nd track.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_binding_verify(track: str = "AnimalQA", start: int = 100, n: int = 100,
                        lam: float = 2.0, lo: int = 24, hi: int = 32, dump: int = 6):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    QM = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[verify] {QM} band L{lo}-L{hi-1} lambda={lam} track={track} [{start}:{start+n}]", flush=True)
    proc = AutoProcessor.from_pretrained(QM)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        QM, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate
    layers = model.language_model.model.layers
    _B = {"on": False, "lam": lam}
    def make_hook():
        def hook(mod, inp, out):
            if not _B["on"]: return out
            h_out = out[0] if isinstance(out, tuple) else out
            h_in = inp[0]
            upd = h_out[:, -1:, :] - h_in[:, -1:, :]
            h_out = h_out.clone(); h_out[:, -1:, :] = h_in[:, -1:, :] + _B["lam"] * upd
            return (h_out,) + tuple(out[1:]) if isinstance(out, tuple) else h_out
        return hook
    for li in range(lo, min(hi, len(layers))):
        layers[li].register_forward_hook(make_hook())

    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."
    def gen(a16, mi, boost):
        _B["on"] = boost
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":mi+PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inputs = {kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        _B["on"] = False
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test").select(range(start, start+n))
    bc = oc = nn = 0
    for ex in ds:
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32)
        if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
        mi, g = ex["multi_instruction"], ex["multi_answer"]
        pb = gen(a, mi, False); pi = gen(a, mi, True)
        ok_b = mcq_correct(pb, g, mi); ok_i = mcq_correct(pi, g, mi)
        bc += int(ok_b); oc += int(ok_i); nn += 1
        if nn <= dump:
            print(f"\n[verify ex{nn}] gold={g!r}", flush=True)
            print(f"   BASE  ok={ok_b}: {pb[:160]!r}", flush=True)
            print(f"   BOOST ok={ok_i}: {pi[:160]!r}", flush=True)
        if nn % 25 == 0: print(f"[verify] {nn}/{n} base={bc/nn:.3f} boost={oc/nn:.3f}", flush=True)
    print(f"\n[verify] ===== {track} [{start}:{start+n}] N={nn} base={bc/nn:.3f} boost={oc/nn:.3f} lift={(oc-bc)/nn:+.3f} =====", flush=True)
    print(f"[verify] {'REPLICATES (boost > base on fresh slice)' if oc>bc else 'DOES NOT replicate (was slice-luck)'}", flush=True)
    return {"track": track, "base": bc/nn, "boost": oc/nn, "n": nn}


# ==================================================================================
# STRUCTURED-CoT SFT (SARI-style, arXiv 2504.15900). SARI's ablation: STRUCTURED chains (Plan/
# Caption/Reasoning/Summary) generalize where UNstructured fail, and SFT-warmup is the RL prereq.
# This is the cheap GATE before GRPO + the novel recurrent-depth adapter: does training STRUCTURED
# reasoning into Qwen2-Audio lift multi-hop binding (vs our earlier UNstructured SFT which hurt SC)?
# Data = text-teacher with gold attribute injected -> grounded, correct, structured traces.
# ==================================================================================
_STRUCT_PROMPT = ("\nAnswer in four labeled sections:\n"
                  "Plan: how you will solve it.\n"
                  "Caption: the key audio attribute.\n"
                  "Reasoning: step-by-step from the attribute to the answer.\n"
                  "Summary: end with 'Answer: (letter) option-text'.")

@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_structured_data(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 200, kchains: int = 4):
    """Text-teacher (gold injected) -> grounded STRUCTURED (Plan/Caption/Reasoning/Summary) traces."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import json, os, re as _re, torch
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    def _summary(c):
        m = _re.search(r"Summary\s*:?\s*(.+)", c, _re.DOTALL | _re.IGNORECASE)
        return m.group(1).strip() if m else c

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[struct] loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    tok = processor.tokenizer

    def gen_text(prompt, k):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=320, do_sample=(k > 1), temperature=0.7,
                                 top_p=0.95, num_return_sequences=k)
        return [d.strip() for d in tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

    os.makedirs("/data/distill", exist_ok=True)
    tot = 0
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start + n, len(ds)); ds_sel = ds.select(range(lo, hi))
        attr_name = _ATTR_NAME[track]
        out = f"/data/distill/{track}_{lo}_{hi}_structured.jsonl"
        kept = nn = 0
        with open(out, "w") as fo:
            for i_in, ex in enumerate(ds_sel):
                _, attr = _gold_letter_text(ex["single_answer"])
                mi, gold = ex["multi_instruction"], ex["multi_answer"]
                student = mi + _STRUCT_PROMPT
                teacher = (f"The {attr_name} is '{attr}'. Answer the question in EXACTLY four labeled sections.\n"
                           f"Plan: state you will use the {attr_name} to answer.\n"
                           f"Caption: the {attr_name} is {attr}.\n"
                           f"Reasoning: connect '{attr}' to the correct option in 1-2 steps.\n"
                           f"Summary: end with 'Answer: (letter) option-text'.\n\nQuestion: {mi}")
                chains = gen_text(teacher, kchains)
                good = [c for c in chains if "Summary" in c and _word_in(c, attr) and mcq_correct(_summary(c), gold, mi)]
                nn += 1
                if good:
                    good.sort(key=len); kept += 1
                    fo.write(json.dumps({"track": track, "ds_index": lo + i_in,
                                         "instruction": student, "trace": good[0], "gold": gold}) + "\n")
                if nn % 40 == 0: print(f"[struct] {track} {nn}/{len(ds_sel)} kept={kept}", flush=True)
        tot += kept
        print(f"[struct] {track} kept={kept}/{nn} -> {out}", flush=True)
    vol.commit()
    print(f"\n[struct] TOTAL structured traces = {tot}", flush=True)
    return {"kept": tot}


@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_structured_eval(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 100,
                         adapter: str = "/data/distill/lora_struct", kchains: int = 5):
    """Base vs structured-SFT vs structured-SFT+SC on held-out, using the STRUCTURED prompt (matches training)."""
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    sft = PeftModel.from_pretrained(
        Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda"),
        adapter).eval()
    SR = processor.feature_extractor.sampling_rate

    def gen(model, a16, mi, k):
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":mi+_STRUCT_PROMPT}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inputs = {kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=320, do_sample=(k>1), temperature=0.7,
                                 top_p=0.9, num_return_sequences=k)
        chains = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if k == 1: return chains[0]
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res = {}
    for track in tracks.split(","):
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start+n, len(ds)); ds_sel = ds.select(range(lo, hi))
        bc = sc = cc = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            mi, gold = ex["multi_instruction"], ex["multi_answer"]
            bc += int(mcq_correct(gen(base, a, mi, 1), gold, mi))
            sc += int(mcq_correct(gen(sft, a, mi, 1), gold, mi))
            cc += int(mcq_correct(gen(sft, a, mi, kchains), gold, mi))
            nn += 1
        res[track] = {"base": bc/nn, "sft": sc/nn, "sft_sc": cc/nn}
        print(f"[seval] {track} [{lo}:{hi}] base={bc/nn:.3f} sSFT={sc/nn:.3f} sSFT+SC={cc/nn:.3f}", flush=True)
    avgb = sum(r["base"] for r in res.values())/len(res)
    avgs = sum(r["sft"] for r in res.values())/len(res)
    avgc = sum(r["sft_sc"] for r in res.values())/len(res)
    print(f"\n[seval] ===== HELD-OUT [{start}:{start+n}] base={avgb:.3f} structSFT={avgs:.3f} structSFT+SC={avgc:.3f} (base+SC ref=0.600) =====", flush=True)
    print(f"[seval] {'STRUCTURED SFT HELPS (sSFT > base greedy)' if avgs>avgb+0.02 else 'structured SFT does not beat base greedy'}", flush=True)
    return {"per_track": res, "base": avgb, "sft": avgs, "sft_sc": avgc}


# ==================================================================================
# DECOMPOSED + RE-GROUNDED SELF-CONSISTENCY (training-free root-cause fix). Root cause (ours +
# SAKURA + AGR 2509.16971 + MPAR 2603.02266): the audio attribute (a) stays LATENT (not in the
# text reasoning stream) and (b) DECAYS over the chain (drifts to language priors). Fix, no training:
#   Stage 1 EXTRACT (audio): ask the EXACT single-hop question -> attribute as TEXT (model is strong here).
#   Stage 2 RE-GROUND+COMPOSE: re-present the AUDIO + inject the extracted attribute as text -> reason -> SC.
# Externalize (AGR) + re-ground (MPAR) + self-consistency (ours). Compare to base+SC=0.600.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_decomp_sc(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 100, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[decomp] loading {MODEL} (K={kchains})...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate

    def gen(a16, instr, k, greedy):
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inputs = {kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inputs.items()}
        with torch.no_grad():
            if greedy:
                out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                     top_p=0.9, num_return_sequences=k)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    res = {}
    for track in tracks.split(","):
        attr_name = _ATTR_NAME[track]
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start+n, len(ds)); ds_sel = ds.select(range(lo, hi))
        c = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            # STAGE 1: extract the attribute as TEXT via the exact single-hop question
            s_pred = gen(a, si, 1, True)[0]
            s_opts = _parse_options(si); s_letter = pick_option(s_pred, si)
            extracted = s_opts.get(s_letter, "") if s_letter else ""
            if not extracted:  # fallback to raw text
                extracted = s_pred.strip().split("\n")[0][:40]
            # STAGE 2: re-ground (audio re-presented) + inject extracted attribute as text -> reason -> SC
            compose = (mi + f"\n[Audio analysis: the {attr_name} is {extracted}. Use BOTH this and the audio.]"
                          "\nReason step by step from the stated attribute to the answer, then give the final answer.")
            chains = gen(a, compose, kchains, False)
            picks = [p for p in (pick_option(ch, mi) for ch in chains) if p]
            pred = f"({Counter(picks).most_common(1)[0][0]})" if picks else ""
            c += int(mcq_correct(pred, gold, mi)); nn += 1
        res[track] = c/nn
        print(f"[decomp] {track} [{lo}:{hi}] decomp+SC={c/nn:.3f}", flush=True)
    avg = sum(res.values())/len(res)
    print(f"\n[decomp] ===== HELD-OUT [{start}:{start+n}] decomp+SC={avg:.3f} (base+SC ref=0.600) =====", flush=True)
    hi3 = sum(res[t] for t in res if t!="EmotionQA")/max(len([t for t in res if t!="EmotionQA"]),1)
    print(f"[decomp] perception-solvable 3-track avg (excl Emotion) = {hi3:.3f}", flush=True)
    return {"per_track": res, "avg": avg, "hi3": hi3}


# ==================================================================================
# DECOMP v2: MULTI-STEP RE-GROUNDING (MPAR-style, training-free). v1 re-grounds ONCE (extract attr
# -> compose). v2 keeps re-attending the audio across the reasoning: (1) extract attribute, (2) a
# SECOND audio-grounded check that asks the model to re-listen and confirm/refine the attribute,
# (3) compose with the confirmed attribute + audio re-presented -> SC. Mitigates "perception decay"
# (attribute drifting to priors over the chain). verify=True also re-runs single-step on a fresh
# slice for replication. Fully training-free.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_decomp_v2(tracks: str = ",".join(_TRACKS), start: int = 100, n: int = 100, kchains: int = 5,
                   multistep: int = 1):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[decomp2] loading {MODEL} (K={kchains}, multistep={multistep})...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate

    def gen(a16, instr, k, greedy, max_new=300):
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inputs = {kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inputs.items()}
        with torch.no_grad():
            if greedy:
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                                     top_p=0.9, num_return_sequences=k)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    res = {}
    for track in tracks.split(","):
        attr_name = _ATTR_NAME[track]
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start+n, len(ds)); ds_sel = ds.select(range(lo, hi))
        c = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            s_opts = _parse_options(si)
            # STEP 1: extract attribute (audio)
            s_pred = gen(a, si, 1, True, max_new=80)[0]
            sL = pick_option(s_pred, si); extracted = s_opts.get(sL, "") if sL else ""
            # STEP 2 (multistep>=2): RE-LISTEN and confirm/refine the attribute (re-ground)
            if multistep >= 2:
                recheck = (si + f"\n[A first pass suggested {attr_name}={extracted or 'unclear'}. "
                                "Re-listen to the audio and give your FINAL choice.]")
                r_pred = gen(a, recheck, 1, True, max_new=80)[0]
                rL = pick_option(r_pred, si)
                if rL and s_opts.get(rL): extracted = s_opts[rL]
            if not extracted: extracted = s_pred.strip().split("\n")[0][:40]
            # STEP 3: compose with confirmed attribute + audio re-presented -> SC
            compose = (mi + f"\n[Audio analysis: the {attr_name} is {extracted}. Use BOTH this and the audio.]"
                          "\nReason step by step from the stated attribute to the answer, then give the final answer.")
            chains = gen(a, compose, kchains, False)
            picks = [p for p in (pick_option(ch, mi) for ch in chains) if p]
            pred = f"({Counter(picks).most_common(1)[0][0]})" if picks else ""
            c += int(mcq_correct(pred, gold, mi)); nn += 1
        res[track] = c/nn
        print(f"[decomp2] {track} [{lo}:{hi}] ms={multistep} acc={c/nn:.3f}", flush=True)
    avg = sum(res.values())/len(res)
    hi3 = sum(res[t] for t in res if t!="EmotionQA")/max(len([t for t in res if t!="EmotionQA"]),1)
    print(f"\n[decomp2] ===== [{start}:{start+n}] ms={multistep} avg={avg:.3f} 3-track(excl Emo)={hi3:.3f} =====", flush=True)
    return {"per_track": res, "avg": avg, "hi3": hi3}


# ==================================================================================
# DECOUPLED CASCADE (specialist perceivers -> Qwen TEXT-ONLY reasoning + SC). Honest framing: NOT
# our novel method but the DECOUPLED UPPER BOUND / control that proves the gap is audio-integration.
# Each specialist sees ONLY audio (no gold). Reasoning runs in Qwen's TEXT-ONLY mode (no audio token)
# -> the SAME model's reasoning, but perception externalized + audio out of the reasoning path.
# Contrast: end-to-end 0.60, tool-aug-into-audio 0.57, cascade(text-only) -> should jump if gap=integration.
# Specialists: gender wav2vec2, MMS-LID (language), SER wav2vec2 (emotion), AST/AudioSet (animal).
# ==================================================================================
_CASCADE_SPEC = {
    "GenderQA":   ("alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech", "cls"),
    "LanguageQA": ("facebook/mms-lid-256", "lid"),
    "EmotionQA":  ("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", "cls"),
    "AnimalQA":   ("MIT/ast-finetuned-audioset-10-10-0.4593", "tag"),
}

@app.function(image=qwen_tool_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_cascade(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 100, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, re as _re
    from collections import Counter
    from datasets import load_dataset
    from transformers import (Qwen2AudioForConditionalGeneration, AutoProcessor,
                              AutoFeatureExtractor, AutoModelForAudioClassification)

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[cascade] loading {MODEL} (text reasoner) + specialists...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
    qwen = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def iso_to_name(code):
        try:
            import langcodes; return langcodes.Language.get(code).display_name()
        except Exception: return code
    def toks(s): return [w for w in _re.findall(r"[a-z]+", s.lower()) if len(w) > 2]
    def match_option(labels, opts):
        for lab in labels:
            lw = toks(lab)
            for L, t in opts.items():
                if any(w in toks(t) for w in lw) or any(w in lw for w in toks(t)):
                    return L, t
        return None, (labels[0] if labels else "")

    # text-only reasoning (NO audio) -> SC
    def text_reason_sc(attr_name, extracted, mi, k):
        prompt = (f"[Audio analysis: the {attr_name} is {extracted}.]\n{mi}\n"
                  f"Using the stated {attr_name}, reason step by step, then give the final answer.")
        text = tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = qwen.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7,
                                top_p=0.9, num_return_sequences=k)
        chains = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res = {}; spec_acc = {}
    for track in tracks.split(","):
        spec_id, kind = _CASCADE_SPEC[track]; attr_name = _ATTR_NAME[track]
        print(f"[cascade] {track}: specialist={spec_id} ({kind})", flush=True)
        sfe = AutoFeatureExtractor.from_pretrained(spec_id)
        smodel = AutoModelForAudioClassification.from_pretrained(spec_id).to("cuda").eval()
        SSR = sfe.sampling_rate; id2label = smodel.config.id2label
        def specialist_labels(a16):
            inp = sfe(a16, sampling_rate=SSR, return_tensors="pt"); inp = {k:v.to("cuda") for k,v in inp.items()}
            with torch.no_grad(): logits = smodel(**inp).logits[0]
            ids = logits.topk(min(5, logits.shape[-1])).indices.tolist()
            return [iso_to_name(id2label[i]) if kind=="lid" else str(id2label[i]) for i in ids]
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start+n, len(ds)); ds_sel = ds.select(range(lo, hi))
        c = sc = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            a_spec = librosa.resample(a, orig_sr=sr, target_sr=SSR) if sr != SSR else a
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            labels = specialist_labels(a_spec)
            sL, sText = match_option(labels, _parse_options(si))
            sgl, _ = _gold_letter_text(ex["single_answer"]); sc += int(sL == sgl)
            extracted = sText or (labels[0] if labels else "unknown")
            pred = text_reason_sc(attr_name, extracted, mi, kchains)
            c += int(mcq_correct(pred, gold, mi)); nn += 1
        res[track] = c/nn; spec_acc[track] = sc/nn
        print(f"[cascade] {track} [{lo}:{hi}] specialist_single={sc/nn:.3f} cascade_multi={c/nn:.3f}", flush=True)
        del smodel; torch.cuda.empty_cache()
    avg = sum(res.values())/len(res)
    hi3 = sum(res[t] for t in res if t!="EmotionQA")/max(len([t for t in res if t!="EmotionQA"]),1)
    print(f"\n[cascade] ===== [{start}:{start+n}] cascade avg={avg:.3f} 3-track(excl Emo)={hi3:.3f} =====", flush=True)
    print(f"[cascade] specialist single-hop per track: {spec_acc}", flush=True)
    print(f"[cascade] vs end-to-end base+SC 0.60 / tool-aug-into-audio 0.57 / decomp 0.70", flush=True)
    return {"per_track": res, "specialist": spec_acc, "avg": avg, "hi3": hi3}


# ==================================================================================
# AF-NEXT DECOMP (cross-model check for the training-free fix -> "model-agnostic" claim).
# Ports the decomp pipeline to AF-Next's API. Per item computes BOTH base+SC and decomp+SC in
# ONE model load (efficient). Same [0:100] slice as Qwen (0.63 base+SC -> 0.71 decomp+SC) so the
# cross-model comparison is apples-to-apples. Decomp = extract attr via single-hop (greedy) ->
# re-ground (audio re-presented) + inject attr as text -> reason -> SC.
# ==================================================================================
@app.function(image=afnext_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=21600)
def afnext_decomp(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 100, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, soundfile as sf, tempfile
    from collections import Counter
    import transformers
    from transformers import AutoProcessor, AutoConfig
    from datasets import load_dataset

    MODEL = "nvidia/audio-flamingo-next-hf"
    print(f"[afx-decomp] loading {MODEL} (K={kchains})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    arch = AutoConfig.from_pretrained(MODEL).architectures[0]
    model = getattr(transformers, arch).from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    SR = processor.feature_extractor.sampling_rate
    BASE_PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."

    def gen(path, instr, k, greedy, max_new):
        conv = [{"role":"user","content":[{"type":"text","text":instr},{"type":"audio","path":path}]}]
        inputs = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True)
        inputs = {kk:(v.to(model.device) if hasattr(v,"to") else v) for kk,v in inputs.items()}
        with torch.no_grad():
            if greedy:
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                                     top_p=0.9, num_return_sequences=k)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def vote(chains, mi):
        picks = [p for p in (pick_option(c, mi) for c in chains) if p]
        return f"({Counter(picks).most_common(1)[0][0]})" if picks else ""

    res_base = {}; res_dec = {}
    for track in tracks.split(","):
        attr_name = _ATTR_NAME[track]
        ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo, hi = start, min(start+n, len(ds)); ds_sel = ds.select(range(lo, hi))
        cb = cd = nn = 0
        for ex in ds_sel:
            a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            if isinstance(a, list): a = np.array(a, dtype=np.float32)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, a, SR); path = f.name
            si, mi, gold = ex["single_instruction"], ex["multi_instruction"], ex["multi_answer"]
            # base+SC
            cb += int(mcq_correct(vote(gen(path, mi + BASE_PROMPT, kchains, False, 300), mi), gold, mi))
            # decomp+SC: extract (greedy) -> re-ground+inject -> SC
            s_pred = gen(path, si, 1, True, 80)[0]
            sL = pick_option(s_pred, si); extracted = _parse_options(si).get(sL, "") if sL else ""
            if not extracted: extracted = s_pred.strip().split("\n")[0][:40]
            compose = (mi + f"\n[Audio analysis: the {attr_name} is {extracted}. Use BOTH this and the audio.]"
                          "\nReason step by step from the stated attribute to the answer, then give the final answer.")
            cd += int(mcq_correct(vote(gen(path, compose, kchains, False, 300), mi), gold, mi))
            nn += 1; _os.unlink(path)
            if nn % 25 == 0: print(f"[afx-decomp] {track} {nn}/{len(ds_sel)} base+SC={cb/nn:.3f} decomp+SC={cd/nn:.3f}", flush=True)
        res_base[track] = cb/nn; res_dec[track] = cd/nn
        print(f"[afx-decomp] {track} [{lo}:{hi}] base+SC={cb/nn:.3f} decomp+SC={cd/nn:.3f} lift={cd/nn-cb/nn:+.3f}", flush=True)
    ab = sum(res_base.values())/len(res_base); ad = sum(res_dec.values())/len(res_dec)
    print(f"\n[afx-decomp] ===== AF-Next [{start}:{start+n}] base+SC={ab:.3f} decomp+SC={ad:.3f} lift={ad-ab:+.3f} =====", flush=True)
    print(f"[afx-decomp] Qwen ref: base+SC 0.63 -> decomp+SC 0.71 (+0.08). Model-agnostic if AF-Next lift>0.", flush=True)
    return {"base_sc": res_base, "decomp_sc": res_dec, "avg_base": ab, "avg_decomp": ad}


# ==================================================================================
# BINDING-AWARE GRPO (smoke test). Custom multimodal GRPO loop (TRL has no audio-rollout support).
# Reward = correctness (1.0) + BINDING/grounding bonus (0.3 if the trace states the gold attribute)
# -> directly incentivizes the "bind attribute -> chain" behavior we localized. Group-relative
# advantage over K rollouts; policy-gradient on LoRA only. Goal of SMOKE TEST: does GREEDY multi-hop
# improve (reward trending up, held-out greedy acc rising)? De-risking gate, NOT a publishable number.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_grpo_smoke(track: str = "LanguageQA", train_start: int = 100, train_n: int = 80,
                    steps: int = 60, kroll: int = 6, lr: float = 1e-5, eval_n: int = 40):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[grpo] loading {MODEL} track={track} steps={steps} K={kroll}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    SR = proc.feature_extractor.sampling_rate
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model = get_peft_model(model, lcfg); model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."

    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    def load_audio(ex):
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32)
        return librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a
    train = [ds[i] for i in range(train_start, min(train_start+train_n, len(ds)))]

    def build_inputs(a16, instr):
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        return {k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}

    @torch.no_grad()
    def eval_greedy(slice_ex):
        model.eval(); c = 0
        for ex in slice_ex:
            a = load_audio(ex); inp = build_inputs(a, ex["multi_instruction"]+PROMPT)
            out = model.generate(**inp, max_new_tokens=250, do_sample=False, use_cache=True)
            pred = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            c += int(mcq_correct(pred, ex["multi_answer"], ex["multi_instruction"]))
        return c/len(slice_ex)

    eval_slice = [ds[i] for i in range(0, eval_n)]
    print(f"[grpo] EVAL greedy multi-hop BEFORE = {eval_greedy(eval_slice):.3f}", flush=True)

    rng = np.random.default_rng(0)
    for step in range(steps):
        ex = train[int(rng.integers(len(train)))]
        a = load_audio(ex); mi, gold = ex["multi_instruction"], ex["multi_answer"]
        _, attr = _gold_letter_text(ex["single_answer"])
        inp = build_inputs(a, mi + PROMPT); plen = inp["input_ids"].shape[1]
        model.eval()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=200, do_sample=True, temperature=1.0,
                                 top_p=0.95, num_return_sequences=kroll, use_cache=True)
        rollouts = gen[:, plen:]
        chains = proc.batch_decode(rollouts, skip_special_tokens=True)
        rewards = np.array([(1.0 if mcq_correct(c, gold, mi) else 0.0) + (0.3 if _word_in(c, attr) else 0.0)
                            for c in chains])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        model.train(); opt.zero_grad(); tot = 0.0
        feats = {k:v for k,v in inp.items() if k in ("input_features","feature_attention_mask")}
        for i in range(kroll):
            if abs(adv[i]) < 1e-3: continue
            full = torch.cat([inp["input_ids"], rollouts[i:i+1]], dim=1)
            am = torch.ones_like(full)
            out = model(input_ids=full, attention_mask=am, use_cache=False, **feats)
            logits = out.logits[0, plen-1:-1, :]                       # predict rollout tokens
            lp = torch.log_softmax(logits.float(), dim=-1)
            tok = rollouts[i]; tlp = lp[torch.arange(len(tok)), tok].mean()
            loss = -float(adv[i]) * tlp / kroll
            loss.backward(); tot += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 5 == 0:
            print(f"[grpo] step {step} reward_mean={rewards.mean():.3f} correct_frac={(rewards>=1.0).mean():.2f} loss={tot:.4f}", flush=True)
        if step > 0 and step % 20 == 0:
            print(f"[grpo] >>> EVAL greedy @ step{step} = {eval_greedy(eval_slice):.3f}", flush=True)
    final = eval_greedy(eval_slice)
    print(f"\n[grpo] ===== {track} EVAL greedy AFTER {steps} steps = {final:.3f} =====", flush=True)
    print(f"[grpo] {'PROMISING (greedy improved -> binding-RL direction alive)' if final > 0 else 'see trend'}", flush=True)
    return {"track": track, "final_greedy": final}


# ==================================================================================
# MMAR inspect (CPU, cheap) — print the dataset schema + a sample so we build the eval correctly.
# ==================================================================================
@app.function(image=qwen_img, volumes={"/data": vol}, timeout=3600)
def mmar_inspect():
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from datasets import load_dataset, get_dataset_config_names
    for name in ["BoJack/MMAR"]:
        try:
            print(f"=== configs for {name}: {get_dataset_config_names(name)}", flush=True)
        except Exception as e:
            print(f"configs err: {e}", flush=True)
        for split in ["test", "train", "validation"]:
            try:
                ds = load_dataset(name, split=split)
                print(f"\n=== {name} split={split} n={len(ds)} ===", flush=True)
                print(f"FEATURES: {ds.features}", flush=True)
                ex = ds[0]
                for k, v in ex.items():
                    sv = str(v)
                    if k == "audio" and isinstance(v, dict):
                        sv = f"dict keys={list(v.keys())} sr={v.get('sampling_rate')}"
                    print(f"  [{k}] = {sv[:200]}", flush=True)
                print(f"\nITEM 1 (raw non-audio):", flush=True)
                print({k: (str(v)[:150] if k != 'audio' else 'AUDIO') for k, v in ds[1].items()}, flush=True)
                return {"n": len(ds), "features": str(ds.features)}
            except Exception as e:
                print(f"split {split} err: {str(e)[:150]}", flush=True)
    return {}


# ==================================================================================
# MMAR base vs base+SC (generalization appendix). Does self-consistency generalize to a DIFFERENT
# deep-reasoning audio benchmark (MMAR, NeurIPS'25, 1000 MCQ)? Qwen2-Audio. Audio files downloaded
# from HF repo (audio_path is a relative ref). Reports base(greedy) and base+SC per item -> is the
# SC lift model/benchmark-general? (decomp NOT tested here — it's SAKURA-attribute-shaped.)
# ==================================================================================
@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def mmar_base_sc(start: int = 0, n: int = 300, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, soundfile as sf, ast as _ast, os, tarfile
    from collections import Counter
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    print("[mmar] downloading + extracting audio tarball (one-time)...", flush=True)
    tar_path = hf_hub_download(repo_id="BoJack/MMAR", repo_type="dataset", filename="mmar-audio.tar.gz")
    extract_dir = "/data/mmar_audio"; marker = extract_dir + "/.done"
    if not os.path.exists(marker):
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path) as t: t.extractall(extract_dir)
        open(marker, "w").close(); vol.commit()
    idx = {}
    for root, _d, fs in os.walk(extract_dir):
        for f in fs:
            if f.endswith(".wav"): idx[f] = os.path.join(root, f)
    ds = load_dataset("BoJack/MMAR", split="test")
    print(f"[mmar] extracted {len(idx)} wavs; dataset n={len(ds)}", flush=True)

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate
    LET = "abcdefgh"

    def build(question, choices):
        opts = " ".join(f"({LET[i]}) {c}" for i, c in enumerate(choices))
        instr = f"{question}\n{opts}\nReason step by step, then give the final answer as the option letter."
        return instr, opts

    def gen(a16, instr, k, greedy):
        conv = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inp = {k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}
        with torch.no_grad():
            if greedy: out = model.generate(**inp, max_new_tokens=250, do_sample=False)
            else: out = model.generate(**inp, max_new_tokens=250, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=k)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    lo, hi = start, min(start+n, len(ds)); sel = ds.select(range(lo, hi))
    cb = cs = nn = skip = 0
    for ex in sel:
        try:
            ch = ex["choices"]
            if isinstance(ch, str): ch = _ast.literal_eval(ch)
            ans = ex["answer"]
            gi = next((i for i, c in enumerate(ch) if c.strip().lower() == ans.strip().lower()), None)
            if gi is None: skip += 1; continue
            path = idx.get(os.path.basename(ex["audio_path"]))
            if not path or not os.path.exists(path): skip += 1; continue
            a, sr = sf.read(path)
            if a.ndim > 1: a = a.mean(axis=1)
            a = a.astype(np.float32)
            if sr != SR: a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            a = a[:SR*30]  # cap 30s (Qwen2-Audio limit)
            instr, _ = build(ex["question"], ch)
            gold = f"({LET[gi]}) {ans}"
            # base greedy
            cb += int(mcq_correct(gen(a, instr, 1, True)[0], gold, instr))
            # base+SC
            chains = gen(a, instr, kchains, False)
            picks = [p for p in (pick_option(c, instr) for c in chains) if p]
            pred = f"({Counter(picks).most_common(1)[0][0]})" if picks else ""
            cs += int(mcq_correct(pred, gold, instr))
            nn += 1
            if nn % 25 == 0: print(f"[mmar] {nn}/{hi-lo} base={cb/nn:.3f} base+SC={cs/nn:.3f} (skip={skip})", flush=True)
        except Exception as e:
            skip += 1
            if skip <= 3: print(f"[mmar] skip: {str(e)[:100]}", flush=True)
    print(f"\n[mmar] ===== MMAR [{lo}:{hi}] N={nn} base={cb/max(nn,1):.3f} base+SC={cs/max(nn,1):.3f} lift={(cs-cb)/max(nn,1):+.3f} (skip={skip}) =====", flush=True)
    print(f"[mmar] SC generalizes to MMAR if lift>0 (SAKURA base->base+SC was +0.11)", flush=True)
    return {"base": cb/max(nn,1), "base_sc": cs/max(nn,1), "n": nn, "skip": skip}


# ==================================================================================
# VISION inspect (CPU, cheap) — find a compositional VQA dataset with embedded images + hop structure
# for the Qwen2-VL binding gate (does the audio binding deficit unify to vision, LLM-side?).
# ==================================================================================
@app.function(image=qwen_img, volumes={"/data": vol}, timeout=3600)
def vision_inspect():
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from datasets import load_dataset
    cands = [("lmms-lab/GQA", "testdev_balanced_instructions"),
             ("lmms-lab/GQA", "test_balanced"),
             ("HuggingFaceM4/CLEVR", None),
             ("clevr", None),
             ("dali-does/clevr-math", None)]
    for name, cfg in cands:
        try:
            ds = load_dataset(name, cfg, split="train") if cfg else load_dataset(name, split="train")
            print(f"\n=== {name} cfg={cfg} n={len(ds)} ===", flush=True)
            print(f"FEATURES: {list(ds.features.keys())}", flush=True)
            ex = ds[0]
            for k, v in ex.items():
                sv = "IMG" if "image" in k.lower() and not isinstance(v, str) else str(v)[:160]
                print(f"  [{k}] = {sv}", flush=True)
        except Exception as e:
            print(f"[{name} cfg={cfg}] ERR: {str(e)[:130]}", flush=True)
    return {}


@app.function(image=qwen_img, volumes={"/data": vol}, timeout=3600)
def gqa_inspect():
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from datasets import load_dataset, get_dataset_config_names
    print("CONFIGS:", get_dataset_config_names("lmms-lab/GQA"), flush=True)
    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    print(f"\nINSTRUCTIONS n={len(ds)} features={list(ds.features.keys())}", flush=True)
    for k, v in ds[0].items():
        print(f"  [{k}] = {('IMG' if 'image' in k.lower() and not isinstance(v,str) else str(v)[:160])}", flush=True)
    # question types present?
    import collections
    if "types" in ds.features or "detailed_type" in ds.features:
        pass
    print("\nsample 5 questions/answers:", flush=True)
    for i in range(5):
        e = ds[i]
        print(f"  Q={e.get('question','')[:80]!r} A={e.get('answer','')!r} type={e.get('types', e.get('detailed_type',''))}", flush=True)
    return {}


# ==================================================================================
# VISION BINDING GATE (Qwen2-VL on GQA) — does the audio binding pattern UNIFY to vision?
# GQA 'semantic' field = functional program -> clean single-hop(perceive) vs multi-hop(chain) split
# by program depth. Gate: is single-hop acc >> multi-hop acc (the perceive-vs-chain gap, LLM-side
# elicitation like audio) OR are both low (perception-limited, encoder-side -> vision does NOT unify)?
# Qwen2-VL-7B = same LLM family as Qwen2-Audio (clean cross-modal parallel).
# ==================================================================================
qwen_vl_img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "transformers==4.49.0", "accelerate",
                 "datasets==2.21.0", "qwen-vl-utils", "pillow", "huggingface_hub", "hf_transfer")
    .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

@app.function(image=qwen_vl_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def vl_binding_gate(n_single: int = 100, n_multi: int = 150):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import torch, re
    from datasets import load_dataset
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print("[vl] loading GQA instructions + images...", flush=True)
    ins = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    key_col = "id" if "id" in imgs.features else "imageId"
    id2img = {v: i for i, v in enumerate(imgs[key_col])}   # column access = no image decode
    print(f"[vl] instructions={len(ins)} images={len(imgs)} keyed={len(id2img)} keycol={key_col}", flush=True)

    MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def hops(ex):
        s = ex.get("semantic") or []
        return len(s)
    def get_img(imageId):
        j = id2img.get(imageId)
        return imgs[j]["image"] if j is not None else None
    def ask(image, question):
        conv = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question+"\nAnswer in one or two words."}]}]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=20, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    def correct(pred, ans):
        p = re.sub(r"[^a-z0-9 ]","",pred.lower()); a = ans.lower().strip()
        return a in p.split() or a in p

    # split by program depth
    singles = [e for e in ins if hops(e) <= 2]
    multis  = [e for e in ins if hops(e) >= 4]
    print(f"[vl] pool: single(<=2 ops)={len(singles)} multi(>=4 ops)={len(multis)}", flush=True)

    def run(pool, k, tag):
        c = nn = 0
        for e in pool:
            if nn >= k: break
            img = get_img(e["imageId"])
            if img is None: continue
            ok = correct(ask(img, e["question"]), e["answer"])
            c += int(ok); nn += 1
            if nn % 25 == 0: print(f"[vl] {tag} {nn}/{k} acc={c/nn:.3f}", flush=True)
        return c/max(nn,1), nn

    s_acc, s_n = run(singles, n_single, "SINGLE")
    m_acc, m_n = run(multis, n_multi, "MULTI")
    print(f"\n[vl] ===== Qwen2-VL GQA: single-hop(perceive)={s_acc:.3f} (n={s_n}) | multi-hop(chain)={m_acc:.3f} (n={m_n}) | gap={s_acc-m_acc:+.3f} =====", flush=True)
    print(f"[vl] {'PERCEIVE-VS-CHAIN GAP EXISTS (like audio -> vision may unify, worth logit-lens)' if s_acc-m_acc>0.10 else 'NO clear gap (multi ~ single) -> vision does NOT show our pattern; ship audio alone'}", flush=True)
    return {"single": s_acc, "multi": m_acc, "gap": s_acc-m_acc}


# ==================================================================================
# VISION SC + DECOMP (the fix AND the diagnostic). On GQA multi-hop (>=4-op programs): base+SC vs
# decomp+SC. decomp = externalize the intermediate referent then answer + SC. If decomp HELPS vision
# multi-hop (like audio) -> LLM-side/elicitation failure -> UNIFIES with audio (big cross-modal paper).
# If decomp does NOT help -> encoder-side -> vision does not unify. Qwen2-VL-7B.
# ==================================================================================
@app.function(image=qwen_vl_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def vl_decomp(n_multi: int = 150, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import torch, re
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print("[vld] loading GQA + Qwen2-VL...", flush=True)
    ins = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    key_col = "id" if "id" in imgs.features else "imageId"
    id2img = {v: i for i, v in enumerate(imgs[key_col])}
    MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def get_img(iid):
        j = id2img.get(iid); return imgs[j]["image"] if j is not None else None
    def gen(image, prompt, k, greedy, mx=40):
        conv = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            if greedy: out = model.generate(**inp, max_new_tokens=mx, do_sample=False)
            else: out = model.generate(**inp, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=k)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
    def correct(pred, ans): a=ans.lower().strip(); p=norm(pred); return a in p.split() or a in p
    def vote(cands):
        keys=[norm(c).split()[0] if norm(c) else "" for c in cands]; keys=[k for k in keys if k]
        return Counter(keys).most_common(1)[0][0] if keys else ""

    def hard_multi(e):
        ans = (e["answer"] or "").lower().strip()
        if ans in ("yes", "no"): return False          # exclude guessable yes/no (gate flaw #3)
        t = e.get("types") or {}
        return len(e.get("semantic") or []) >= 3 and t.get("structural") in ("query", "choose", "compare")
    multis = [e for e in ins if hard_multi(e)]
    print(f"[vld] multi pool (non-yes/no, compositional query)={len(multis)}", flush=True)
    cb = cd = nn = 0
    for e in multis:
        if nn >= n_multi: break
        img = get_img(e["imageId"])
        if img is None: continue
        q, ans = e["question"], e["answer"]
        # base+SC
        base_pick = vote(gen(img, q+"\nAnswer in one or two words.", kchains, False))
        cb += int(correct(base_pick, ans))
        # decomp+SC: externalize intermediate referent, then answer
        inter = gen(img, f"To answer '{q}', first identify the specific object the question refers to and describe it in a few words.", 1, True, mx=40)[0]
        dprompt = f"{q}\n[Focus: {inter.strip()[:80]}]\nUsing that, answer in one or two words."
        dec_pick = vote(gen(img, dprompt, kchains, False))
        cd += int(correct(dec_pick, ans)); nn += 1
        if nn % 25 == 0: print(f"[vld] {nn}/{n_multi} base+SC={cb/nn:.3f} decomp+SC={cd/nn:.3f}", flush=True)
    print(f"\n[vld] ===== Qwen2-VL GQA multi-hop [n={nn}] base+SC={cb/max(nn,1):.3f} decomp+SC={cd/max(nn,1):.3f} lift={(cd-cb)/max(nn,1):+.3f} =====", flush=True)
    print(f"[vld] {'DECOMP HELPS VISION -> LLM-side/elicitation -> UNIFIES with audio' if (cd-cb)/max(nn,1)>0.03 else 'decomp does NOT help vision -> likely encoder-side -> does NOT unify'}", flush=True)
    return {"base_sc": cb/max(nn,1), "decomp_sc": cd/max(nn,1), "n": nn}


# ==================================================================================
# BINDING-AWARE GRPO v2 — KL-STABILIZED (the proper version; the smoke test lacked KL -> it
# oscillated/collapsed). Adds: reference-policy KL penalty (via PEFT disable_adapter, no 2nd model),
# k3 KL estimator, beta anchor. Reward = correctness + binding/grounding bonus (states gold attribute).
# UNTESTED (built with no budget to validate) — needs ~$100+ to run: ~300-500 steps, K=8, eval greedy.
# Goal: make GREEDY multi-hop strong (train the bind->chain in) -> the one path to a NOVEL method.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=28800)
def qwen_grpo_v2(track: str = "LanguageQA", train_start: int = 100, train_n: int = 200,
                 steps: int = 300, kroll: int = 8, lr: float = 1e-6, beta: float = 0.04, eval_n: int = 50):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[grpo2] loading {MODEL} track={track} steps={steps} K={kroll} beta={beta}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    SR = proc.feature_extractor.sampling_rate
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."
    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    def audio(ex):
        a = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if isinstance(a, list): a = np.array(a, dtype=np.float32)
        a = a.astype(np.float32); return librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a
    train = [ds[i] for i in range(train_start, min(train_start+train_n, len(ds)))]
    def build(a16, instr):
        conv=[{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        t=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp=proc(text=t, audios=[a16], sampling_rate=SR, return_tensors="pt")
        return {k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}

    def tok_logprobs(full_ids, plen, feats, use_ref):
        # logprobs of rollout tokens under policy (use_ref=False) or frozen base (use_ref=True)
        am = torch.ones_like(full_ids)
        ctx = model.disable_adapter() if use_ref else _nullctx()
        with ctx:
            out = model(input_ids=full_ids, attention_mask=am, use_cache=False, **feats)
        lg = torch.log_softmax(out.logits[0, plen-1:-1, :].float(), dim=-1)
        toks = full_ids[0, plen:]
        return lg[torch.arange(len(toks)), toks]
    import contextlib
    def _nullctx(): return contextlib.nullcontext()

    @torch.no_grad()
    def eval_greedy(sl):
        model.eval(); c=0
        for ex in sl:
            inp=build(audio(ex), ex["multi_instruction"]+PROMPT)
            o=model.generate(**inp, max_new_tokens=250, do_sample=False, use_cache=True)
            c+=int(mcq_correct(proc.batch_decode(o[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0], ex["multi_answer"], ex["multi_instruction"]))
        return c/len(sl)
    eval_sl=[ds[i] for i in range(eval_n)]                       # trained track (held-out slice)
    dsx = load_dataset("SLLM-multi-hop/AnimalQA", split="test")   # UNTRAINED track (overfitting check)
    eval_x=[dsx[i] for i in range(eval_n)]
    def evals(): return eval_greedy(eval_sl), eval_greedy(eval_x)
    _b1,_b2 = evals()
    print(f"[grpo2] EVAL BEFORE  train-track({track})={_b1:.3f}  UNTRAINED(AnimalQA)={_b2:.3f}", flush=True)

    rng=np.random.default_rng(0)
    for step in range(steps):
        ex=train[int(rng.integers(len(train)))]; a=audio(ex)
        mi,gold=ex["multi_instruction"],ex["multi_answer"]; _,attr=_gold_letter_text(ex["single_answer"])
        inp=build(a, mi+PROMPT); plen=inp["input_ids"].shape[1]
        feats={k:v for k,v in inp.items() if k in ("input_features","feature_attention_mask")}
        model.eval()
        with torch.no_grad():
            gen=model.generate(**inp, max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.95, num_return_sequences=kroll, use_cache=True)
        rolls=gen[:,plen:]; chains=proc.batch_decode(rolls, skip_special_tokens=True)
        rew=np.array([(1.0 if mcq_correct(c,gold,mi) else 0.0)+(0.3 if _word_in(c,attr) else 0.0) for c in chains])
        adv=(rew-rew.mean())/(rew.std()+1e-4)
        model.train(); opt.zero_grad(); tot=0.0
        for i in range(kroll):
            full=torch.cat([inp["input_ids"], rolls[i:i+1]], dim=1)
            pol=tok_logprobs(full, plen, feats, use_ref=False)
            with torch.no_grad(): ref=tok_logprobs(full, plen, feats, use_ref=True)
            kl=(torch.exp(ref-pol)-(ref-pol)-1).mean()          # k3 KL estimator (>=0)
            loss=(-float(adv[i])*pol.mean() + beta*kl)/kroll
            loss.backward(); tot+=loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step%10==0: print(f"[grpo2] step {step} reward={rew.mean():.3f} correct={(rew>=1.0).mean():.2f} loss={tot:.4f}", flush=True)
        if step>0 and step%50==0:
            e1,e2 = evals(); print(f"[grpo2] >>> EVAL @ {step}  train({track})={e1:.3f}  UNTRAINED(Animal)={e2:.3f}", flush=True)
    f1,f2 = evals()
    print(f"\n[grpo2] ===== AFTER {steps} steps | train-track({track}) {_b1:.3f}->{f1:.3f} ({f1-_b1:+.3f}) | UNTRAINED(AnimalQA) {_b2:.3f}->{f2:.3f} ({f2-_b2:+.3f}) =====", flush=True)
    print(f"[grpo2] {'GENERALIZES (untrained track also improved -> learned binding, NOT overfit)' if (f1>_b1 and f2>_b2+0.02) else ('OVERFIT (train improved, untrained did not)' if f1>_b1+0.03 else 'no clear gain')}", flush=True)
    return {"track":track,"train_before":_b1,"train_after":f1,"untrained_before":_b2,"untrained_after":f2}


# ==================================================================================
# BINDING-AWARE GRPO v3 — the batch-1 FIX (DAPO-style). Two changes vs v2 that address WHY v2 was
# flat: (1) EFFECTIVE BATCH = accumulate `batch` items per optimizer step (not 1); (2) DYNAMIC
# SAMPLING = skip items whose K rollouts all agree (advantage=0 -> zero signal). So every update
# sees multiple items with REAL gradient. KL-stabilized, cross-track overfitting guard kept.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=28800)
def qwen_grpo_v3(track: str = "LanguageQA", train_start: int = 100, train_n: int = 200,
                 steps: int = 80, kroll: int = 8, batch: int = 4, lr: float = 1e-6, beta: float = 0.04, eval_n: int = 50):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, contextlib
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[grpo3] loading {MODEL} steps={steps} K={kroll} batch={batch} beta={beta} (dynamic sampling)...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    SR = proc.feature_extractor.sampling_rate
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    PROMPT = "\nFirst state the audio attribute, then reason step by step, then give the final answer."
    ds = load_dataset(f"SLLM-multi-hop/{track}", split="test")
    def audio(ex):
        a=ex["audio"]["array"]; sr=ex["audio"]["sampling_rate"]
        if isinstance(a,list): a=np.array(a,dtype=np.float32)
        a=a.astype(np.float32); return librosa.resample(a,orig_sr=sr,target_sr=SR) if sr!=SR else a
    train=[ds[i] for i in range(train_start, min(train_start+train_n,len(ds)))]
    def build(a16,instr):
        conv=[{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        t=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
        inp=proc(text=t,audios=[a16],sampling_rate=SR,return_tensors="pt")
        return {k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}
    def tok_lp(full,plen,feats,ref):
        ctx = model.disable_adapter() if ref else contextlib.nullcontext()
        with ctx:
            out=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False,**feats)
        lp=torch.log_softmax(out.logits[0,plen-1:-1,:].float(),dim=-1); toks=full[0,plen:]
        return lp[torch.arange(len(toks)),toks]
    @torch.no_grad()
    def eg(sl):
        model.eval(); c=0
        for ex in sl:
            inp=build(audio(ex), ex["multi_instruction"]+PROMPT)
            o=model.generate(**inp,max_new_tokens=250,do_sample=False,use_cache=True)
            c+=int(mcq_correct(proc.batch_decode(o[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0],ex["multi_answer"],ex["multi_instruction"]))
        return c/len(sl)
    esl=[ds[i] for i in range(eval_n)]; dsx=load_dataset("SLLM-multi-hop/AnimalQA",split="test"); ex2=[dsx[i] for i in range(eval_n)]
    def evals(): return eg(esl), eg(ex2)
    b1,b2=evals(); print(f"[grpo3] BEFORE train({track})={b1:.3f} UNTRAINED(Animal)={b2:.3f}", flush=True)

    rng=np.random.default_rng(0)
    for step in range(steps):
        opt.zero_grad(); tot=0.0; kept=0; att=0; rlist=[]
        while kept<batch and att<batch*5:
            att+=1; ex=train[int(rng.integers(len(train)))]; a=audio(ex)
            mi,gold=ex["multi_instruction"],ex["multi_answer"]; _,attr=_gold_letter_text(ex["single_answer"])
            inp=build(a,mi+PROMPT); plen=inp["input_ids"].shape[1]
            feats={k:v for k,v in inp.items() if k in ("input_features","feature_attention_mask")}
            model.eval()
            with torch.no_grad():
                gen=model.generate(**inp,max_new_tokens=200,do_sample=True,temperature=1.0,top_p=0.95,num_return_sequences=kroll,use_cache=True)
            rolls=gen[:,plen:]; ch=proc.batch_decode(rolls,skip_special_tokens=True)
            rew=np.array([(1.0 if mcq_correct(c,gold,mi) else 0.0)+(0.3 if _word_in(c,attr) else 0.0) for c in ch])
            if rew.std()<0.01: continue                      # DYNAMIC SAMPLING: skip zero-signal items
            adv=(rew-rew.mean())/(rew.std()+1e-4); rlist.append(rew.mean())
            model.train()
            for i in range(kroll):
                full=torch.cat([inp["input_ids"],rolls[i:i+1]],dim=1)
                pol=tok_lp(full,plen,feats,False)
                with torch.no_grad(): ref=tok_lp(full,plen,feats,True)
                kl=(torch.exp(ref-pol)-(ref-pol)-1).mean()
                ((-float(adv[i])*pol.mean()+beta*kl)/(batch*kroll)).backward(); 
            kept+=1
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
        if step%5==0: print(f"[grpo3] step {step} kept={kept}/{att} mean_reward={np.mean(rlist) if rlist else 0:.3f}", flush=True)
        if step>0 and step%20==0:
            e1,e2=evals(); print(f"[grpo3] >>> EVAL @ {step} train({track})={e1:.3f} UNTRAINED(Animal)={e2:.3f}", flush=True)
    f1,f2=evals()
    print(f"\n[grpo3] ===== AFTER {steps}x(batch{batch}) | train({track}) {b1:.3f}->{f1:.3f} ({f1-b1:+.3f}) | UNTRAINED(Animal) {b2:.3f}->{f2:.3f} ({f2-b2:+.3f}) =====", flush=True)
    print(f"[grpo3] {'GENERALIZES (both up, not overfit)' if (f1>b1+0.03 and f2>b2+0.02) else ('OVERFIT (train up, untrained flat)' if f1>b1+0.05 else 'FLAT (no gain -> binding-RL does not work here)')}", flush=True)
    return {"train_before":b1,"train_after":f1,"untrained_before":b2,"untrained_after":f2}


# ==================================================================================
# MMAR breakdown — base vs base+SC by reasoning-layer CATEGORY (Signal/Perception/Semantic/Cultural)
# and by MODALITY (speech/sound/music/mix). Turns the single MMAR number into a rich table AND tests
# the thesis: does SC help the reasoning-heavy layers more than perception?
# ==================================================================================
@app.function(image=qwen_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def mmar_breakdown(start: int = 0, n: int = 350, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa, soundfile as sf, ast as _ast, os, tarfile
    from collections import Counter, defaultdict
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    print("[mmarB] audio tarball...", flush=True)
    tar = hf_hub_download(repo_id="BoJack/MMAR", repo_type="dataset", filename="mmar-audio.tar.gz")
    ext = "/data/mmar_audio"
    if not os.path.exists(ext + "/.done"):
        os.makedirs(ext, exist_ok=True)
        with tarfile.open(tar) as t: t.extractall(ext)
        open(ext + "/.done", "w").close(); vol.commit()
    idx = {}
    for root, _d, fs in os.walk(ext):
        for f in fs:
            if f.endswith(".wav"): idx[f] = os.path.join(root, f)
    ds = load_dataset("BoJack/MMAR", split="test")
    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate; LET = "abcdefgh"

    def gen(a16, instr, k, greedy):
        conv=[{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        text=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp=proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inp={k2:(v.to("cuda") if hasattr(v,"to") else v) for k2,v in inp.items()}
        with torch.no_grad():
            out=model.generate(**inp, max_new_tokens=250, do_sample=not greedy, temperature=0.7, top_p=0.9,
                               num_return_sequences=1 if greedy else k)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def modality_of(m):
        m=(m or "").lower()
        if "mix" in m: return "mix"
        for x in ("speech","sound","music"): 
            if x in m: return x
        return "other"

    lo,hi=start,min(start+n,len(ds)); sel=ds.select(range(lo,hi))
    cat_b=defaultdict(int); cat_s=defaultdict(int); cat_n=defaultdict(int)
    mod_b=defaultdict(int); mod_s=defaultdict(int); mod_n=defaultdict(int)
    cb=cs=nn=skip=0
    for ex in sel:
        try:
            ch=ex["choices"]; ch=_ast.literal_eval(ch) if isinstance(ch,str) else ch
            ans=ex["answer"]; gi=next((i for i,c in enumerate(ch) if c.strip().lower()==ans.strip().lower()),None)
            if gi is None: skip+=1; continue
            p=idx.get(os.path.basename(ex["audio_path"]))
            if not p or not os.path.exists(p): skip+=1; continue
            a,sr=sf.read(p); a=a.mean(axis=1) if a.ndim>1 else a; a=a.astype(np.float32)
            if sr!=SR: a=librosa.resample(a,orig_sr=sr,target_sr=SR)
            a=a[:SR*30]
            opts=" ".join(f"({LET[i]}) {c}" for i,c in enumerate(ch))
            instr=f"{ex['question']}\n{opts}\nReason step by step, then give the final answer as the option letter."
            gold=f"({LET[gi]}) {ans}"
            okb=int(mcq_correct(gen(a,instr,1,True)[0], gold, instr))
            chains=gen(a,instr,kchains,False); picks=[pp for pp in (pick_option(c,instr) for c in chains) if pp]
            oks=int(mcq_correct(f"({Counter(picks).most_common(1)[0][0]})" if picks else "", gold, instr))
            cat=(ex.get("category") or "Other").replace(" Layer",""); mod=modality_of(ex.get("modality"))
            cb+=okb; cs+=oks; nn+=1
            cat_b[cat]+=okb; cat_s[cat]+=oks; cat_n[cat]+=1
            mod_b[mod]+=okb; mod_s[mod]+=oks; mod_n[mod]+=1
            if nn%25==0: print(f"[mmarB] {nn}/{hi-lo} base={cb/nn:.3f} base+SC={cs/nn:.3f}", flush=True)
        except Exception as e:
            skip+=1
            if skip<=3: print(f"[mmarB] skip {str(e)[:80]}", flush=True)
    print(f"\n[mmarB] ===== OVERALL N={nn} base={cb/max(nn,1):.3f} base+SC={cs/max(nn,1):.3f} (skip={skip}) =====", flush=True)
    print("[mmarB] --- by REASONING LAYER ---", flush=True)
    for c in sorted(cat_n, key=lambda x:-cat_n[x]):
        print(f"[mmarB] CAT {c:12s} n={cat_n[c]:3d} base={cat_b[c]/cat_n[c]:.3f} base+SC={cat_s[c]/cat_n[c]:.3f} d={ (cat_s[c]-cat_b[c])/cat_n[c]:+.3f}", flush=True)
    print("[mmarB] --- by MODALITY ---", flush=True)
    for m in sorted(mod_n, key=lambda x:-mod_n[x]):
        print(f"[mmarB] MOD {m:8s} n={mod_n[m]:3d} base={mod_b[m]/mod_n[m]:.3f} base+SC={mod_s[m]/mod_n[m]:.3f} d={(mod_s[m]-mod_b[m])/mod_n[m]:+.3f}", flush=True)
    return {"overall_base":cb/max(nn,1),"overall_sc":cs/max(nn,1),"n":nn}


# ==================================================================================
# SELF-DERIVING BIND — the deployable test. Instead of SAKURA's PROVIDED single-hop question,
# the model SELF-DERIVES which attribute the multi-hop question depends on (self-ask), determines
# it from the audio, then re-grounds + reasons + SC. No benchmark structure used. If this matches
# provided-BIND (~0.70), BIND is genuinely general/deployable; if it drops toward SC (~0.60), the
# provided question was doing the work.
# ==================================================================================
@app.function(image=qwen_train_img, gpu="A100-80GB", volumes={"/data": vol}, timeout=14400)
def qwen_bind_selfask(tracks: str = ",".join(_TRACKS), start: int = 0, n: int = 100, kchains: int = 5):
    import os as _os; _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np, torch, librosa
    from collections import Counter
    from datasets import load_dataset
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"[selfask] loading {MODEL} (K={kchains})...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    SR = proc.feature_extractor.sampling_rate

    def gen(a16, instr, k, greedy):
        conv=[{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":instr}]}]
        text=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp=proc(text=text, audios=[a16], sampling_rate=SR, return_tensors="pt")
        inp={kk:(v.to("cuda") if hasattr(v,"to") else v) for kk,v in inp.items()}
        with torch.no_grad():
            if greedy: out=model.generate(**inp, max_new_tokens=60, do_sample=False)
            else: out=model.generate(**inp, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=k)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    res_b={}; res_s={}
    for track in tracks.split(","):
        ds=load_dataset(f"SLLM-multi-hop/{track}", split="test")
        lo,hi=start,min(start+n,len(ds)); sel=ds.select(range(lo,hi))
        cb=cs=nn=0
        for ex in sel:
            a=ex["audio"]["array"]; sr=ex["audio"]["sampling_rate"]
            if isinstance(a,list): a=np.array(a,dtype=np.float32)
            a=a.astype(np.float32)
            if sr!=SR: a=librosa.resample(a,orig_sr=sr,target_sr=SR)
            mi,gold=ex["multi_instruction"],ex["multi_answer"]
            # base+SC
            chains=gen(a, mi+"\nReason step by step, then give the final answer.", kchains, False)
            picks=[p for p in (pick_option(c,mi) for c in chains) if p]
            cb+=int(mcq_correct(f"({Counter(picks).most_common(1)[0][0]})" if picks else "", gold, mi))
            # SELF-DERIVING BIND: model derives the attribute itself (no single_instruction)
            selfask=mi+"\nBefore answering, identify the single key audio attribute this question depends on, and determine it from the audio. Reply with ONLY that attribute in a few words."
            attr=gen(a, selfask, 1, True)[0].strip().split("\n")[0][:60]
            compose=mi+f"\n[Audio analysis: {attr}. Use BOTH this and the audio.]\nReason step by step, then give the final answer."
            ch2=gen(a, compose, kchains, False)
            p2=[p for p in (pick_option(c,mi) for c in ch2) if p]
            cs+=int(mcq_correct(f"({Counter(p2).most_common(1)[0][0]})" if p2 else "", gold, mi)); nn+=1
            if nn<=3: print(f"[selfask ex{nn}] {track} self-derived attr={attr!r}", flush=True)
            if nn%25==0: print(f"[selfask] {track} {nn}/{hi-lo} base+SC={cb/nn:.3f} selfBIND={cs/nn:.3f}", flush=True)
        res_b[track]=cb/nn; res_s[track]=cs/nn
        print(f"[selfask] {track} [{lo}:{hi}] base+SC={cb/nn:.3f} self-BIND={cs/nn:.3f} lift={cs/nn-cb/nn:+.3f}", flush=True)
    ab=sum(res_b.values())/len(res_b); asel=sum(res_s.values())/len(res_s)
    print(f"\n[selfask] ===== [{start}:{start+n}] base+SC={ab:.3f} self-BIND={asel:.3f} lift={asel-ab:+.3f} (provided-BIND ref=0.708) =====", flush=True)
    print(f"[selfask] {'SELF-BIND GENERALIZES (close to provided-BIND, no benchmark structure)' if asel>=ab+0.04 else 'self-BIND ~ SC (provided question was doing the work; ship SC+cascade as deployable)'}", flush=True)
    return {"base_sc":ab, "self_bind":asel}
