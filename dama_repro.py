import os, json, time, copy, math, argparse, traceback, unicodedata
ABL = os.environ.get("ABL", "")
RH = int(os.environ.get("RH", 32)); RL = int(os.environ.get("RL", 8))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa, jiwer
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor

ORDER = ["fleurs:or_in","cv:kmr","cv:ia","cv:fy-NL","fleurs:hi_in","fleurs:ast_es","cv:ckb","fleurs:be_by","fleurs:ckb_iq",
         "cv:rw","cv:eo","cv:kab","cv:lg","cv:mhr","cv:ab","fleurs:cy_gb","fleurs:fa_ir","fleurs:sw_ke","fleurs:lg_ug","fleurs:ceb_ph"]
SEEN = {"hi_in":"hi","cy_gb":"cy","be_by":"be","fa_ir":"fa","sw_ke":"sw"}
BASE = {"fy-NL":"fy","lg_ug":"lg","ckb_iq":"ckb","ast_es":"ast","ceb_ph":"ceb","or_in":"or"}
LR = {"zeroshot":0,"ft":1e-5,"lora":5e-4,"lorafa":5e-4,"dora":5e-4,"dama":5e-4,"loraxs":1e-3,"vblora":1e-3,"adalora":1e-3}
REGEX = r"model\.decoder\.layers\.\d+\.((self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"

p = argparse.ArgumentParser()
p.add_argument("--methods", nargs="+", default=["lora","dama","ft","lorafa","dora","loraxs","adalora","vblora"])
p.add_argument("--langs", nargs="+", default=ORDER)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--bs", type=int, default=6)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--tag", default="")
p.add_argument("--max-train", type=int, default=0)
p.add_argument("--max-eval", type=int, default=0)
p.add_argument("--workers", type=int, default=6)
p.add_argument("--model", default="openai/whisper-small")
a = p.parse_args()
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs("runs2", exist_ok=True)

def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return " ".join("".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t).split())
def padl(ys):
    y = torch.full((len(ys), max(map(len, ys))), -100, dtype=torch.long)
    for k, v in enumerate(ys): y[k, :len(v)] = torch.tensor(v)
    return y

# ---------------- adapters (decoder only) ----------------
class Ad(nn.Module):
    def __init__(s, base, kind, r, alpha, drop=0.0, freezeA=False, svd_tail=False):
        super().__init__()
        s.base, s.kind, s.scale, s.drop = base, kind, alpha / r, nn.Dropout(drop)
        W = base.weight.data.float(); o, i = W.shape
        if kind == "loraxs":
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            s.register_buffer("A", Vh[:r].clone()); s.register_buffer("B", (U[:, :r] * S[:r]).clone())
            s.R = nn.Parameter(torch.zeros(r, r, device=W.device)); s.scale = 1.0
            return
        A = torch.empty(r, i, device=W.device); nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        if svd_tail:
            A = torch.linalg.svd(W, full_matrices=False)[2][-r:].clone()
        s.A = nn.Parameter(A, requires_grad=not freezeA)
        s.B = nn.Parameter(torch.zeros(o, r, device=W.device))
        if kind == "dora": s.m = nn.Parameter(W.norm(dim=1).clone())
    def forward(s, x):
        if s.kind == "loraxs":
            return s.base(x) + F.linear(F.linear(F.linear(x, s.A), s.R), s.B)
        if s.kind == "dora":
            w = s.base.weight + s.scale * (s.B @ s.A)
            y = F.linear(x, w) * (s.m / w.norm(dim=1).detach())
            return y if s.base.bias is None else y + s.base.bias
        return s.base(x) + F.linear(F.linear(s.drop(x), s.A), s.B) * s.scale

def dama_cfg(l, L, th1=0.3, th2=0.7, rh=32, rl=8):
    le, ll = int(th1 * L), int(th2 * L)
    if l <= le: return round(rh - (l - 1) / max(le - 1, 1) * (rh - rl)), False
    if l < ll: return rl, True
    return round(rl + (l - ll) / max(L - ll, 1) * (rh - rl)), False

def adapt(m, method, steps, new):
    for q in m.parameters(): q.requires_grad = False
    dec = m.model.decoder; L = len(dec.layers)
    if method == "zeroshot": return m
    if method == "ft":
        for q in dec.parameters(): q.requires_grad = True
        return m
    if method == "adalora":
        from peft import AdaLoraConfig, get_peft_model
        m = get_peft_model(m, AdaLoraConfig(target_r=32, init_r=48, tinit=steps // 10, tfinal=steps // 5, deltaT=10,
              beta1=0.85, beta2=0.85, lora_alpha=32, total_step=steps, target_modules=REGEX))
    elif method == "vblora":
        from peft import VBLoRAConfig, get_peft_model
        m = get_peft_model(m, VBLoRAConfig(r=64, num_vectors=90, vector_length=m.config.d_model, topk=2, target_modules=REGEX))
    else:
        for li, layer in enumerate(dec.layers, 1):
            r, mid = dama_cfg(li, L, rh=RH, rl=RL); r = 16 if ABL in ("uniform", "uniformrank") else r
            for holder, name in [(layer.self_attn, n) for n in ("q_proj","k_proj","v_proj","out_proj")] + \
                                [(layer.encoder_attn, n) for n in ("q_proj","k_proj","v_proj","out_proj")] + [(layer,"fc1"), (layer,"fc2")]:
                b = getattr(holder, name)
                ad = {"lora": lambda: Ad(b, "lora", 64, 64, 0.1), "lorafa": lambda: Ad(b, "lora", 64, 64, 0.1, freezeA=True),
                      "dora": lambda: Ad(b, "dora", 64, 64, 0.0), "loraxs": lambda: Ad(b, "loraxs", 64, 64),
                      "dama": lambda: Ad(b, "lora", r, RH, 0.1, freezeA=(mid and ABL not in ("nofreeze", "uniform")), svd_tail=(mid and ABL not in ("nosvd", "uniform")))}[method]()
                setattr(holder, name, ad)
    if new:
        emb = m.get_input_embeddings().weight; emb.requires_grad_(True); old = emb.shape[0] - 1
        emb.register_hook(lambda g: torch.cat([torch.zeros_like(g[:old]), g[old:]]))
    return m

# ---------------- data cache (features computed once per language) ----------------
def prep(ds, lang):
    code = SEEN.get(lang) or BASE.get(lang, lang); ltok = f"<|{code}|>"
    tok = WhisperTokenizer.from_pretrained(a.model); fe = WhisperFeatureExtractor.from_pretrained(a.model)
    base = WhisperForConditionalGeneration.from_pretrained(a.model)
    new = ltok not in tok.get_vocab()
    if new: tok.add_tokens([ltok], special_tokens=True); base.resize_token_embeddings(len(tok))
    T = {k: tok.convert_tokens_to_ids(v) for k, v in dict(LID=ltok, SOT="<|startoftranscript|>", TR="<|transcribe|>", NT="<|notimestamps|>").items()}
    T["EOT"] = tok.eos_token_id
    class DS(Dataset):
        def __init__(s, rows): s.r = rows
        def __len__(s): return len(s.r)
        def __getitem__(s, i):
            r = s.r[i]; w, _ = librosa.load(r["audio"], sr=16000, mono=True)
            x = fe(w, sampling_rate=16000, return_tensors="np").input_features[0]
            return x, [T["LID"], T["TR"], T["NT"]] + tok.encode(r["text"], add_special_tokens=False)[:430] + [T["EOT"]], r["text"]
    def coll(b): return torch.from_numpy(np.stack([i[0] for i in b])).half(), [i[1] for i in b], [i[2] for i in b]
    C = {}
    for s, n in [("train", a.max_train), ("dev", 200), ("test", a.max_eval)]:
        rows = [json.loads(l) for l in open(f"manifests/{ds}_{lang}_{s}.jsonl", encoding="utf8")]
        rows = rows[:n] if n else rows
        X = torch.empty(len(rows), 80, 3000, dtype=torch.float16); Y, TX, k = [], [], 0
        for x, y, t in DataLoader(DS(rows), batch_size=16, num_workers=a.workers, collate_fn=coll, multiprocessing_context="fork"):
            X[k:k + len(x)] = x; k += len(x); Y += y; TX += t
            if (k // 16) % 40 == 0: print(f"  caching {s}: {k}/{len(rows)}", flush=True)
        C[s] = (X, Y, TX)
    return base, new, T, C, ltok

def evaluate(m, C, T, ds):
    m.eval(); inner = m.get_base_model() if hasattr(m, "get_base_model") else m
    X = C[0]; maxnew = 80 if ds == "cv" else 120
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 16):
            x = X[b:b + 16].to(dev).float(); n = len(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc = inner.model.encoder(x)
                cur = torch.tensor([[T["SOT"], T["LID"], T["TR"], T["NT"]]] * n, device=dev)
                fin = torch.zeros(n, dtype=torch.bool, device=dev)
                for _ in range(maxnew):
                    lg = m(encoder_outputs=enc, decoder_input_ids=cur).logits[:, -1].float()
                    lg[:, T["SOT"]:] = -float("inf")
                    nxt = lg.argmax(-1); nxt[fin] = T["EOT"]
                    cur = torch.cat([cur, nxt[:, None]], 1); fin |= nxt == T["EOT"]
                    if fin.all(): break
            out += [r[:r.index(T["EOT"])] if T["EOT"] in r else r for r in cur[:, 4:].tolist()]
    return out

def vloss(m, C):
    m.eval(); X, Y, _ = C["dev"]; t = n = 0
    with torch.no_grad():
        for b in range(0, len(X), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += m(input_features=X[b:b + 16].to(dev).float(), labels=padl(Y[b:b + 16]).to(dev)).loss.item(); n += 1
    m.train(); return t / max(n, 1)

def run(ds, lang, method, base, new, T, C, tok):
    torch.manual_seed(int(os.environ.get("SEED", 0))); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    X, Y, _ = C["train"]; N = len(X); spe = N // a.bs; steps = spe * a.epochs
    m = adapt(copy.deepcopy(base).to(dev), method, steps, new)
    params = [q for q in m.parameters() if q.requires_grad]; ntrain = sum(q.numel() for q in params)
    lr = a.lr or LR[method]
    opt = torch.optim.AdamW(params or [nn.Parameter(torch.zeros(1))], lr=lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=0, threshold=0.0025, threshold_mode="rel")
    t0 = time.time(); step = 0; m.train()
    for ep in range(0 if method == "zeroshot" else a.epochs):
        perm = torch.randperm(N)
        for b in range(spe):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(input_features=X[idx].to(dev).float(), labels=padl([Y[i] for i in idx]).to(dev)).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            if method == "adalora": m.base_model.update_and_allocate(step)
            opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  {method} step {step}/{steps} ({100*step/steps:.0f}%) loss {loss.item():.3f} "
                      f"elapsed {el/60:.1f}m, ~{el/step*(steps-step)/60:.1f}m left", flush=True)
            if step % max(1, spe // 8) == 0: sch.step(vloss(m, C))
    train_s = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9
    hyp = evaluate(m, C["test"], T, ds)
    hyp = [tok.decode(h, skip_special_tokens=True) for h in hyp]
    pairs = [(norm(r), norm(h)) for r, h in zip(C["test"][2], hyp) if norm(r)]
    wer = 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
    res = dict(ds=ds, lang=lang, method=method, lr=lr, wer=round(wer, 2), trainable_M=round(ntrain / 1e6, 2),
               peak_gb=round(peak, 2), train_s=round(train_s), n_test=len(pairs))
    del m, opt; return res

TOTAL = len(a.langs) * len(a.methods); done_t = []; T_START = time.time()
def n_done(): return sum(os.path.exists(f"runs2/{k.split(':')[0]}_{k.split(':')[1]}_{m}{a.tag}.json") for k in a.langs for m in a.methods)
for key in a.langs:
    ds, lang = key.split(":")
    todo = [m for m in a.methods if not os.path.exists(f"runs2/{ds}_{lang}_{m}{a.tag}.json")]
    if not todo: continue
    print(f"\n=== {key}: caching features ({time.strftime('%H:%M')}) ===", flush=True)
    try: base, new, T, C, ltok = prep(ds, lang)
    except Exception: traceback.print_exc(); continue
    tok = WhisperTokenizer.from_pretrained(a.model)
    if new: tok.add_tokens([ltok], special_tokens=True)
    for method in todo:
        nd = n_done(); avg = (sum(done_t) / len(done_t)) if done_t else None
        eta = f", overall ETA ~{avg*(TOTAL-nd)/3600:.1f}h" if avg else ""
        print(f"--- [{nd}/{TOTAL} runs done, {100*nd/TOTAL:.0f}%] {key} {method} start {time.strftime('%H:%M')}{eta}", flush=True)
        t_run = time.time()
        try:
            res = run(ds, lang, method, base, new, T, C, tok); done_t.append(time.time() - t_run)
            json.dump(res, open(f"runs2/{ds}_{lang}_{method}{a.tag}.json", "w")); print("RESULT", json.dumps(res), flush=True)
        except Exception:
            traceback.print_exc(); print(f"FAILED {key} {method}", flush=True)
        torch.cuda.empty_cache()
    del C, base
print("ALL DONE", time.strftime("%H:%M"))
