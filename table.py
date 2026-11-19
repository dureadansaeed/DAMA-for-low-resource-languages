import json, glob, os, collections as c, statistics as st
SEEN = {"hi_in", "cy_gb", "be_by", "fa_ir", "sw_ke"}; EMB = 39.83
R = {}
for f in glob.glob("runs2/*.json"):
    r = json.load(open(f)); b = os.path.basename(f)[:-5]; k = (r["ds"], r["lang"], r["method"])
    base = f"{r['ds']}_{r['lang']}_{r['method']}"
    if b == base: R[(0,) + k] = r
    elif b == base + "_s1": R[(1,) + k] = r
def grp(ds, l): return "CV" if ds == "cv" else ("FLseen" if l in SEEN else "FLunseen")
def adp(r):
    new = not (r["ds"] == "fleurs" and r["lang"] in SEEN)
    return r["trainable_M"] if r["method"] == "ft" else r["trainable_M"] - (EMB if new else 0)
avg = lambda v: sum(v) / len(v) if v else float("nan")
M = sorted({k[3] for k in R if k[0] == 0 and k[3] != "zeroshot"})
print(f"{'method':8s} {'CV':>6s} {'FLseen':>7s} {'FLunsn':>7s} {'ALL':>6s} {'ALL_s1':>7s} {'adapterM':>9s} {'GB':>5s} {'min':>5s}")
for m in M:
    s0 = [r for k, r in R.items() if k[0] == 0 and k[3] == m]
    g = c.defaultdict(list)
    for r in s0: g[grp(r["ds"], r["lang"])].append(r["wer"])
    s1 = [r["wer"] for k, r in R.items() if k[0] == 1 and k[3] == m]
    print(f"{m:8s} {avg(g['CV']):6.1f} {avg(g['FLseen']):7.1f} {avg(g['FLunseen']):7.1f} {avg([r['wer'] for r in s0]):6.1f} "
          f"{avg(s1):7.1f} {avg([adp(r) for r in s0]):9.1f} {avg([r['peak_gb'] for r in s0]):5.1f} {avg([r['train_s'] for r in s0])/60:5.1f}")
print("\nDAMA minus LoRA per language (negative = DAMA better)")
for s in (0, 1):
    d = [R[(s, ds, l, "dama")]["wer"] - R[(s, ds, l, "lora")]["wer"] for (q, ds, l, m) in R if q == s and m == "dama" and (s, ds, l, "lora") in R]
    if d: print(f"seed {s}: mean {avg(d):+.2f}  sd {st.pstdev(d):.2f}  DAMA better on {sum(x < 0 for x in d)}/{len(d)} languages")
print("\nSeen languages (WER): zeroshot vs methods")
L = sorted(SEEN); print("method  " + " ".join(f"{l[:2]:>6s}" for l in L) + "   avg")
for m in ["zeroshot"] + M:
    v = [R.get((0, "fleurs", l, m), {}).get("wer", float("nan")) for l in L]
    print(f"{m:8s}" + " ".join(f"{x:6.1f}" for x in v) + f" {avg(v):6.1f}")
