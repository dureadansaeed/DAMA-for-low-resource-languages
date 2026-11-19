import json, glob, os
R = {}
for f in glob.glob("runs2/*.json"):
    r = json.load(open(f)); b = os.path.basename(f)[:-5]; t = b[len(f"{r['ds']}_{r['lang']}_{r['method']}"):]
    if r["method"] in ("dama", "lora") and t in ("", "_abluniform", "_ablnosvd", "_abluniformrank"):
        R.setdefault(r["method"] + t, {})[(r["ds"], r["lang"])] = r["wer"]
base, lora = R["dama"], R["lora"]
print(f"{'variant':16s} {'n':>3s} {'avg':>6s} {'vs DAMA':>8s} {'wins':>5s} {'vs LoRA':>8s}")
for k in ("dama", "dama_abluniform", "dama_ablnosvd", "dama_abluniformrank", "lora"):
    v = R.get(k, {}); L = [l for l in v if l in base and l in lora]
    if not L: continue
    d1 = [v[l] - base[l] for l in L]; d2 = [v[l] - lora[l] for l in L]
    print(f"{k:16s} {len(L):3d} {sum(v[l] for l in L)/len(L):6.1f} {sum(d1)/len(L):+8.2f} {sum(x < 0 for x in d1):2d}/{len(L):<2d} {sum(d2)/len(L):+8.2f}")
print("wins = languages where the variant beats default DAMA; negative = better")
