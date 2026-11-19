import os, csv, sys, json, random, statistics
csv.field_size_limit(sys.maxsize)
random.seed(0)
CV = "data/cv/CL-MASR"; FL = "data/fleurs/data"
cv_langs = {"rw":600,"eo":600,"kab":600,"lg":600,"mhr":600,"ckb":484,"ab":600,"kmr":296,"fy-NL":330,"ia":313}
fl_langs = {"hi_in":399,"cy_gb":600,"be_by":571,"fa_ir":600,"sw_ke":600,
            "lg_ug":600,"ckb_iq":584,"ast_es":452,"ceb_ph":600,"or_in":206}
CAP = {"train":600, "dev":60, "test":60}   # minutes
def rd(p, header=True):
    with open(p, encoding="utf8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if header:
            cols = next(r)
            for row in r:
                if len(row) >= len(cols): yield dict(zip(cols,row))
        else:
            yield from r
def take(rows, cap_min):
    out, tot = [], 0.0
    for r in rows:
        if tot + r["duration"]/60 > cap_min: continue
        out.append(r); tot += r["duration"]/60
    return out
def dump(name, rows):
    with open(f"manifests/{name}.jsonl","w",encoding="utf8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    return sum(r["duration"] for r in rows)/60

print("== COMMON VOICE (minutes: train/dev/test | paper train) ==")
for l, paper in cv_langs.items():
    d = f"{CV}/{l}"; have = set(os.listdir(f"{d}/clips")); res = {}
    for s in ["train","dev","test"]:
        rows = []
        for r in rd(f"{d}/{s}_with_duration.tsv"):
            if r["path"] in have:
                rows.append({"audio":f"{d}/clips/{r['path']}","text":r["sentence"],
                             "lang":l,"duration":float(r["duration"])})
        med = statistics.median([r["duration"] for r in rows]) if rows else 0
        if med > 100:                       # durations stored in ms
            for r in rows: r["duration"] /= 1000
        rows = [r for r in rows if r["duration"] <= 10]   # paper: drop >10 s
        random.shuffle(rows)
        res[s] = dump(f"cv_{l}_{s}", take(rows, CAP[s]))
    print(f"{l:6} {res['train']:6.0f} {res['dev']:5.0f} {res['test']:5.0f} | {paper}")

print("\n== FLEURS (minutes: train/dev/test | paper train) ==")
def fl_rows(l, s):
    rows = []
    for r in rd(f"{FL}/{l}/{s}.tsv", header=False):
        dur = int(r[5])/16000
        if dur <= 30:                       # paper: drop >30 s
            rows.append({"audio":f"{FL}/{l}/audio/{s}/{r[1]}","text":r[2],
                         "text_norm":r[3],"lang":l,"duration":dur})
    random.shuffle(rows); return rows
for l, paper in fl_langs.items():
    tr, dv, te = fl_rows(l,"train"), fl_rows(l,"dev"), fl_rows(l,"test")
    te_sel = take(te, CAP["test"]); used = {r["audio"] for r in te_sel}
    dv_sel = take(dv, CAP["dev"])
    short = CAP["dev"] - sum(r["duration"] for r in dv_sel)/60
    if short > 0: dv_sel += take([r for r in te if r["audio"] not in used], short)
    a = dump(f"fleurs_{l}_train", take(tr, CAP["train"]))
    b = dump(f"fleurs_{l}_dev", dv_sel); c = dump(f"fleurs_{l}_test", te_sel)
    print(f"{l:7} {a:6.0f} {b:5.0f} {c:5.0f} | {paper}")
