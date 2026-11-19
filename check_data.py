import os, csv, sys
csv.field_size_limit(sys.maxsize)
CV = "data/cv/CL-MASR"
cv_langs = ["rw","eo","kab","lg","mhr","ckb","ab","kmr","fy-NL","ia"]
fl = {"hi_in":"Hindi","cy_gb":"Welsh","be_by":"Belarusian","fa_ir":"Persian",
      "sw_ke":"Swahili","lg_ug":"Luganda","ckb_iq":"Central Kurdish",
      "ast_es":"Asturian","ceb_ph":"Cebuano","or_in":"Oriya"}

print("== COMMON VOICE ==")
for l in cv_langs:
    d = f"{CV}/{l}"
    if not os.path.isdir(d): print(l, "MISSING"); continue
    n = len(os.listdir(f"{d}/clips")) if os.path.isdir(f"{d}/clips") else 0
    rows = {s: sum(1 for _ in open(f"{d}/{s}.tsv", encoding="utf8")) - 1 for s in ["train","dev","test"]}
    print(l, "clips:", n, "rows:", rows)
print("duration file header:", open(f"{CV}/rw/train_with_duration.tsv", encoding="utf8").readline().strip())

print("\n== FLEURS ==")
for l, name in fl.items():
    out = []
    for s in ["train","dev","test"]:
        tsv = f"data/fleurs/data/{l}/{s}.tsv"
        wavs = f"data/fleurs/data/{l}/audio/{s}"
        if not os.path.exists(tsv) or not os.path.isdir(wavs): out.append(f"{s}: MISSING"); continue
        rows = [r for r in csv.reader(open(tsv, encoding="utf8"), delimiter="\t", quoting=csv.QUOTE_NONE)]
        hrs = sum(int(r[5]) for r in rows) / 16000 / 3600
        nw = len([f for f in os.listdir(wavs) if f.endswith(".wav")])
        out.append(f"{s}: {len(rows)} rows, {nw} wavs, {hrs:.1f} h")
    print(f"{l} ({name}) | " + " | ".join(out))
