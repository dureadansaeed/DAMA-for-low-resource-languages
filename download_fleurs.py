from huggingface_hub import snapshot_download
langs = ["hi_in","cy_gb","be_by","fa_ir","sw_ke","lg_ug","ckb_iq","ast_es","ceb_ph","or_in"]
for l in langs:
    snapshot_download("google/fleurs", repo_type="dataset",
        allow_patterns=[f"data/{l}/*"], local_dir="data/fleurs")
    print("done", l, flush=True)
