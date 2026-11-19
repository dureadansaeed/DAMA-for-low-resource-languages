#!/bin/bash
cd ~/dama && source .venv/bin/activate
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO CUDA"; exit 1; }
[ -f dama_repro.py ] || { echo "dama_repro.py missing in ~/dama"; exit 1; }
[ "$(ls manifests | wc -l)" -eq 60 ] || { echo "manifests missing (expected 60 files)"; exit 1; }
mkdir -p runs2; rm -f runs2/*_smoke.json
echo "== SMOKE TEST $(date +%H:%M), about 5-10 min =="
python dama_repro.py --langs cv:ia --epochs 1 --max-train 60 --max-eval 32 --tag _smoke > smoke.log 2>&1
for m in lora dama ft lorafa dora loraxs; do
  [ -f runs2/cv_ia_${m}_smoke.json ] || { echo "SMOKE FAILED for $m. Last errors:"; grep -i -B3 -A6 "error\|Traceback" smoke.log | tail -40; exit 1; }
done
METHODS="lora dama ft lorafa dora loraxs"
for m in adalora vblora; do
  if [ -f runs2/cv_ia_${m}_smoke.json ]; then METHODS="$METHODS $m"; echo "$m OK"; else echo "$m FAILED, left out of the run (tell Claude)"; fi
done
rm -f runs2/*_smoke.json
echo "== SMOKE OK, starting full run with: $METHODS =="
nohup python dama_repro.py --methods $METHODS > overnight.log 2>&1 &
echo "started, pid $!. You can leave now."
