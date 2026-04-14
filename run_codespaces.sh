#!/usr/bin/env bash
# ============================================================
#  Run in GitHub Codespaces — fire and forget
# ============================================================
#
#  1. Open your repo in Codespaces
#  2. Drop rerun_full_stats.py into the repo root
#  3. In the terminal:
#
#       chmod +x run_codespaces.sh
#       nohup bash run_codespaces.sh > run.log 2>&1 &
#
#  4. Close the tab. Come back later. Check run.log.
#
#     Or watch live:  tail -f run.log
# ============================================================

set -euo pipefail

echo "=== SQPP Codespaces Runner ==="
echo "Started: $(date)"
echo ""

# ── Install deps ─────────────────────────────────────────────
echo "[1/3] Installing dependencies..."
pip install -q scikit-learn xgboost pandas matplotlib scipy joblib 2>/dev/null \
  || pip install --break-system-packages -q scikit-learn xgboost pandas matplotlib scipy joblib

# ── Check BIRD data exists ───────────────────────────────────
echo "[2/3] Checking BIRD databases..."
BIRD_FOUND=false
for candidate in "Mini Dev/MINIDEV" "data/bird" "MINIDEV"; do
    if [ -d "$candidate/dev_databases" ]; then
        echo "  Found BIRD at: $candidate/"
        BIRD_FOUND=true
        break
    fi
done

if [ "$BIRD_FOUND" = false ]; then
    echo "  WARNING: BIRD dev_databases not found. main.py may fail."
    echo "  The rerun script will try to use existing CSVs if available."
fi

# ── Run the full stats pipeline ──────────────────────────────
echo "[3/3] Running rerun_full_stats.py..."
echo ""
python -u rerun_full_stats.py

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo ""
echo "Your reports/ folder now contains:"
ls -la reports/ 2>/dev/null || echo "(empty)"
echo ""
echo "Your artifacts/ folder now contains:"
ls -la artifacts/*.joblib 2>/dev/null || echo "(empty)"
echo ""
echo "Next: git add -A && git commit -m 'Full stats rerun' && git push"
