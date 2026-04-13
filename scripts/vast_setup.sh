#!/bin/bash
# Run this on a Vast.ai instance after uploading the kairos/ directory.
# Recommended: A100 40GB, PyTorch 2.1+ image

set -e

echo "=== Installing dependencies ==="
pip install -q numpy pandas torch transformers scipy scikit-learn matplotlib tqdm properscoring

echo "=== Installing TSFM packages ==="
pip install -q chronos-forecasting
pip install -q timesfm
pip install -q "uni2ts[notebook] @ git+https://github.com/SalesforceAIResearch/uni2ts.git"

echo "=== Setup complete ==="
echo "Run experiments with:"
echo "  python scripts/experiment_01_baselines.py --models naive        # Quick sanity check (no GPU needed)"
echo "  python scripts/experiment_01_baselines.py --models chronos      # ~10 min on A100"
echo "  python scripts/experiment_01_baselines.py --models timesfm      # ~10 min on A100"
echo "  python scripts/experiment_01_baselines.py --models moirai       # ~15 min on A100"
echo "  python scripts/experiment_01_baselines.py --models all          # ~40 min total"
