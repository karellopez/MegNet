# python easy_megnet.py \
#   --filename /data/datasets/ds003483/sub-012/ses-1/meg/sub-012_ses-1_task-deduction_run-1_meg.fif \
#   --results-dir /data/datasets/MEGnet_results/ds003483/ \
#   --line-freq 50 \
#   --skip-apply \
#   --run-qc \
#   --run-ref-compare
import subprocess
from pathlib import Path

DATASET_ROOT = Path("/data/datasets/ds003483/")
RESULTS_DIR = Path("/data/datasets/MEGnet_results/ds003483/")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fif_files = [
    p for p in DATASET_ROOT.rglob("*_meg.fif")
    if (
        p.is_file()
        and "derivatives" not in p.parts
        and ".git" not in p.parts
        and "sourcedata" not in p.parts
    )
]

print(f"Found {len(fif_files)} FIF files")

for fif in fif_files:

    print(f"\nRunning MEGnet on: {fif}")

    cmd = [
        "python",
        "easy_megnet.py",
        "--filename", str(fif),
        "--results-dir", str(RESULTS_DIR),
        "--line-freq", "50",
        "--skip-apply",
        "--run-qc",
        "--run-ref-compare",
    ]

    subprocess.run(cmd)