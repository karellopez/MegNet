import subprocess
from pathlib import Path

INPUT_PATHS = [
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC510483/meg/sub-CC510483_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC320621/meg/sub-CC320621_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC420180/meg/sub-CC420180_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC711245/meg/sub-CC711245_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC520136/meg/sub-CC520136_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC320361/meg/sub-CC320361_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/smt/sub-CC620413/meg/sub-CC620413_task-smt_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC220518/meg/sub-CC220518_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC320107/meg/sub-CC320107_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC221740/meg/sub-CC221740_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC221737/meg/sub-CC221737_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC120470/meg/sub-CC120470_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC410287/meg/sub-CC410287_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/passive/sub-CC710548/meg/sub-CC710548_task-passive_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC220828/meg/sub-CC220828_task-rest_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC710382/meg/sub-CC710382_task-rest_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC520247/meg/sub-CC520247_task-rest_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC320089/meg/sub-CC320089_task-rest_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC410297/meg/sub-CC410297_task-rest_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC120120/meg/sub-CC120120_task-rest_meg.fif",
"/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/rest/sub-CC420286/meg/sub-CC420286_task-rest_meg.fif"
]
RESULTS_DIR = Path("/sloth/camcan01884/cc700/meg/pipeline/release005/BIDSsep/MEGnet_results/")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fif_files = []

for p in INPUT_PATHS:

    path = Path(p)

    if path.is_file() and path.suffix == ".fif":
        fif_files.append(path)

    elif path.is_dir():
        fif_files.extend(
            f for f in path.rglob("*_meg.fif")
            if (
                "derivatives" not in f.parts
                and ".git" not in f.parts
                and "sourcedata" not in f.parts
            )
        )

fif_files = sorted(set(fif_files))

print(f"Total FIF files collected: {len(fif_files)}")



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