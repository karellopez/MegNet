import subprocess
from pathlib import Path

INPUT_PATHS = [
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2002/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2003/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2004/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2005/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2006/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2007/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2008/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2009/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2010/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2011/meg/",
"/sloth/ds_MOUS_decom/ds_MOUS_decompressed/sub-A2012/meg/"
]
RESULTS_DIR = Path("/sloth/ds_MOUS_decom/ds_MOUS_decompressed/MEGnet_results/")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

meg_files = []

for p in INPUT_PATHS:

    path = Path(p)

    # case 1  direct file
    if path.is_file() and path.suffix in [".fif"]:
        meg_files.append(path)

    # case 2  direct CTF dataset
    elif path.is_dir() and path.suffix == ".ds":
        meg_files.append(path)

    # case 3  search inside directory
    elif path.is_dir():
        for f in path.rglob("*"):
            if (
                "derivatives" in f.parts
                or ".git" in f.parts
                or "sourcedata" in f.parts
            ):
                continue

            if f.suffix == ".fif":
                meg_files.append(f)

            if f.suffix == ".ds":
                meg_files.append(f)

meg_files = sorted(set(meg_files))

print(f"Total FIF files collected: {len(meg_files)}")



for meg in meg_files :

    print(f"\nRunning MEGnet on: {meg}")

    cmd = [
        "python",
        "easy_megnet.py",
        "--filename", str(meg),
        "--results-dir", str(RESULTS_DIR),
        "--line-freq", "50",
        "--skip-apply",
        "--run-qc",
        "--run-ref-compare",
    ]

    subprocess.run(cmd)