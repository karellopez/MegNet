# MEGnet Wrapper Tutorial

This workspace includes `easy_megnet.py`, a wrapper around `MEGnet-neuro` (`MEGnet` package) for:

- preprocessing + ICA
- IC labeling
- optional ICA application
- optional QC plotting
- optional IC vs ECG/EOG comparison plotting

## 1) Verify environment

```bash
.venv/bin/python -c "import MEGnet; print(MEGnet.__version__, MEGnet.__path__[0])"
.venv/bin/megnet_init
```

## 2) Minimal run

```bash
.venv/bin/python easy_megnet.py \
  --filename /ABS/PATH/TO/recording_meg.fif \
  --results-dir /ABS/PATH/TO/MEGnet_results \
  --line-freq 50
```

## 3) Full run example

```bash
.venv/bin/python easy_megnet.py \
  --filename /ABS/PATH/TO/recording_meg.fif \
  --results-dir /ABS/PATH/TO/MEGnet_results \
  --line-freq 50 \
  --skip-apply \
  --run-qc \
  --run-ref-compare
```

## 4) Flag Reference (Every Flag)

1. `--filename`: input MEG dataset path (`.fif`, `.ds`, `.sqd`, BTI). Example: `--filename /data/sub-001_run-1_meg.fif`
2. `--results-dir`: parent output directory. Example: `--results-dir /data/MEGnet_results`
3. `--line-freq`: mains frequency (`50` or `60`). Example: `--line-freq 60`
4. `--filename-raw`: optional non-SSS raw file (mainly for MEGIN bad-channel logic). Example: `--filename-raw /data/sub-001_raw.fif`
5. `--outbasename`: force output subfolder name. Example: `--outbasename sub-001_task-rest_run-1`
6. `--bad-channels`: comma-separated channels to drop before ICA. Example: `--bad-channels MEG0113,MEG2443`
7. `--classify-only`: skip preprocessing/ICA creation and only classify existing MEGnet outputs in the results folder. Example: `--classify-only`
8. `--skip-init`: skip `megnet_init` model check/download. Example: `--skip-init`
9. `--skip-apply`: do not apply/remove predicted bad ICs from raw data. Example: `--skip-apply`
10. `--run-qc`: run QC plotting after classification. Example: `--run-qc`
11. `--qc-apply-filter`: apply visualization filter during QC. Example: `--run-qc --qc-apply-filter`
12. `--qc-block`: use blocking plot behavior for QC. Example: `--run-qc --qc-block`
13. `--run-ref-compare`: generate IC vs ECG/EOG comparison plots. Example: `--run-ref-compare`
14. `--compare-max-seconds`: seconds shown per comparison trace. Example: `--run-ref-compare --compare-max-seconds 45`
15. `--compare-out-dir`: custom folder for comparison plots. Example: `--run-ref-compare --compare-out-dir /data/my_compare_plots`
16. `--report-file`: custom JSON report output path. Example: `--report-file /data/my_report.json`

## 5) Outputs

Main outputs are created in:

```text
<results-dir>/<file_base>/
```

Core files:

- `ICATimeSeries.mat`
- `component1.mat` to `component20.mat`
- `component1.png` to `component20.png`
- `<file_base>_0-ica.fif`
- `<file_base>_250srate_meg.fif`
- `megnet_summary.json`
- `ica_clean.fif` and `<file_base>_0-ica_applied.fif` (unless `--skip-apply`)

Reference comparison outputs (`--run-ref-compare`):

- folder: `IC_ref_comparisons/`
- ocular file: `IC_ocular_combined_all_EOG.png` with separate panels:
  - panel(s) for selected ocular ICs (`vEOG` and/or `hEOG`)
  - one panel for each available real EOG channel
- cardiac files: one file per cardiac IC when ECG exists
- IC-only fallback:
  - if ECG is missing, cardiac plots are IC-only
  - if EOG is missing, ocular plot is IC-only
- CSV summary: `comparison_summary.csv`

QC outputs (`--run-qc`):

- folder: `MEGnetExtPlots/<data_stem>/`
- the wrapper first tries MEGnet QC directly
- if MEGnet QC fails in the environment, wrapper automatically writes fallback QC plots

## 6) Class IDs

- `0`: Neural/other
- `1`: Eye blink (`vEOG`)
- `2`: Cardiac (`ECG/EKG`)
- `3`: Horizontal eye movement (`hEOG`/saccade)

## 7) Direct package CLIs (optional)

```bash
.venv/bin/ICA.py --help
.venv/bin/python -m MEGnet.megnet_qc_plots --help
```
