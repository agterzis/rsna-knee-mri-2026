# RSNA 2026 Knee MRI Abnormality Detection

My entry to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle challenge (July–October 2026). The task: given a knee MRI exam (several DICOM series), predict the probability of 12 findings. These are ACL and MCL tears, medial and lateral meniscus tears, osteoarthritis in each of the three compartments, effusion, synovitis, Baker's cyst, bone contusion and fracture. Submissions are scored by macro-averaged ROC AUC.

The catch: only **58 of 4,407** training studies come with expert labels. Every other study comes with its free-text radiology report, in about nine languages. So most of the work is turning reports into training labels, and then teaching an image model to beat those labels.

## Results

| Submission | Public LB | Private LB |
|---|---:|---:|
| Own multi-plane ResNet50, single fold | 0.881 | 0.870 |
| Public ensemble (Oct 3 snapshot) | 0.943 | 0.927 |
| **Final: public R6/Raptor rank blend** | **0.958** | **0.942** |

Final standing: **903rd of 6,300 teams (top 15%)** on the private leaderboard.

The final submissions are forks of community ensembles (credited below). My own model was not strong enough to help them in a blend. This repository holds the parts I built myself.

## What's here

```
scripts/
  label_reports.py    LLM report reader: graded findings (not mentioned / normal / below threshold /
                      equivocal / positive / severe) per finding, multilingual, resumable
  label_v2.py         budget version: 8 reports per request, compact schema, exact cost accounting
  grades_to_labels.py graded output -> soft labels, scored against the 58 expert-labelled studies
kernels/
  prep/      DICOM -> canonical multi-plane volumes (4 planes x 28 slices x 256 px, uint8)
  train/     multi-plane 2.5D model + Kaggle kernel builder (one training run per GPU)
  infer/     test-time pipeline that rank-blends our model into a base notebook
  bench/     backbone / data-loader throughput benchmark for Kaggle T4s
```

## Approach

**1. Labels from reports.** An LLM (Gemini) reads each report and grades every finding against the challenge's annotation criteria. It answers with grades rather than yes/no: *not mentioned*, *normal*, *below threshold*, *equivocal*, *positive* or *severe*. Two things mattered more than which model did the reading:
- Findings the report calls "small" or "mild" turned out to be positive about half the time in the expert labels. The annotators read the images and are more liberal than the report's wording, so *below threshold* has to rank above silence.
- Silence means different things for different findings. An unmentioned Baker's cyst is almost never there. Synovitis is rarely written down even when present, so silent synovitis is better imputed from effusion.

Every labeller variant I tried (Flash, Pro, Flash-Lite, different thinking levels) landed between 0.85 and 0.89 macro AUC against the 58 expert studies. That is within the bootstrap noise of the public label sets. The reports themselves cap label quality, so I kept API spend under $5 by packing reports per request, using a compact output schema and limiting thinking.

**2. Canonical multi-plane volumes.** Each study is reduced to four slots: sagittal fluid-sensitive, coronal fluid-sensitive, axial fluid-sensitive and sagittal non-fat-suppressed, with fallbacks when one is missing. Each series is:
- re-oriented from `ImageOrientationPatient` onto a fixed anatomical frame,
- cropped to 160 mm around the centre,
- resampled to 28 slices and robust-scaled to uint8.

Left knees are mirrored onto right knees, so "medial" is always on the same image side. The side comes from the DICOM laterality tag, or from the sign of the volume's physical x-coordinate when the tag is missing. The geometry rule agrees with the tag on 98.3% of the studies that have both.

**3. Model.** A 2.5D design:
- Three neighbouring slices form one RGB window, and every window from every plane goes through a shared 2D backbone.
- Windows get plane and slice-position embeddings.
- A separate attention pool **per plane and per finding** lets an ACL tear visible on two sagittal slices avoid competing with fifty other windows.

Starting from RadImageNet ResNet50 weights and moving to per-plane pooling raised validation AUC from 0.849 to 0.875. The biggest gains were on the lateral meniscus (+0.10) and the MCL (+0.06).

## Engineering notes (Kaggle 2×T4)

Things I measured that cost me GPU hours before I measured them:

- **`nn.DataParallel` was slower than a single GPU**: 136 vs 313 img/s for the same model. Running one independent process per GPU (one fold or one variant each) roughly quadrupled throughput.
- **Depthwise-conv backbones (ConvNeXt, EfficientNet) ran an order of magnitude slower than ResNet50** on the T4 image I used. ConvNeXt-tiny trained at ~32 img/s where ResNet50 managed several hundred.
- **`np.memmap` over a 32 GB corpus got the kernel OOM-killed.** Mapped page cache counts against the 30 GB container limit. Plain `os.pread` reads fixed it.
- **A recursive `glob('/kaggle/input/**')` at inference walked ~750k training DICOMs** and cost minutes per lookup. Pruning those directories fixed it.
- **The public leaderboard is ~30% of the test set.** Its 0.001 steps are smaller than its noise, and my 0.958 → 0.942 public-to-private drop shows how hard a crowded score block is to read from the public half.

## Running it

The competition data, the radiology reports and anything derived from them (labels, volumes, predictions) are **not included**: the competition rules do not allow redistributing them. To reproduce:

1. Join the competition and download `train.csv` / `train_series.csv` into `data/`.
2. Labels: `GEMINI_API_KEY=... python scripts/label_v2.py eval` scores a labeller on the 58 expert studies, `run --subset all` labels the full corpus. `python scripts/grades_to_labels.py` maps the grades to soft labels.
3. Volumes: `python kernels/prep/build.py --nshard 3`, then `kaggle kernels push -p kernels/prep/shard0` (and shards 1 and 2). These are CPU kernels, ~20 minutes each.
4. Training: `python kernels/train/build.py --slug <name> --args "<run on GPU 0>" "<run on GPU 1>"`, then push. See `train.py --help` for the model options (`--pool slot --pos --init radimagenet`).
5. Inference: `kernels/infer/build_solo.py` builds a standalone submission notebook. `kernels/infer/build.py` appends our model to a base notebook that you provide.

The `kernel-metadata.json` files the builders write point at my Kaggle username. Change `asteriosterzis` to yours.

```
pip install -r requirements.txt
```

## Acknowledgements

The final submissions are forks of public work by **heliosli** (R6/R7 ConvNeXt and blend recipe), **nartaa** (Raptor CoAtNet 384/224 checkpoints), **goodpjw2008** (2.5D reader), **jiweiliu**, **mattiaangeli** and **dreaddevelopment** (the earlier public stack and Raptor recipe). Public report-label sets from **stevenleehans**, **pilkwang** and **lixin73** were used for training labels. RadImageNet ResNet50 weights from [Lab-Rasool/RadImageNet](https://huggingface.co/Lab-Rasool/RadImageNet).

## License

MIT for the code in this repository. Third-party models and data remain under their own licenses.
