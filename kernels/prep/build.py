"""Inline knee_prep.py + driver.py into one Kaggle script kernel per shard.

  python kernels/prep/build.py --nshard 3 [--limit 24]   -> kernels/prep/shard{K}/
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--nshard", type=int, default=3)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only", type=int, default=None)
a = ap.parse_args()

src = (HERE / "knee_prep.py").read_text() + "\n\n" + (HERE / "driver.py").read_text()
for k in range(a.nshard):
    if a.only is not None and k != a.only:
        continue
    d = HERE / f"shard{k}"
    d.mkdir(exist_ok=True)
    slug = f"rsna-knee-prep-s{k}" + ("-probe" if a.limit else "")
    code = src.replace("__SHARD__", str(k)).replace("__NSHARD__", str(a.nshard)).replace("__LIMIT__", str(a.limit))
    (d / "prep.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"asteriosterzis/{slug}", "title": slug, "code_file": "prep.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "kernel_sources": [], "model_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
    print("built", d, slug)
