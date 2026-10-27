"""Standalone kernel: only our leg (w=1), no public stack. Fast to score our models alone.

  python kernels/infer/build_solo.py --slug rsna-knee-solo-r50 --tags r50 --train-kernels rsna-knee-train-r50-f0
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--tags", default="r50")
ap.add_argument("--train-kernels", required=True)
a = ap.parse_args()

train_src = (ROOT / "kernels/train/train.py").read_text().split('\nif __name__ == "__main__":')[0]
leg = (HERE / "ours_leg.py").read_text().replace("__OURS_W__", "1.0").replace("__OURS_TAGS__", repr(a.tags.split(",")))
code = "\n\n".join([(ROOT / "kernels/prep/knee_prep.py").read_text(), train_src, "import time\n" + leg])
d = HERE / a.slug
d.mkdir(exist_ok=True)
(d / "solo.py").write_text(code)
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"asteriosterzis/{a.slug}", "title": a.slug, "code_file": "solo.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "model_sources": [],
    "kernel_sources": [f"asteriosterzis/{k}" for k in a.train_kernels.split(",")],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built", d)
