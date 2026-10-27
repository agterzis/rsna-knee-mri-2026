"""Append our leg (prep + 2.5D models + rank blend) to the public anchor notebook.

  python kernels/infer/build.py --slug rsna-knee-blend-v1 --w 0.3 --tags cnxt_t \
      --train-kernels rsna-knee-train-cnxt-f0
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--w", type=float, default=0.3)
ap.add_argument("--tags", default="cnxt_t")
ap.add_argument("--train-kernels", required=True, help="comma-separated kernel slugs holding checkpoints")
a = ap.parse_args()

train_src = (ROOT / "kernels/train/train.py").read_text()
train_src = train_src.split('\nif __name__ == "__main__":')[0]  # definitions only
leg = (HERE / "ours_leg.py").read_text()
leg = leg.replace("__OURS_W__", repr(a.w)).replace("__OURS_TAGS__", repr(a.tags.split(",")))
code = "\n\n".join([(ROOT / "kernels/prep/knee_prep.py").read_text(), train_src, leg])

nb = json.loads((ROOT / "kernels/anchor/rsna-knee-anchor.ipynb").read_text())
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
    "## Our leg\n", "Multi-plane 2.5D models on canonical volumes, rank-blended per label into the public stack."]})
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                    "source": code.splitlines(keepends=True)})
d = HERE / a.slug
d.mkdir(exist_ok=True)
(d / f"{a.slug}.ipynb").write_text(json.dumps(nb))
meta = json.loads((ROOT / "kernels/anchor/kernel-metadata.json").read_text())
meta.update({"id": f"asteriosterzis/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.ipynb"})
meta["kernel_sources"] = meta["kernel_sources"] + [f"asteriosterzis/{k}" for k in a.train_kernels.split(",")]
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built", d, "sources:", meta["kernel_sources"])
