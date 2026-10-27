"""Build a Kaggle GPU script kernel that runs train.py.

One run per GPU: DataParallel over the two T4s was measured at 136 img/s against 313 img/s for
the same model on one GPU, so separate commands (folds or variants) are pinned to separate GPUs.

  python kernels/train/build.py --slug rsna-knee-train-x --args "--folds 0 --tag a" "--folds 1 --tag b"
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--args", nargs="+", required=True, help="one argument string per GPU (max 2)")
a = ap.parse_args()
assert 1 <= len(a.args) <= 2

launcher = f'''import os, subprocess, sys, threading
ARGS = {[s.split() for s in a.args]!r}
with open("/kaggle/working/train_impl.py", "w") as f:
    f.write(TRAIN_SRC)


def run(gpu, args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.Popen([sys.executable, "-u", "/kaggle/working/train_impl.py", *args], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(f"[gpu{{gpu}}] {{line}}", end="", flush=True)
    p.wait()
    print(f"[gpu{{gpu}}] exit code {{p.returncode}}", flush=True)


threads = [threading.Thread(target=run, args=(g, args)) for g, args in enumerate(ARGS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
os.remove("/kaggle/working/train_impl.py")
'''
d = HERE / a.slug
d.mkdir(exist_ok=True)
src = (HERE / "train.py").read_text()
(d / "train_kernel.py").write_text(f"TRAIN_SRC = {src!r}\n" + launcher)
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"asteriosterzis/{a.slug}", "title": a.slug, "code_file": "train_kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["asteriosterzis/rsna-knee-labels-at", "shigechan/radimagenet-resnet50-official"],
    "kernel_sources": [f"asteriosterzis/rsna-knee-prep-s{k}" for k in range(3)],
    "model_sources": [], "competition_sources": []}, indent=2))
print("built", d, "runs:", a.args)
