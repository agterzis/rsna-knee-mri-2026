import glob, os, time, subprocess
import numpy as np, torch, torch.nn as nn, timm
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc", shell=True, capture_output=True, text=True).stdout, flush=True)
# ---- A: loader throughput with pread (no mmap) ----
vp = sorted(glob.glob("/kaggle/input/**/vols_*.npy", recursive=True))[0]
with open(vp, "rb") as f:
    ver = np.lib.format.read_magic(f); shape, _, _ = np.lib.format._read_array_header(f, ver); off = f.tell()
N, S, D, R, _ = shape; SL = R * R
fd = os.open(vp, os.O_RDONLY); rng = np.random.default_rng(0)
def study(i):
    out = []
    for s in range(S):
        z = np.sort(rng.choice(np.arange(1, D - 1), 8, replace=False))
        for zz in np.stack([z - 1, z, z + 1], 1).ravel():
            b = os.pread(fd, SL, off + ((i * S + s) * D + int(zz)) * SL)
            out.append(np.frombuffer(b, np.uint8))
    return out
t = time.time(); n = 30
for _ in range(n): study(int(rng.integers(N)))
print(f"pread loader: {(time.time()-t)/n:.3f} s/study (single process)", flush=True)
# ---- B: model throughput, fwd+bwd fp16, 192 imgs at 224 over 2 GPUs ----
for arch in ["convnext_tiny", "convnext_nano", "resnet50", "tf_efficientnetv2_s", "efficientnet_b2", "coatnet_0_rw_224", "resnet34", "regnety_016"]:
    try:
        m = nn.DataParallel(timm.create_model(arch, pretrained=False, num_classes=12).cuda().to(memory_format=torch.channels_last))
        opt = torch.optim.AdamW(m.parameters(), 1e-4); sc = torch.cuda.amp.GradScaler()
        x = torch.randn(192, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
        for it in range(8):
            if it == 3: torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(x).float().mean()
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
        torch.cuda.synchronize(); dt = (time.time() - t) / 5
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5): m(x)
            torch.cuda.synchronize(); di = (time.time() - t) / 5
        print(f"{arch:22s} train {dt:.2f} s/192img ({192/dt:.0f} img/s)  infer {192/di:.0f} img/s  mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(arch, "FAILED", repr(e)[:150], flush=True); torch.cuda.empty_cache()
