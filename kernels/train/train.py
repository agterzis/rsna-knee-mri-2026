"""Train a multi-plane 2.5D model on the canonical volumes from the prep kernels.

Each study is (4 slots, 28 slices, 256, 256) uint8. Three neighbouring slices form one RGB
window; windows from all four planes go through a shared 2D backbone, get a slot embedding,
and are pooled by a separate attention per finding (an ACL tear lives on two sagittal slices,
OA spreads across many coronal ones, so one shared pooling makes the findings compete).
"""
import argparse, glob, json, math, os, random, shutil, time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_NAMES = ["SAG_FS", "COR_FS", "AXI_FS", "SAG_NFS"]
NSLOT, D = 4, 28


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits


class Corpus:
    """All prep shards, read with pread instead of mmap: mapped pages of a 32 GB corpus count
    against the 30 GB container limit and got the kernel OOM-killed, plain reads do not."""
    def __init__(self, local=False):
        self.files, self.index, self.ok = [], {}, {}
        for vp in find("vols_*.npy"):
            if local:
                vp = self._copy_local(vp)
            k = vp.rsplit("_", 1)[1][:-4]
            base = os.path.dirname(vp)
            with open(vp, "rb") as f:
                ver = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, ver)
                off = f.tell()
            self.files.append((vp, off))
            self.shape = shape[1:]                      # (slots, D, R, R)
            ids = np.load(os.path.join(base, f"ids_{k}.npy"), allow_pickle=True)
            meta = pd.read_csv(os.path.join(base, f"meta_{k}.csv")).set_index("StudyInstanceUID")
            for j, u in enumerate(ids):
                u = str(u)
                self.index[u] = (len(self.files) - 1, j)
                self.ok[u] = np.array([float(meta.at[u, f"{n}_ok"] == 1) if f"{n}_ok" in meta else 0.0
                                       for n in SLOT_NAMES], np.float32)
        self.fds = {}
        print(f"corpus: {len(self.index)} studies from {len(self.files)} shards", flush=True)

    @staticmethod
    def _copy_local(src):
        """The input mount reads ~40 MB/s at random; the local disk is >10x faster. Copy once,
        dropping the page cache as we go so the copy itself does not count toward the RAM limit."""
        dst_dir = "/kaggle/local_vols"
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        k = src.rsplit("_", 1)[1][:-4]
        for small in (f"ids_{k}.npy", f"meta_{k}.csv"):
            shutil.copy(os.path.join(os.path.dirname(src), small), dst_dir)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return dst
        t0, chunk = time.time(), 64 << 20
        fi, fo = os.open(src, os.O_RDONLY), os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        pos = 0
        while True:
            b = os.pread(fi, chunk, pos)
            if not b:
                break
            os.write(fo, b)
            pos += len(b)
            if pos % (1 << 30) < chunk:
                os.fsync(fo)
                os.posix_fadvise(fi, 0, pos, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(fo, 0, pos, os.POSIX_FADV_DONTNEED)
        os.fsync(fo)
        os.posix_fadvise(fo, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fi), os.close(fo)
        print(f"copied {os.path.basename(src)} ({pos / 2**30:.1f} GiB) in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        return dst

    def _fd(self, s):
        key = (os.getpid(), s)                          # one descriptor per worker process
        if key not in self.fds:
            self.fds[key] = os.open(self.files[s][0], os.O_RDONLY)
        return self.fds[key]

    def windows(self, uid, slot, centers):
        """(len(centers), 3, R, R) uint8: slices c-1, c, c+1 read as one contiguous block each."""
        s, j = self.index[uid]
        S, Dd, R, _ = self.shape
        sl = R * R
        fd, off = self._fd(s), self.files[s][1]
        out = np.empty((len(centers), 3, R, R), np.uint8)
        for i, c in enumerate(centers):
            b = os.pread(fd, 3 * sl, off + ((j * S + slot) * Dd + c - 1) * sl)
            out[i] = np.frombuffer(b, np.uint8).reshape(3, R, R)
        return out

    def slot_ok(self, uid):
        return self.ok[uid]


class KneeDS(Dataset):
    def __init__(self, corpus, ids, y, train, k, crop, eval_stride=2):
        self.c, self.ids, self.y, self.train, self.k, self.crop = corpus, ids, y, train, k, crop
        self.eval_centers = list(range(1, D - 1, eval_stride))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        ok = self.c.slot_ok(uid)
        res = self.c.shape[-1]
        wins, slots, pos = [], [], []
        for s in range(NSLOT):
            if self.train:
                cs = sorted(random.sample(range(1, D - 1), self.k))
                sz = random.randint(int(self.crop * 0.85), min(res, int(self.crop * 1.15)))
                y0, x0 = random.randint(0, res - sz), random.randint(0, res - sz)
            else:
                cs = self.eval_centers
                sz = self.crop
                y0 = x0 = (res - sz) // 2
            w = self.c.windows(uid, s, cs)[:, :, y0:y0 + sz, x0:x0 + sz]
            wins.append(torch.from_numpy(np.ascontiguousarray(w)))
            slots.append(torch.full((len(cs),), s, dtype=torch.long))
            pos.append(torch.tensor(cs, dtype=torch.long))
        # Crop sizes differ per slot in training; they are resized to `crop` on the GPU.
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return wins, torch.cat(slots), torch.cat(pos), torch.from_numpy(ok), y


def collate(batch):
    return batch


def load_radimagenet(bb):
    """RadImageNet ResNet50 (torchvision layout, children()[:-2] under 'backbone.') -> timm resnet50."""
    path = find("ResNet50.pt")[0]
    sd = torch.load(path, map_location="cpu", weights_only=True)
    names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
    out = {}
    for k, v in sd.items():
        parts = k.split(".")
        if parts[0] == "backbone":
            parts = parts[1:]
        out[".".join([names[parts[0]]] + parts[1:])] = v
    missing, unexpected = bb.load_state_dict(out, strict=False)
    assert not unexpected and all(m.startswith("fc") for m in missing), (missing, unexpected)
    print(f"loaded RadImageNet weights from {path}", flush=True)


class Net(nn.Module):
    """pool='global': one attention over every window of every plane (v1).
    pool='slot': attention within each plane, then a per-finding, per-plane linear read-out, so a
    finding seen on two sagittal slices does not have to win against fifty other windows.
    pos=True adds a learned embedding of the window's slice index (e.g. central sagittal = ACL)."""
    def __init__(self, arch, pretrained=True, drop=0.2, pool="global", pos=False, init="imagenet", **_):
        super().__init__()
        self.bb = timm.create_model(arch, pretrained=pretrained and init == "imagenet", num_classes=0)
        if pretrained and init == "radimagenet":
            load_radimagenet(self.bb)
        f = self.bb.num_features
        self.pool, self.use_pos = pool, pos
        self.slot_emb = nn.Embedding(NSLOT, f)
        if pos:
            self.pos_emb = nn.Embedding(D, f)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.attn = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Linear(256, len(LABELS)))
        self.drop = nn.Dropout(drop)
        shape = (len(LABELS), NSLOT, f) if pool == "slot" else (len(LABELS), f)
        self.head_w = nn.Parameter(torch.randn(*shape) * 0.01)
        self.head_b = nn.Parameter(torch.zeros(len(LABELS)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x, slot, pos, mask):
        """x (N,3,H,W) 0-255 for all windows of the batch, slot/pos (N,), mask (B,Nw) bool.
        Windows of a study are laid out slot by slot with the same count per slot."""
        x = ((x.float() - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        h = self.bb(x)
        h = h + self.slot_emb(slot)
        if self.use_pos:
            h = h + self.pos_emb(pos)
        B, Nw = mask.shape
        h = h.view(B, Nw, -1)
        a = self.attn(h).float().masked_fill(~mask[..., None], -1e4)
        if self.pool == "global":
            w = a.softmax(1)                               # (B,Nw,12)
            pooled = torch.einsum("bnl,bnf->blf", w, h.float())
            return (self.drop(pooled) * self.head_w).sum(-1) + self.head_b
        n = Nw // NSLOT
        w = a.view(B, NSLOT, n, -1).softmax(2)             # attention within each plane
        pooled = torch.einsum("bsnl,bsnf->bslf", w, h.float().view(B, NSLOT, n, -1))
        present = mask.view(B, NSLOT, n).any(2).float()    # a missing plane contributes nothing
        pooled = self.drop(pooled) * present[:, :, None, None]
        return torch.einsum("bslf,lsf->bl", pooled, self.head_w) + self.head_b


def to_batch(batch, crop, device):
    xs, sl, ps, masks, ys = [], [], [], [], []
    for wins, slots, pos, ok, y in batch:
        per = []
        for s, w in enumerate(wins):
            w = w.to(device, non_blocking=True)
            if w.shape[-1] != crop:
                w = F.interpolate(w.float(), size=(crop, crop), mode="bilinear", align_corners=False)
            per.append(w.float())
        xs.append(torch.cat(per))
        sl.append(slots)
        ps.append(pos)
        okw = ok[slots]                                    # missing slots are masked out
        if okw.sum() == 0:
            okw[:] = 1
        masks.append(okw.bool())
        ys.append(y)
    return (torch.cat(xs), torch.cat(sl).to(device), torch.cat(ps).to(device),
            torch.stack(masks).to(device), torch.stack(ys).to(device))


def augment(x):
    """Per-window brightness/contrast/gamma jitter on 0-255 floats (no flips: side is signal)."""
    n = x.shape[0]
    g = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    b = torch.empty(n, 1, 1, 1, device=x.device).uniform_(-12, 12)
    gam = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = ((x / 255).clamp(0, 1) ** gam) * 255
    return (x * g + b).clamp(0, 255)


def macro_auc(y, p, hard=False):
    out = {}
    for j, l in enumerate(LABELS):
        t = (y[:, j] > 0.5).astype(int) if not hard else y[:, j].astype(int)
        if 0 < t.sum() < len(t):
            out[l] = roc_auc_score(t, p[:, j])
    return float(np.mean(list(out.values()))) if out else float("nan"), out


@torch.no_grad()
def predict(model, dl, crop, device):
    model.eval()
    ps = []
    for batch in dl:
        x, sl, pos, m, _ = to_batch(batch, crop, device)
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(model(x, sl, pos, m).float()).cpu().numpy())
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50.a1_in1k")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--k", type=int, default=8, help="windows per slot in training")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--labels", default="labels_v1.csv")
    ap.add_argument("--tag", default="r50")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--local", action="store_true", help="copy the corpus to local disk first")
    ap.add_argument("--hours", type=float, default=11.0, help="stop starting epochs after this")
    ap.add_argument("--pool", default="global", choices=["global", "slot"])
    ap.add_argument("--pos", action="store_true", help="slice-position embedding")
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "radimagenet"])
    ap.add_argument("--eval_stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t_start = time.time()
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    corpus = Corpus(local=a.local)
    lab = pd.read_csv(find(a.labels)[0]).set_index("StudyInstanceUID")
    ids = np.array([u for u in lab.index if u in corpus.index])
    y = lab.loc[ids, LABELS].values.astype(np.float32)
    gold = lab.loc[ids, "is_gold"].values.astype(bool)
    print(f"{len(ids)} labelled studies with volumes, {gold.sum()} gold", flush=True)
    folds = list(KFold(a.nfold, shuffle=True, random_state=42).split(ids))
    os.makedirs("/kaggle/working/out", exist_ok=True)
    for f in [int(v) for v in a.folds.split(",")]:
        tr, va = folds[f]
        if a.smoke:
            tr, va = tr[:48], va[:24]
        ds_tr = KneeDS(corpus, ids[tr], y[tr], True, a.k, a.crop)
        ds_va = KneeDS(corpus, ids[va], y[va], False, a.k, a.crop, a.eval_stride)
        dl_tr = DataLoader(ds_tr, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                           drop_last=True, persistent_workers=True, prefetch_factor=4)
        dl_va = DataLoader(ds_va, 2, shuffle=False, num_workers=a.workers, collate_fn=collate)
        cfg = {"pool": a.pool, "pos": a.pos, "init": a.init, "eval_stride": a.eval_stride}
        model = Net(a.arch, **cfg).to(device).to(memory_format=torch.channels_last)
        if torch.cuda.device_count() > 1:
            model.bb = nn.DataParallel(model.bb)
        bb_params = list(model.bb.parameters())
        head_params = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
        opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr},
                                 {"params": head_params, "lr": a.lr * 10}], weight_decay=0.05)
        steps = a.epochs * len(dl_tr)
        warm = len(dl_tr) // 2
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
        scaler = torch.amp.GradScaler("cuda")
        best, hist = -1, []
        for ep in range(a.epochs):
            if (time.time() - t_start) / 3600 > a.hours:
                print("time budget reached, stopping", flush=True)
                break
            model.train()
            t0, tl, n = time.time(), 0.0, 0
            for it, batch in enumerate(dl_tr):
                x, sl, pos, m, yy = to_batch(batch, a.crop, device)
                x = augment(x)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x, sl, pos, m)
                loss = F.binary_cross_entropy_with_logits(out.float(), yy)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl, n = tl + loss.item(), n + 1
                if it % 100 == 0:
                    print(f"  ep{ep} it{it}/{len(dl_tr)} loss {tl / n:.4f} {(time.time() - t0) / 60:.1f}m", flush=True)
            p = predict(model, dl_va, a.crop, device)
            auc, _ = macro_auc(y[va], p)
            g = gold[va]
            gauc = macro_auc(y[va][g], p[g], hard=True)[0] if g.sum() >= 5 else float("nan")
            hist.append({"ep": ep, "loss": tl / n, "val_auc_llm": auc, "val_auc_gold": gauc,
                         "min": (time.time() - t0) / 60})
            print(f"fold{f} ep{ep} loss {tl / n:.4f} val_auc(llm labels) {auc:.4f} gold({g.sum()}) {gauc:.4f} "
                  f"{(time.time() - t0) / 60:.1f}m", flush=True)
            if auc > best:
                best = auc
                sd = {k.replace("bb.module.", "bb."): v.half().cpu() for k, v in model.state_dict().items()}
                torch.save({"model": sd, "arch": a.arch, "crop": a.crop, "labels": LABELS, "cfg": cfg},
                           f"/kaggle/working/out/{a.tag}_f{f}.pt")
                oof = pd.DataFrame(p, columns=LABELS, index=ids[va])
                oof.index.name = "StudyInstanceUID"
                oof.to_csv(f"/kaggle/working/out/{a.tag}_oof_f{f}.csv")
        json.dump(hist, open(f"/kaggle/working/out/{a.tag}_hist_f{f}.json", "w"), indent=1)
        print(f"fold{f} best val_auc(llm) {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
