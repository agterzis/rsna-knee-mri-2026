# ==================== OUR LEG: multi-plane 2.5D models, rank-blended into the public stack ====================
# Runs after the public pipeline has written submission.csv. Any failure here keeps the public
# submission untouched, so the worst case is the public score rather than a broken submission.
import shutil as _sh
import traceback as _tb
from multiprocessing import Pool as _Pool

OURS_W = __OURS_W__          # weight of our leg in the per-label rank blend
OURS_TAGS = __OURS_TAGS__    # checkpoint filename prefixes to load (all folds found are averaged)
_WORK = "/kaggle/working"
_SUB = os.path.join(_WORK, "submission.csv")
_PUB = os.path.join(_WORK, "_public_submission.csv")


def _shallow_glob(pattern):
    """Find files under /kaggle/input without entering the DICOM trees: the attached competition
    data holds ~750k train DICOMs and walking them cost minutes per lookup."""
    import fnmatch
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in fnmatch.filter(files, pattern)]
    return sorted(hits)


def _ours_main():
    t0 = time.time()
    troot = os.path.dirname(_shallow_glob("test_series.csv")[0])
    if not os.path.exists(_SUB):  # standalone run: no public stack, start from the sample file
        _sh.copy2(os.path.join(troot, "sample_submission.csv"), _SUB)
    _sh.copy2(_SUB, _PUB)
    pub = pd.read_csv(_PUB)
    tser = pd.read_csv(os.path.join(troot, "test_series.csv"))
    groups = {u: g.to_dict("records") for u, g in tser.groupby("StudyInstanceUID")}
    uids = pub.StudyInstanceUID.astype(str).tolist()
    ckpts = sorted(p for t in OURS_TAGS for p in _shallow_glob(f"{t}_f*.pt"))
    print(f"[ours] {len(uids)} studies, {len(ckpts)} checkpoints: {[os.path.basename(c) for c in ckpts]}", flush=True)
    assert ckpts, "no checkpoints attached"

    os.makedirs("/kaggle/tmp", exist_ok=True)
    vols = np.lib.format.open_memmap("/kaggle/tmp/ours_vols.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(uids), len(SLOTS), D, RES, RES))
    oks = np.zeros((len(uids), len(SLOTS)), np.float32)

    global _ours_prep

    def _ours_prep(i):
        u = uids[i]
        try:
            arr, meta = process_study(os.path.join(troot, "test_series", u), groups.get(u, []))
        except Exception:
            arr, meta = np.zeros((len(SLOTS), D, RES, RES), np.uint8), {}
        ok = np.array([int(meta.get(f"{n}_ok", 0) == 1) for n in SLOT_NAMES], np.float32)
        return i, arr, ok

    with _Pool(os.cpu_count()) as pool:
        for i, arr, ok in pool.imap_unordered(_ours_prep, range(len(uids)), chunksize=1):
            vols[i], oks[i] = arr, ok
    print(f"[ours] prep done in {(time.time() - t0) / 60:.1f} min; slot coverage {oks.mean(0).round(3).tolist()}",
          flush=True)

    class _TestCorpus:
        index = {u: i for i, u in enumerate(uids)}
        shape = vols.shape[1:]

        def windows(self, u, slot, centers):
            v = vols[self.index[u], slot]
            return np.stack([v[c - 1:c + 2] for c in centers])

        def slot_ok(self, u):
            return oks[self.index[u]]

    preds = []
    for ck in ckpts:
        sd = torch.load(ck, map_location="cpu")
        cfg = sd.get("cfg", {})
        model = Net(sd["arch"], pretrained=False, **cfg)
        model.load_state_dict({k: v.float() for k, v in sd["model"].items()})
        model = model.cuda().eval()
        ds = KneeDS(_TestCorpus(), np.array(uids), np.zeros((len(uids), len(LABELS)), np.float32),
                    False, 0, sd["crop"], cfg.get("eval_stride", 2))
        dl = DataLoader(ds, 2, shuffle=False, num_workers=2, collate_fn=collate)
        preds.append(predict(model, dl, sd["crop"], "cuda"))
        del model
        torch.cuda.empty_cache()
        print(f"[ours] {os.path.basename(ck)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    ours = pd.DataFrame(np.mean(preds, 0), columns=LABELS)
    ours.insert(0, "StudyInstanceUID", uids)
    ours.to_csv(os.path.join(_WORK, "_ours.csv"), index=False)
    assert np.isfinite(ours[LABELS].values).all()

    out = pub.copy()
    for l in LABELS:
        out[l] = (1 - OURS_W) * pub[l].rank(pct=True).values + OURS_W * ours[l].rank(pct=True).values
    out.to_csv(_SUB, index=False)
    print(f"[ours] blended w={OURS_W} -> {_SUB}; total {(time.time() - t0) / 60:.1f} min", flush=True)


try:
    _ours_main()
except Exception:
    _tb.print_exc()
    if os.path.exists(_PUB):
        _sh.copy2(_PUB, _SUB)
    print("[ours] FAILED -> kept public submission", flush=True)
