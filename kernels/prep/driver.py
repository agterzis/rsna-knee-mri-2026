# ---- driver: preprocess one shard of the training studies on a Kaggle CPU kernel ----
import time
from multiprocessing import Pool

import pandas as pd

SHARD, NSHARD, LIMIT = __SHARD__, __NSHARD__, __LIMIT__


def find_root():
    for p in glob.glob("/kaggle/input/**/train_series.csv", recursive=True):
        return os.path.dirname(p)
    raise FileNotFoundError("competition data not attached")


ROOT = find_root()
SER = pd.read_csv(os.path.join(ROOT, "train_series.csv"))
GROUPS = {u: g.to_dict("records") for u, g in SER.groupby("StudyInstanceUID")}


def work(uid):
    t0 = time.time()
    try:
        arr, meta = process_study(os.path.join(ROOT, "train_series", uid), GROUPS[uid])
    except Exception as e:
        arr, meta = np.zeros((len(SLOTS), D, RES, RES), np.uint8), {"fatal": repr(e)[:200]}
    meta["sec"] = round(time.time() - t0, 2)
    return uid, arr, meta


if __name__ == "__main__":
    ids = sorted(pd.read_csv(os.path.join(ROOT, "train.csv")).StudyInstanceUID)[SHARD::NSHARD]
    if LIMIT:
        ids = ids[:LIMIT]
    out = np.lib.format.open_memmap(f"/kaggle/working/vols_{SHARD}.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(ids), len(SLOTS), D, RES, RES))
    metas, t0 = [], time.time()
    print(f"shard {SHARD}/{NSHARD}: {len(ids)} studies, {os.cpu_count()} cpus, root={ROOT}", flush=True)
    with Pool(os.cpu_count()) as pool:
        for i, (uid, arr, meta) in enumerate(pool.imap(work, ids, chunksize=1)):
            out[i] = arr
            metas.append({"StudyInstanceUID": uid, **meta})
            if (i + 1) % 50 == 0 or i + 1 == len(ids):
                el = time.time() - t0
                print(f"{i + 1}/{len(ids)} {el / 60:.1f} min, eta {el / (i + 1) * (len(ids) - i - 1) / 60:.1f} min",
                      flush=True)
    out.flush()
    np.save(f"/kaggle/working/ids_{SHARD}.npy", np.array(ids))
    m = pd.DataFrame(metas)
    m.to_csv(f"/kaggle/working/meta_{SHARD}.csv", index=False)
    print(m.describe(include="all").T.to_string(), flush=True)
    print("side_src:", m.side_src.value_counts().to_dict(), "side:", m.side.value_counts().to_dict(), flush=True)
    print("cx quantiles:", m.cx.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict(), flush=True)
    # Previews: middle slice of each slot for the first studies, to check orientation by eye.
    k = min(8, len(ids))
    rows = [np.concatenate([out[j, s, D // 2] for s in range(len(SLOTS))], 1) for j in range(k)]
    cv2.imwrite(f"/kaggle/working/preview_{SHARD}.png", np.concatenate(rows, 0))
    for s in range(len(SLOTS)):  # slice sweep for study 0, slot s
        cv2.imwrite(f"/kaggle/working/sweep_{SHARD}_{SLOT_NAMES[s]}.png",
                    np.concatenate([out[0, s, z] for z in range(0, D, 4)], 1))
