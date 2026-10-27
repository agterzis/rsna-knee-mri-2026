"""DICOM -> canonical multi-plane uint8 volumes, shared by the Kaggle prep and inference kernels.

Each study becomes a (SLOTS, D, RES, RES) uint8 array:
  slot 0 SAG_FS  (fallback SAG_NFS)    slot 2 AXI_FS (fallback AXI_NFS)
  slot 1 COR_FS  (fallback COR_NFS)    slot 3 SAG_NFS (fallback SAG_FS second-best)
Canonical orientation (patient LPS axes):
  SAG: columns -> posterior, rows -> inferior, slices ordered lateral -> medial
  COR: columns -> lateral,   rows -> inferior, slices ordered anterior -> posterior
  AXI: columns -> lateral,   rows -> posterior, slices ordered superior -> inferior
"Lateral" is resolved per study from the knee's side, so left knees are mirrored onto right
knees; without this a coronal medial meniscus sits on opposite image sides for the two knees.
"""
import glob, os

import numpy as np
import pydicom
import cv2

SLOTS = [("Sagittal", 1, "Sagittal", 0), ("Coronal", 1, "Coronal", 0),
         ("Axial", 1, "Axial", 0), ("Sagittal", 0, "Sagittal", 1)]
SLOT_NAMES = ["SAG_FS", "COR_FS", "AXI_FS", "SAG_NFS"]
D, RES, CROP_MM = 28, 256, 160.0
# Target direction (LPS) of increasing column index, increasing row index, and slice order,
# for a RIGHT knee. +x = patient left, +y = posterior, +z = superior.
TARGET = {"Sagittal": ((0, 1, 0), (0, 0, -1), (1, 0, 0)),
          "Coronal": ((-1, 0, 0), (0, 0, -1), (0, 1, 0)),
          "Axial": ((-1, 0, 0), (0, 1, 0), (0, 0, -1))}


def pick_series(rows):
    """rows: list of dicts (SeriesInstanceUID, Fluid_Sensitive, Anatomical_Plane) for one study.
    Returns list of (series_uid or None, fallback_used) per slot; a series is used at most once."""
    used, out = set(), []
    for plane, fs, fb_plane, fb_fs in SLOTS:
        pick, fb = None, False
        for want_fs, is_fb in ((fs, False), (fb_fs, True)):
            cands = [r for r in rows if r["Anatomical_Plane"] == plane
                     and int(r["Fluid_Sensitive"]) == want_fs and r["SeriesInstanceUID"] not in used]
            if cands:
                pick, fb = cands[0]["SeriesInstanceUID"], is_fb
                break
        if pick is not None:
            used.add(pick)
        out.append((pick, fb))
    return out


def _headers(sdir):
    recs = []
    for f in glob.glob(os.path.join(sdir, "*.dcm")):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            recs.append((f, h))
        except Exception:
            continue
    return recs


def read_series(sdir):
    """Returns dict(vol=(n,H,W) float32 sorted along the slice normal, r, c, n, ps, pos, lat_tag, cx)."""
    recs = _headers(sdir)
    if not recs:
        return None
    h0 = recs[len(recs) // 2][1]
    iop = np.array(getattr(h0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    items = []
    for f, h in recs:
        ipp = getattr(h, "ImagePositionPatient", None)
        pos = float(np.dot(np.array(ipp, float), n)) if ipp is not None else float(getattr(h, "InstanceNumber", 0))
        items.append((pos, f, h))
    items.sort(key=lambda x: (x[0], x[1]))
    # Several series repeat a position (e.g. multi-echo); keep the first image per position.
    dedup, last = [], None
    for it in items:
        if last is None or abs(it[0] - last) > 1e-3:
            dedup.append(it)
            last = it[0]
    imgs, shape = [], None
    for _, f, _h in dedup:
        try:
            d = pydicom.dcmread(f)
            a = d.pixel_array.astype(np.float32)
            a = a * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0)
            if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
                a = a.max() - a
            if a.ndim == 3:
                a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
        except Exception:
            continue
        if shape is None:
            shape = a.shape
        if a.shape != shape:
            a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        imgs.append(a)
    if not imgs:
        return None
    ps = getattr(h0, "PixelSpacing", None)
    ps = float(ps[0]) if ps is not None else 0.5
    ipp0 = np.array(getattr(dedup[len(dedup) // 2][2], "ImagePositionPatient", [0, 0, 0]), float)
    # Physical centre of the middle slice; its x tells which side of the patient the knee is on.
    cx = float((ipp0 + r * ps * shape[1] / 2 + c * ps * shape[0] / 2)[0])
    lat = None
    for tag in ("Laterality", "ImageLaterality"):
        v = str(getattr(h0, tag, "") or "").strip().upper()[:1]
        if v in ("L", "R"):
            lat = v
            break
    return dict(vol=np.stack(imgs), r=r, c=c, n=n, ps=ps,
                pos=np.array([p for p, _, _ in dedup][:len(imgs)]), lat_tag=lat, cx=cx)


def canonicalize(s, plane, left):
    """Transpose/flip so axes match TARGET[plane]; mirror medial/lateral for left knees."""
    vol, r, c, n = s["vol"], s["r"], s["c"], s["n"]
    tr, tc, ts = (np.array(v, float) for v in TARGET[plane])
    if left:  # mirror x: left knee lateral side is at +x
        tr, tc, ts = tr * [-1, 1, 1], tc * [-1, 1, 1], ts * [-1, 1, 1]
    # vol axes: (slice along n, row along c, col along r)
    if abs(np.dot(r, tr)) + abs(np.dot(c, tc)) < abs(np.dot(c, tr)) + abs(np.dot(r, tc)):
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
    if np.dot(r, tr) < 0:
        vol = vol[:, :, ::-1]
    if np.dot(c, tc) < 0:
        vol = vol[:, ::-1, :]
    pos = s["pos"]
    if np.dot(n, ts) < 0:
        vol, pos = vol[::-1], -pos[::-1]
    return np.ascontiguousarray(vol), pos


def to_fixed(vol, pos, ps):
    """Centre-crop CROP_MM, resize to RES, resample slices to D, robust-scale to uint8."""
    n, h, w = vol.shape
    cpx = int(round(CROP_MM / max(ps, 1e-3)))
    if cpx < min(h, w):
        y0, x0 = (h - cpx) // 2, (w - cpx) // 2
        vol = vol[:, y0:y0 + cpx, x0:x0 + cpx]
    else:  # FOV smaller than the crop: pad to square so pixel size stays isotropic
        s = max(h, w)
        pad = np.zeros((n, s, s), np.float32)
        pad[:, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = vol
        vol = pad
    vol = np.stack([cv2.resize(v, (RES, RES), interpolation=cv2.INTER_AREA) for v in vol])
    if n > 1:
        if n > 2 * D:  # thin-slice 3D series: smooth along z before subsampling
            k = int(round(n / D))
            ker = np.ones(k, np.float32) / k
            vol = np.apply_along_axis(lambda x: np.convolve(x, ker, mode="same"), 0, vol)
        tgt = np.linspace(pos[0], pos[-1], D)
        idx = np.interp(tgt, pos, np.arange(n))
        i0 = np.clip(np.floor(idx).astype(int), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        wgt = (idx - i0)[:, None, None].astype(np.float32)
        vol = vol[i0] * (1 - wgt) + vol[i1] * wgt
    else:
        vol = np.repeat(vol, D, 0)
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8)


def process_study(study_dir, series_rows):
    """Returns (array (4,D,RES,RES) uint8, meta dict)."""
    picks = pick_series(series_rows)
    plane_of = {r["SeriesInstanceUID"]: r["Anatomical_Plane"] for r in series_rows}
    reads = [read_series(os.path.join(study_dir, uid)) if uid else None for uid, _ in picks]
    # Knee side: DICOM tag if any series has it, else the sign of the physical centre's x.
    tags = [s["lat_tag"] for s in reads if s and s["lat_tag"]]
    cxs = [s["cx"] for s in reads if s]
    if tags:
        side, src = max(set(tags), key=tags.count), "tag"
    elif cxs and abs(np.median(cxs)) > 5:
        side, src = ("L" if np.median(cxs) > 0 else "R"), "geom"
    else:
        side, src = "R", "default"
    out = np.zeros((len(SLOTS), D, RES, RES), np.uint8)
    meta = {"side": side, "side_src": src, "cx": float(np.median(cxs)) if cxs else np.nan}
    for i, ((uid, fb), s) in enumerate(zip(picks, reads)):
        name = SLOT_NAMES[i]
        meta[f"{name}_ok"] = int(s is not None)
        meta[f"{name}_fb"] = int(fb)
        if s is None:
            continue
        try:
            vol, pos = canonicalize(s, plane_of[uid], side == "L")
            out[i] = to_fixed(vol, pos, s["ps"])
            meta[f"{name}_n"] = int(vol.shape[0])
            meta[f"{name}_fov"] = float(s["ps"] * max(s["vol"].shape[1:]))
        except Exception as e:
            meta[f"{name}_ok"] = 0
            meta[f"{name}_err"] = repr(e)[:120]
    return out, meta
