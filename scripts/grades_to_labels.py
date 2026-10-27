"""Map graded LLM output (labels/raw/<model>.jsonl) to soft labels and score vs the 58 gold.

The mapping is fixed a priori from the host's criteria (borderline -> negative); it is
NOT tuned on the 58 gold studies, which are only used for reporting.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from label_reports import LABELS, KEYS  # noqa: E402

GRADE_P = {"normal": 0.02, "below_threshold": 0.45, "equivocal": 0.55,
           "positive": 0.85, "positive_severe": 0.97}
# Reports grade severity more strictly than the image-based gold: a "small" or "mild"
# finding in the report is positive about half the time, so below_threshold sits well above
# silence. What silence means also differs per finding: a radiologist who does not mention a Baker's cyst
# almost never missed one, but synovitis is rarely written down even when present.
SILENCE_P = {"acl": 0.05, "mcl": 0.05, "medial_meniscus": 0.08, "lateral_meniscus": 0.08,
             "medial_oa": 0.06, "lateral_oa": 0.06, "pf_oa": 0.10, "effusion": 0.10,
             "synovitis": None, "bakers_cyst": 0.04, "bone_contusion": 0.06, "fracture": 0.04}


V2_GRADE = {"N": "normal", "LOW": "below_threshold", "EQ": "equivocal", "POS": "positive",
            "SEV": "positive_severe", "NM": "not_mentioned"}


def load(model):
    rows = [json.loads(l) for l in (ROOT / "labels/raw" / f"{model}.jsonl").open()]
    out = {}
    for r in rows:
        p = {}
        for k, lab in zip(KEYS, LABELS):
            if isinstance(r[k], list):  # v2: [grade, severity 0-4]; severity breaks ties within a grade
                g, sev = V2_GRADE[r[k][0]], r[k][1]
            else:
                g, sev = r[k]["grade"], 0
            p[lab] = SILENCE_P[k] if g == "not_mentioned" else GRADE_P[g] + 0.02 * sev
        # Silent synovitis: fall back to effusion, which co-occurs with it far more often
        # than the synovitis field is filled in.
        syn = r["synovitis"]
        if (syn[0] if isinstance(syn, list) else syn["grade"]) in ("NM", "not_mentioned"):
            p["Synovitis"] = 0.1 + 0.6 * p["Effusion"]
        out[r["StudyInstanceUID"]] = p
    return pd.DataFrame.from_dict(out, orient="index")[LABELS]


def score(df, gold):
    d = df.reindex(gold.index)
    r = {l: roc_auc_score(gold[l], d[l].fillna(0.5)) for l in LABELS}
    r["MEAN"] = np.mean(list(r.values()))
    return r


if __name__ == "__main__":
    t = pd.read_csv(ROOT / "data/train.csv")
    gold = t.dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS]
    models = sys.argv[1:] or [p.stem for p in sorted((ROOT / "labels/raw").glob("*.jsonl"))]
    res, frames = {}, {}
    for m in models:
        frames[m] = load(m)
        res[m] = score(frames[m], gold)
    sv = pd.read_csv(ROOT / "labels/public/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv")
    sv = sv.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[LABELS]
    res["public_steven_v4"] = score(sv, gold)
    if len(frames) > 1:
        res["ours_mean"] = score(sum(frames.values()) / len(frames), gold)
    for m in frames:
        res[f"{m}+steven"] = score((frames[m].rank(pct=True) + sv.reindex(frames[m].index).rank(pct=True)) / 2, gold)
    print(pd.DataFrame(res).T.round(3).to_string())
