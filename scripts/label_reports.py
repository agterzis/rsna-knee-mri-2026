"""Extract graded findings from radiology reports with Gemini.

Usage:
  python scripts/label_reports.py --model gemini-3.8-flash --subset gold
  python scripts/label_reports.py --model gemini-3.8-flash --subset all

Raw graded JSON is cached in labels/raw/<model>.jsonl (one line per study) so runs
resume and the grade->probability mapping can be changed without re-querying.
"""
import argparse, asyncio, json, os, random, ssl
from pathlib import Path

import aiohttp
import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa",
        "pf_oa", "effusion", "synovitis", "bakers_cyst", "bone_contusion", "fracture"]
GRADES = ["not_mentioned", "normal", "below_threshold", "equivocal", "positive", "positive_severe"]

SYSTEM = """You are a musculoskeletal radiologist extracting structured findings from a knee MRI report.
The report may be in any language (English, German, Spanish, Portuguese, French, Italian, Dutch, Turkish, Chinese, ...).
Read the whole report (findings AND impression) and grade each of 12 findings for the imaged knee.

Grade scale (use exactly one per finding):
- not_mentioned: the report says nothing that lets you judge this finding (no direct or clearly implied statement).
- normal: explicitly normal/intact/absent, or clearly implied normal (e.g. "menisci intact", "no fracture", "ligaments unremarkable", "no joint effusion").
- below_threshold: abnormal but BELOW the challenge criteria (see definitions), e.g. sprain/low-grade/degeneration, small/trace/physiologic effusion, small cyst, mild/low-grade or focal chondral thinning, intrasubstance meniscal degeneration not reaching the surface, chronic/old healed findings.
- equivocal: the report itself is uncertain ("possible", "suspected", "cannot exclude", "likely", "Verdacht auf", "V.a.") about a finding that would meet criteria.
- positive: meets the challenge criteria.
- positive_severe: clearly and unambiguously meets the criteria with high-grade features (complete/full-thickness tear, displaced/bucket-handle tear, large effusion, large cyst, full-thickness cartilage loss/bone-on-bone, displaced fracture).

Challenge criteria (annotating radiologists graded borderline cases as NEGATIVE):
- acl: high-grade partial (>50% fibers) or complete ACL tear. Mild signal change, mucoid degeneration, thickening, low-grade partial tear, intact graft -> below_threshold or normal. Torn ACL graft counts as positive.
- mcl: high-grade partial or complete ACUTE MCL tear (grade 2-3 with fiber disruption and edema). Low-grade sprain (grade 1), periligamentous edema only, chronic/remote changes (Pellegrini-Stieda, thickening) -> below_threshold.
- medial_meniscus / lateral_meniscus: tear = signal reaching the articular surface, or morphologic abnormality (truncated, diminutive, displaced fragment, flap, bucket-handle, root tear, maceration, extrusion with tear). Intrasubstance degeneration (grade 1-2) not reaching the surface -> below_threshold. Post-meniscectomy remnant without new tear -> below_threshold unless a tear is described.
- medial_oa / lateral_oa / pf_oa: moderate-to-large area (~1 cm or more) of HIGH-grade cartilage loss (>50% thickness, grade 3-4, full-thickness, exposed bone), +/- subchondral edema/cysts/osteophytes in that compartment. Mild/low-grade chondromalacia, grade 1-2, superficial fissuring, small focal defect -> below_threshold. "Moderate/advanced/severe osteoarthritis" or "tricompartmental OA" with cartilage loss in that compartment -> positive. Patellofemoral = patella + trochlea.
- effusion: MODERATE or LARGE joint effusion. Small/trace/minimal/physiologic -> below_threshold.
- synovitis: synovial thickening/inflammation, synovitis, hypertrophic synovium, Hoffa fat pad synovitis, reactive synovitis. "Effusion-synovitis" counts as synovitis.
- bakers_cyst: MODERATE or LARGE popliteal (Baker) cyst. Small Baker cyst -> below_threshold. Ruptured Baker cyst -> positive.
- bone_contusion: bone marrow edema-like signal from impact (bone bruise, contusion, trabecular microfracture/bone bruise pattern, pivot-shift contusions) WITHOUT a discrete fracture line. Degenerative subchondral edema in OA -> below_threshold. Edema due to stress reaction alone -> equivocal.
- fracture: ACUTE cortical break or fracture line (incl. osteochondral fracture, impaction fracture, Segond, avulsion, subchondral insufficiency fracture with fracture line). Old healed fracture or hardware only -> below_threshold.

Return JSON only."""

def schema():
    item = {"type": "OBJECT", "properties": {
        "grade": {"type": "STRING", "enum": GRADES},
        "evidence": {"type": "STRING"}}, "required": ["grade", "evidence"]}
    props = {k: item for k in KEYS}
    props.update({
        "language": {"type": "STRING"},
        "postoperative": {"type": "BOOLEAN"},
        "acute_trauma": {"type": "BOOLEAN"},
    })
    return {"type": "OBJECT", "properties": props,
            "required": KEYS + ["language", "postoperative", "acute_trauma"],
            "propertyOrdering": ["language", "postoperative", "acute_trauma"] + KEYS}

def build_body(report):
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text":
            "Report:\n<<<\n" + report + "\n>>>\n\nFor each finding give the grade and a very short evidence quote "
            "(original language, <=15 words; empty string if not_mentioned)."}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                             "responseSchema": schema()},
    }

async def call(session, model, key, body, sem, retries=8):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(retries):
        async with sem:
            try:
                async with session.post(url, json=body, headers={"x-goog-api-key": key},
                                        timeout=aiohttp.ClientTimeout(total=300)) as r:
                    d = await r.json(content_type=None)
                    if r.status == 200:
                        txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                        return json.loads(txt), d.get("usageMetadata", {})
                    err = f"HTTP {r.status}: {str(d)[:200]}"
            except Exception as e:
                err = repr(e)[:200]
        await asyncio.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(err)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--subset", default="gold", choices=["gold", "all"])
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]
    t = pd.read_csv(ROOT / "data/train.csv")
    if a.subset == "gold":
        t = t.dropna(subset=LABELS)
    out = ROOT / "labels/raw" / f"{a.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["StudyInstanceUID"] for l in out.open()}
    todo = t[~t.StudyInstanceUID.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{a.model}: {len(done)} cached, {len(todo)} to do", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    tok = {"in": 0, "out": 0}
    fails = 0
    conn = aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
    async with aiohttp.ClientSession(connector=conn) as s:
        async def one(uid, rep):
            nonlocal fails
            try:
                res, usage = await call(s, a.model, key, build_body(rep), sem)
            except Exception as e:
                fails += 1
                print("FAIL", uid[-12:], e, flush=True)
                return
            tok["in"] += usage.get("promptTokenCount", 0)
            tok["out"] += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            with out.open("a") as f:
                f.write(json.dumps({"StudyInstanceUID": uid, **res}, ensure_ascii=False) + "\n")
        tasks = [one(u, r) for u, r in zip(todo.StudyInstanceUID, todo.Report)]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            await fut
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} tokens in={tok['in']} out={tok['out']} fails={fails}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
