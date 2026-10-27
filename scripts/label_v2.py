"""Budget report labeller: several reports per request, compact output, cost accounting.

  python scripts/label_v2.py eval  --model gemini-3.8-flash --thinking low   # 58 gold, sync
  python scripts/label_v2.py count                                           # exact input tokens, free
  python scripts/label_v2.py run   --model ... --thinking ... --subset all   # sync, resumable

Output: labels/raw/v2__<model>__<thinking>.jsonl, one line per study:
  {"StudyInstanceUID", "<key>": [grade, severity], ...}
"""
import argparse, asyncio, json, os, random, ssl, sys
from pathlib import Path

import aiohttp
import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from label_reports import LABELS, KEYS  # noqa: E402

# $ per 1M tokens (input, output incl. thinking), standard tier, valid through 2026-12-31.
PRICE = {"gemini-3.8-flash": (0.75, 3.75), "gemini-3.7-flash": (0.75, 3.75),
         "gemini-3.1-flash-lite": (0.25, 1.50), "gemini-3.5-flash-lite": (0.30, 2.50),
         "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
         "gemini-3.1-pro-preview": (2.00, 12.00), "gemma-4-31b-it": (0.0, 0.0),
         "gemma-4-26b-a4b-it": (0.0, 0.0)}

GRADES = ["NM", "N", "LOW", "EQ", "POS", "SEV"]

SYSTEM = """You are a musculoskeletal radiologist extracting findings from knee MRI reports (any language).
For EACH report and EACH of 12 findings output [grade, severity].

grade:
 NM  = not mentioned / cannot be judged from the report
 N   = explicitly or clearly implied normal/absent ("menisci intact", "no fracture", "no effusion")
 LOW = abnormal but below the criteria below (sprain, low-grade, degeneration, small/trace, mild, focal, chronic/old)
 EQ  = report is uncertain (possible, suspected, cannot exclude, V.a., likely) about a finding meeting criteria
 POS = meets criteria
 SEV = clearly meets criteria with high-grade features (complete/full-thickness, displaced, bucket-handle, large, bone-on-bone)
severity (0-4): 0 none/not stated, 1 trace/minimal/grade 1, 2 small/mild/low-grade/grade 2, 3 moderate/high-grade partial/grade 3, 4 large/severe/complete/grade 4.

Criteria (borderline = not POS):
acl: high-grade partial (>50%) or complete tear; torn graft counts. Mucoid degeneration, thickening, low-grade partial, intact graft -> LOW/N.
mcl: acute grade 2-3 tear with fiber disruption + edema. Grade-1 sprain, periligamentous edema only, chronic thickening -> LOW.
mm / lm (medial / lateral meniscus): signal reaching the surface, or morphologic tear (truncated, displaced fragment, flap, bucket-handle, root tear, radial, complex, maceration). Intrasubstance degeneration grade 1-2 -> LOW. Post-meniscectomy without new tear -> LOW.
moa / loa / pfoa (medial / lateral tibiofemoral / patellofemoral OA): ~>=1 cm of high-grade cartilage loss (>50% thickness, grade 3-4, full-thickness), +/- subchondral edema, cysts, osteophytes. Grade 1-2 chondromalacia, fissuring, small focal defect -> LOW. Patellofemoral = patella + trochlea.
eff: moderate or large joint effusion. Small/trace/physiologic -> LOW (severity 1-2).
syn: synovitis, synovial thickening/hypertrophy, Hoffa synovitis, effusion-synovitis.
bak: moderate or large popliteal (Baker) cyst, or ruptured cyst. Small -> LOW.
con: bone contusion / bone bruise / marrow edema from impact, WITHOUT a fracture line. Degenerative subchondral edema -> LOW.
fx: acute fracture line or cortical break (osteochondral, impaction, Segond, avulsion, insufficiency fracture). Old/healed -> LOW.

Return JSON: a list with one object per report, in input order, with its id."""

SHORT = ["acl", "mcl", "mm", "lm", "moa", "loa", "pfoa", "eff", "syn", "bak", "con", "fx"]


def schema():
    pair = {"type": "ARRAY", "items": {"type": "STRING"}, "minItems": 2, "maxItems": 2}
    obj = {"type": "OBJECT", "properties": {"id": {"type": "STRING"}, **{k: pair for k in SHORT}},
           "required": ["id"] + SHORT, "propertyOrdering": ["id"] + SHORT}
    return {"type": "ARRAY", "items": obj}


def body(reports, thinking):
    txt = "\n\n".join(f"### id={i}\n{r}" for i, r in reports)
    cfg = {"temperature": 0.0, "responseMimeType": "application/json", "responseSchema": schema()}
    if thinking != "default":
        cfg["thinkingConfig"] = {"thinkingLevel": thinking}
    return {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": txt}]}],
            "generationConfig": cfg}


def parse(res, ids):
    out = {}
    for o in res:
        if o.get("id") not in ids:
            continue
        rec = {}
        for s, k in zip(SHORT, KEYS):
            g, v = (o[s] + ["NM", "0"])[:2]
            g = g.upper() if g.upper() in GRADES else "NM"
            try:
                v = max(0, min(4, int(str(v).strip()[:1])))
            except ValueError:
                v = 0
            rec[k] = [g, v]
        out[ids[o["id"]]] = rec
    return out


class Client:
    def __init__(self, model, sem):
        self.model, self.sem = model, sem
        self.key = os.environ["GEMINI_API_KEY"]
        self.tok = {"in": 0, "out": 0}

    async def post(self, s, path, payload, retries=8):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:{path}"
        err = None
        for attempt in range(retries):
            async with self.sem:
                try:
                    async with s.post(url, json=payload, headers={"x-goog-api-key": self.key},
                                      timeout=aiohttp.ClientTimeout(total=600)) as r:
                        d = await r.json(content_type=None)
                        if r.status == 200:
                            return d
                        err = f"HTTP {r.status}: {str(d)[:300]}"
                        if r.status in (400, 403, 404):
                            break
                except Exception as e:
                    err = repr(e)[:200]
            await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(err)

    async def label(self, s, reports, thinking):
        d = await self.post(s, "generateContent", body(reports, thinking))
        u = d.get("usageMetadata", {})
        self.tok["in"] += u.get("promptTokenCount", 0)
        self.tok["out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]
                      if not p.get("thought"))
        return json.loads(txt)

    def cost(self):
        pi, po = PRICE.get(self.model, (float("nan"),) * 2)
        return (self.tok["in"] * pi + self.tok["out"] * po) / 1e6


def session():
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(
        ssl=ssl.create_default_context(cafile=certifi.where())))


async def run(a):
    t = pd.read_csv(ROOT / "data/train.csv")
    if a.subset == "gold":
        t = t.dropna(subset=LABELS)
    out = ROOT / "labels/raw" / f"v2__{a.model}__{a.thinking}{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["StudyInstanceUID"] for l in out.open()} if out.exists() else set()
    todo = t[~t.StudyInstanceUID.isin(done)].sample(frac=1, random_state=0)
    if a.limit:
        todo = todo.head(a.limit)
    chunks = [todo.iloc[i:i + a.pack] for i in range(0, len(todo), a.pack)]
    print(f"{out.name}: {len(done)} cached, {len(todo)} todo in {len(chunks)} requests", flush=True)
    cl = Client(a.model, asyncio.Semaphore(a.concurrency))
    missing = 0
    async with session() as s:
        async def one(ch):
            nonlocal missing
            ids = {f"R{j + 1}": u for j, u in enumerate(ch.StudyInstanceUID)}
            reps = [(f"R{j + 1}", r) for j, r in enumerate(ch.Report)]
            try:
                got = parse(await cl.label(s, reps, a.thinking), ids)
            except Exception as e:
                print("FAIL", e, flush=True)
                got = {}
            missing += len(ids) - len(got)
            with out.open("a") as f:
                for u, rec in got.items():
                    f.write(json.dumps({"StudyInstanceUID": u, **rec}) + "\n")
        tasks = [one(c) for c in chunks]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            await fut
            if i % 20 == 0 or i == len(tasks):
                n = (i * a.pack) or 1
                print(f"{i}/{len(tasks)} in={cl.tok['in']} out={cl.tok['out']} cost=${cl.cost():.3f} "
                      f"(~${cl.cost() / min(n, len(todo)) * 4407:.2f} per 4407) missing={missing}", flush=True)


async def count(a):
    """Exact prompt tokens for the whole corpus at the chosen pack size (countTokens is free)."""
    t = pd.read_csv(ROOT / "data/train.csv").sample(frac=1, random_state=0)
    chunks = [t.iloc[i:i + a.pack] for i in range(0, len(t), a.pack)]
    cl = Client(a.model, asyncio.Semaphore(a.concurrency))
    tot = 0
    async with session() as s:
        async def one(ch):
            nonlocal tot
            b = body([(f"R{j + 1}", r) for j, r in enumerate(ch.Report)], "default")
            d = await cl.post(s, "countTokens", {"generateContentRequest": {"model": f"models/{a.model}", **b}})
            tot += d["totalTokens"]
        await asyncio.gather(*[one(c) for c in chunks])
    print(f"{a.model} pack={a.pack}: {len(chunks)} requests, {tot} input tokens, {tot / len(t):.0f}/report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "run", "count"])
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--thinking", default="low", help="minimal|low|medium|high|default")
    ap.add_argument("--subset", default="gold", choices=["gold", "all"])
    ap.add_argument("--pack", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "eval":
        a.subset = "gold"
    asyncio.run(count(a) if a.cmd == "count" else run(a))
