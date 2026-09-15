"""Forward-citation sweep via the Semantic Scholar Graph API (no key needed, rate-limited).

For each anchor paper, pull every citing paper, keep those from 2025 onward whose title or
abstract mentions any sensor/PRNU/camera-fingerprint term together with any generative term.
Writes out/sweep_s2.json and prints the hits. This is the sweep the v1 run list recorded as
"still owed".
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, os, time, urllib.request, urllib.parse

OUT = os.path.join(EINV.V2, 'out'); os.makedirs(OUT, exist_ok=True)
API = "https://api.semanticscholar.org/graph/v1"
ANCHORS = {
    "yu2021_artificial_fingerprinting": "Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data",
    "chen2008_sensor_noise":            "Determining Image Origin and Integrity Using Sensor Noise",
    "siren_verification":               "Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models",
    "promark":                          "ProMark: Proactive Diffusion Watermarking for Causal Attribution",
    "klier_baier_boon_bane":            "Boon or Bane: Source Camera Identification meets AI-generated images",
}
SENSOR = ["prnu", "sensor pattern noise", "sensor noise", "camera fingerprint", "source camera",
          "photo-response", "photo response", "device fingerprint", "camera identification"]
GEN = ["diffusion", "generative", "text-to-image", "lora", "personaliz", "fine-tun", "dreambooth",
       "stable diffusion", "generated image", "synthetic image"]

def get(url, tries=6):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "einv-sweep"}), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            wait = 3 * (t + 1); print(f"   retry {t+1} after {type(e).__name__}: {str(e)[:60]} (sleep {wait}s)", flush=True); time.sleep(wait)
    return None

def find(title):
    q = urllib.parse.quote(title)
    d = get(f"{API}/paper/search?query={q}&limit=3&fields=title,year,paperId,citationCount")
    if not d or not d.get("data"): return None
    return d["data"][0]

def citations(pid):
    out, off = [], 0
    while True:
        d = get(f"{API}/paper/{pid}/citations?fields=title,year,abstract,venue,externalIds&limit=500&offset={off}")
        if not d: break
        out += [c["citingPaper"] for c in d.get("data", []) if c.get("citingPaper")]
        if "next" not in d: break
        off = d["next"]; time.sleep(1.2)
    return out

report = {}
for key, title in ANCHORS.items():
    p = find(title); time.sleep(1.2)
    if not p: print(f"[{key}] not found", flush=True); continue
    print(f"[{key}] {p['title'][:70]} ({p.get('year')}) citations={p.get('citationCount')}", flush=True)
    cits = citations(p["paperId"]); time.sleep(1.2)
    hits = []
    for c in cits:
        if not c.get("year") or c["year"] < 2025: continue
        txt = ((c.get("title") or "") + " " + (c.get("abstract") or "")).lower()
        if any(s in txt for s in SENSOR) and any(g in txt for g in GEN):
            hits.append({"title": c.get("title"), "year": c.get("year"), "venue": c.get("venue"),
                         "ids": c.get("externalIds"), "abstract": (c.get("abstract") or "")[:600]})
    report[key] = {"anchor": p, "n_citing": len(cits), "hits": hits}
    print(f"   {len(cits)} citing papers, {len(hits)} match sensor+generative from 2025:", flush=True)
    for h in hits: print(f"     - ({h['year']}) {h['title'][:110]}  [{(h['venue'] or '')[:30]}]", flush=True)

json.dump(report, open(os.path.join(OUT, "sweep_s2.json"), "w"), indent=1)
print("written out/sweep_s2.json", flush=True)
