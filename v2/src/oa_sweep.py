"""Forward-citation sweep via OpenAlex (no key; polite pool via mailto).

For each anchor: resolve the work, then list every citing work from 2025-01-01 and keep those
whose title/abstract mention a sensor/PRNU term AND a generative term. Writes out/sweep_oa.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, os, time, urllib.request, urllib.parse

OUT = os.path.join(EINV.V2, 'out'); os.makedirs(OUT, exist_ok=True)
MAILTO = "cv@upjao.ai"
ANCHORS = {
    "yu2021":   "Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data",
    "chen2008": "Determining Image Origin and Integrity Using Sensor Noise",
    "siren":    "Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models",
    "promark":  "ProMark: Proactive Diffusion Watermarking for Causal Attribution",
    "klier":    "Boon or Bane: Source Camera Identification meets AI-generated images",
    "lukas2006":"Digital camera identification from sensor pattern noise",
}
SENSOR = ["prnu", "sensor pattern noise", "sensor noise", "camera fingerprint", "source camera",
          "photo-response", "photo response", "device fingerprint", "camera identification", "sensor fingerprint"]
GEN = ["diffusion", "generative", "text-to-image", "lora", "personaliz", "fine-tun", "dreambooth",
       "stable diffusion", "generated image", "synthetic image", "ai-generated", "gan"]

def get(url, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url + ("&" if "?" in url else "?") + "mailto=" + MAILTO,
                                         headers={"User-Agent": f"einv-sweep ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=60) as r: return json.load(r)
        except Exception as e:
            print(f"   retry {t+1}: {type(e).__name__} {str(e)[:60]}", flush=True); time.sleep(3*(t+1))
    return None

def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    if not inv: return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs: pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))

report = {}
for key, title in ANCHORS.items():
    d = get("https://api.openalex.org/works?search=" + urllib.parse.quote(title) + "&per-page=3")
    if not d or not d.get("results"): print(f"[{key}] not resolved", flush=True); continue
    w = d["results"][0]; wid = w["id"].rsplit("/", 1)[-1]
    print(f"[{key}] {w['display_name'][:70]} ({w.get('publication_year')}) cited_by={w.get('cited_by_count')}", flush=True)
    hits, n, page = [], 0, 1
    while True:
        c = get(f"https://api.openalex.org/works?filter=cites:{wid},from_publication_date:2025-01-01&per-page=200&page={page}")
        if not c or not c.get("results"): break
        for cw in c["results"]:
            n += 1
            txt = ((cw.get("display_name") or "") + " " + abstract(cw)).lower()
            if any(s in txt for s in SENSOR) and any(g in txt for g in GEN):
                hits.append({"title": cw.get("display_name"), "year": cw.get("publication_year"),
                             "venue": ((cw.get("primary_location") or {}).get("source") or {}).get("display_name"),
                             "doi": cw.get("doi"), "abstract": abstract(cw)[:700]})
        if len(c["results"]) < 200: break
        page += 1; time.sleep(0.5)
    report[key] = {"anchor": w["display_name"], "openalex": wid, "n_citing_since_2025": n, "hits": hits}
    json.dump(report, open(os.path.join(OUT, "sweep_oa.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"   {n} citing works since 2025; {len(hits)} match sensor+generative:", flush=True)
    for h in hits: print(f"     - ({h['year']}) {(h['title'] or '')[:100]}  [{(h['venue'] or '')[:28]}]", flush=True)
    time.sleep(0.5)
json.dump(report, open(os.path.join(OUT, "sweep_oa.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("written out/sweep_oa.json", flush=True)
