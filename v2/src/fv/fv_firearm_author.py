"""Firearm-classifier validation against the AUTHOR's own labels (recorded 30 Sep 2026).

The author labelled the 64 validation images by eye on a blind page (seeded shuffled order 20260930, no
classifier output, same written criterion as out/t6_weapon_validation_labels.json). The raw page export is
stored verbatim below (keys = page positions); page positions map to image ids through the order file.
Writes out/t6_weapon_validation_labels_author.json and out/t6_weapon_validation_author.json.
The earlier AI-made labels (out/t6_weapon_validation_labels.json) are kept unchanged and compared.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import os

OUT = (EINV.V2 + "/out")
ORDER = OUT + "/t6_weapon_validation_page_order.json"   # the blind page's order (shipped beside the labels)
PAGE_EXPORT = ('{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":1,"7":0,"8":0,"9":0,"10":1,"11":1,"12":1,"13":1,'
               '"14":0,"15":0,"16":1,"17":1,"18":0,"19":1,"20":1,"21":0,"22":0,"23":0,"24":0,"25":0,"26":0,'
               '"27":0,"28":1,"29":0,"30":0,"31":1,"32":0,"33":1,"34":0,"35":0,"36":0,"37":0,"38":1,"39":1,'
               '"40":1,"41":0,"42":1,"43":0,"44":0,"45":0,"46":1,"47":1,"48":1,"49":0,"50":0,"51":1,"52":1,'
               '"53":0,"54":0,"55":0,"56":0,"57":0,"58":1,"59":0,"60":1,"61":0,"62":1,"63":1}')


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    order = json.load(open(ORDER, encoding="utf-8"))
    page = json.loads(PAGE_EXPORT)
    assert len(page) == 64 and len(order) == 64
    author = {it["id"]: int(page[str(it["k"])]) for it in order}
    ai_rec = json.load(open(os.path.join(OUT, "t6_weapon_validation_labels.json"), encoding="utf-8"))
    clip = json.load(open(os.path.join(OUT, "t6_weapon_clip.json"), encoding="utf-8"))["validation"]
    rows = {r["id"]: r for r in clip["rows"]}
    assert set(author) == set(rows) == set(ai_rec["labels"])

    ids = sorted(author)
    a = [author[i] for i in ids]
    c = [int(bool(rows[i]["clip_firearm"])) for i in ids]
    m = [int(ai_rec["labels"][i]) for i in ids]
    strata = {}
    for i in ids:
        s = rows[i]["stratum"]
        d = strata.setdefault(s, {"n": 0, "agree": 0, "author_firearm": 0, "clip_firearm": 0})
        d["n"] += 1
        d["agree"] += int(author[i] == int(bool(rows[i]["clip_firearm"])))
        d["author_firearm"] += author[i]
        d["clip_firearm"] += int(bool(rows[i]["clip_firearm"]))
    res = {
        "labeller": "the author, by eye, on a blind page (seeded shuffled order 20260930; no classifier output; "
                    "criterion as in t6_weapon_validation_labels.json); recorded 2026-09-30",
        "criterion": ai_rec["criterion"],
        "n": len(ids),
        "author_firearm": sum(a),
        "clip_vs_author": {
            "agree": sum(x == y for x, y in zip(a, c)),
            "agreement": sum(x == y for x, y in zip(a, c)) / len(ids),
            "cohen_kappa": kappa(a, c),
            "tp": sum(1 for x, y in zip(a, c) if x == 1 and y == 1),
            "tn": sum(1 for x, y in zip(a, c) if x == 0 and y == 0),
            "fp_clip": [i for i, x, y in zip(ids, a, c) if x == 0 and y == 1],
            "fn_clip": [i for i, x, y in zip(ids, a, c) if x == 1 and y == 0],
        },
        "author_vs_ai_labels": {
            "agree": sum(x == y for x, y in zip(a, m)),
            "cohen_kappa": kappa(a, m),
            "disagree": [i for i, x, y in zip(ids, a, m) if x != y],
        },
        "ai_labels_vs_clip_as_previously_reported": {"agree": clip["agree"], "cohen_kappa": clip["cohen_kappa"]},
        "by_stratum": strata,
    }
    json.dump({"labeller": res["labeller"], "criterion": res["criterion"], "labels": author,
               "page_export_verbatim": PAGE_EXPORT, "page_order_file_seed": 20260930},
              open(os.path.join(OUT, "t6_weapon_validation_labels_author.json"), "w", encoding="utf-8"), indent=1)
    json.dump(res, open(os.path.join(OUT, "t6_weapon_validation_author.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "by_stratum"}, indent=1))
    print(json.dumps(strata, indent=1))


if __name__ == "__main__":
    main()
