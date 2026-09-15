"""Fetch every PNG in a public Drive folder via the embeddedfolderview listing (no pagination).

    python drive_fetch.py <folder_id> <dest_dir> [--limit N]
Skips files already present with a valid PNG header. Prints a one-line summary.
"""
import os, re, sys, subprocess, time

def listing(folder_id):
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    html = subprocess.run(["curl", "-sS", "-m", "120", "-L", "-A", "Mozilla/5.0", url],
                          capture_output=True, text=True, check=True).stdout
    # entries: <div class="flip-entry" id="entry-<id>"> ... <div class="flip-entry-title">NAME</div>
    out = []
    for m in re.finditer(r'id="entry-([A-Za-z0-9_-]{20,})".*?class="flip-entry-title">([^<]+)<', html, re.S):
        out.append((m.group(2).strip(), m.group(1)))
    return out

def is_png(p):
    try:
        with open(p, "rb") as f:
            return f.read(4) == b"\x89PNG"
    except OSError:
        return False

def fetch(fid, dest):
    url = "https://drive.google.com/uc?export=download&id=" + fid
    tmp = dest + ".part"
    for attempt in range(3):
        r = subprocess.run(["curl", "-sSL", "-m", "180", "-A", "Mozilla/5.0", "-o", tmp, url])
        if r.returncode == 0 and is_png(tmp):
            os.replace(tmp, dest); return True
        time.sleep(2 + 3 * attempt)
    if os.path.exists(tmp): os.remove(tmp)
    return False

def main():
    fid, dest = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    os.makedirs(dest, exist_ok=True)
    ents = [(n, i) for n, i in listing(fid) if n.lower().endswith(".png")]
    ents.sort()
    if limit: ents = ents[:limit]
    got = skip = fail = 0
    todo = []
    for n, i in ents:
        p = os.path.join(dest, n)
        if is_png(p): skip += 1
        else: todo.append((n, i, p))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, ok in zip([t[0] for t in todo], ex.map(lambda t: fetch(t[1], t[2]), todo)):
            if ok: got += 1
            else: fail += 1; print("  FAIL", n, flush=True)
    print(f"{os.path.basename(dest)}: listed {len(ents)}  fetched {got}  present {skip}  failed {fail}", flush=True)

if __name__ == "__main__":
    main()
