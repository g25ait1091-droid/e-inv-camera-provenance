"""Fetch every file (any type) in a public Drive folder, recursively one level.
    python drive_fetch_any.py <folder_id> <dest_dir>
"""
import os, re, sys, subprocess, time

def listing(folder_id):
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    html = subprocess.run(["curl", "-sS", "-m", "120", "-L", "-A", "Mozilla/5.0", url],
                          capture_output=True, text=True, check=True).stdout
    ents = []
    for m in re.finditer(r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]{20,})"(.*?)class="flip-entry-title">([^<]+)<', html, re.S):
        fid, chunk, name = m.group(1), m.group(2), m.group(3).strip()
        is_dir = "drive.google.com/drive/folders/" in chunk or 'folder' in chunk.lower()
        ents.append((name, fid, is_dir))
    return ents

def fetch(fid, dest):
    url = "https://drive.google.com/uc?export=download&id=" + fid + "&confirm=t"
    tmp = dest + ".part"
    for attempt in range(3):
        r = subprocess.run(["curl", "-sSL", "-m", "600", "-A", "Mozilla/5.0", "-o", tmp, url])
        if r.returncode == 0 and os.path.getsize(tmp) > 0:
            head = open(tmp, "rb").read(64)
            if b"<html" not in head.lower():
                os.replace(tmp, dest); return os.path.getsize(dest)
        time.sleep(2 + 3*attempt)
    if os.path.exists(tmp): os.remove(tmp)
    return -1

def main():
    fid, dest = sys.argv[1], sys.argv[2]; os.makedirs(dest, exist_ok=True)
    for name, i, is_dir in listing(fid):
        p = os.path.join(dest, name)
        if is_dir:
            sub = listing(i); os.makedirs(p, exist_ok=True)
            for n2, i2, d2 in sub:
                if d2: continue
                q = os.path.join(p, n2)
                if os.path.exists(q): continue
                print(f"  {name}/{n2}: {fetch(i2, q)/1e6:.1f} MB", flush=True)
        else:
            if os.path.exists(p): continue
            print(f"  {name}: {fetch(i, p)/1e6:.1f} MB", flush=True)
    print("done", dest, flush=True)

if __name__ == "__main__":
    main()
