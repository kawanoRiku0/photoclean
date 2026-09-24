# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pillow", "pillow-heif", "imagehash", "opencv-python-headless", "numpy",
#   "open_clip_torch", "torch",
# ]
# ///
"""写真フォルダの重複・スクショ・ブレ・誤撮影を検出して _trash/ に移動する。

  uv run photoclean.py scan  DIR   # 特徴量を計算して DIR/.photoclean/ にキャッシュ
  uv run photoclean.py plan  DIR   # 判定して plan.csv と review.html を出力
  uv run photoclean.py apply DIR [plan.csv]   # plan.csv の行を DIR/_trash/<reason>/ へ移動
"""
import argparse
import csv
import glob
import hashlib
import html
import os
import re
import shutil
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
SIDECAR_EXTS = {".mov", ".mp4", ".aae"}
SCREENSHOT_NAME = re.compile(r"screenshot|スクリーンショット|screen shot", re.I)
# 先頭の "photo" が通常写真。これ以外の確率の合計で誤撮影を判定する
PROMPTS = {
    "photo": "a nice photo",
    "ground": "a photo of the ground, floor or asphalt taken by accident",
    "pocket": "a dark accidental photo taken inside a pocket or bag",
    "blurry": "a very blurry out of focus photo",
    "screen": "a screenshot of a phone screen",
}
SCHEMA = """create table if not exists img(
  path text primary key, mtime real, size int, taken text, sha1 text, phash text,
  sharp real, w int, h int, make text, comment text,
  p_photo real, p_ground real, p_pocket real, p_blurry real, p_screen real)"""


def work_dir(root: Path) -> Path:
    d = root / ".photoclean"
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    return d


def thumb_path(root: Path, rel: str) -> Path:
    return work_dir(root) / "thumbs" / (hashlib.md5(rel.encode()).hexdigest() + ".jpg")


def analyze(args):
    root, rel = args
    p = root / rel
    st = p.stat()
    sha1 = hashlib.sha1(p.read_bytes()).hexdigest()
    im = Image.open(p)
    w, h = im.size
    exif = im.getexif()
    sub = exif.get_ifd(0x8769)
    taken = sub.get(36867) or exif.get(306)
    taken = taken.strip() if isinstance(taken, str) else datetime.fromtimestamp(st.st_mtime).strftime("%Y:%m:%d %H:%M:%S")
    make = str(exif.get(271) or "").strip("\x00 ")
    comment = sub.get(37510)
    comment = comment.decode("utf-8", "ignore") if isinstance(comment, bytes) else str(comment or "")
    im.draft("RGB", (512, 512))  # JPEG はデコード時に縮小して高速化
    im = im.convert("RGB")
    im.thumbnail((512, 512))
    phash = str(imagehash.phash(im))
    gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    im.thumbnail((256, 256))
    im.save(thumb_path(root, rel), quality=80)
    return (rel, st.st_mtime, st.st_size, taken, sha1, phash, sharp, w, h, make, comment)


def clip_scores(root: Path, rels: list[str]):
    import open_clip
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok(list(PROMPTS.values())).to(dev))
        t /= t.norm(dim=-1, keepdim=True)
        for i in range(0, len(rels), 64):
            batch = rels[i:i + 64]
            x = torch.stack([pre(Image.open(thumb_path(root, r))) for r in batch]).to(dev)
            f = model.encode_image(x)
            f /= f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ t.T).softmax(dim=-1).cpu().tolist()
            for r, pr in zip(batch, probs):
                yield r, pr
            print(f"  clip {min(i + 64, len(rels))}/{len(rels)}", file=sys.stderr)


def cmd_scan(root: Path):
    db = sqlite3.connect(work_dir(root) / "cache.db")
    db.execute(SCHEMA)
    known = {r[0]: (r[1], r[2]) for r in db.execute("select path, mtime, size from img")}
    todo = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in EXTS or ".photoclean" in p.parts or "_trash" in p.parts:
            continue
        rel = str(p.relative_to(root))
        st = p.stat()
        if known.get(rel) != (st.st_mtime, st.st_size):
            todo.append(rel)
    print(f"{len(todo)} 枚を解析", file=sys.stderr)
    with ProcessPoolExecutor() as ex:
        for n, res in enumerate(ex.map(analyze, [(root, r) for r in todo], chunksize=16), 1):
            db.execute("insert or replace into img(path,mtime,size,taken,sha1,phash,sharp,w,h,make,comment)"
                       " values(?,?,?,?,?,?,?,?,?,?,?)", res)
            if n % 200 == 0:
                db.commit()
                print(f"  {n}/{len(todo)}", file=sys.stderr)
    db.commit()
    need = [r[0] for r in db.execute("select path from img where p_photo is null")]
    for rel, pr in clip_scores(root, need):
        db.execute("update img set p_photo=?,p_ground=?,p_pocket=?,p_blurry=?,p_screen=? where path=?", (*pr, rel))
    db.commit()


def judge(rows, window, dup_dist, blur_max, junk_prob):
    """rows は撮影日時順の dict。{path: reason} を返す。"""
    reasons = {}
    seen = {}
    for r in rows:
        if r["sha1"] in seen:
            reasons[r["path"]] = f"exact:{seen[r['sha1']]}"
        else:
            seen[r["sha1"]] = r["path"]
    live = [r for r in rows if r["path"] not in reasons]

    # 直近 window 枚だけと比較し、連鎖的に似ているものを1グループにまとめる
    parent = list(range(len(live)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [imagehash.hex_to_hash(r["phash"]) for r in live]
    for i in range(len(live)):
        for j in range(max(0, i - window), i):
            if hashes[i] - hashes[j] <= dup_dist:
                parent[find(i)] = find(j)
    groups = {}
    for i, r in enumerate(live):
        groups.setdefault(find(i), []).append(r)
    for g in groups.values():
        if len(g) < 2:
            continue
        best = max(g, key=lambda r: (r["sharp"], r["w"] * r["h"]))
        for r in g:
            if r is not best:
                reasons[r["path"]] = f"dup:{best['path']}"

    for r in live:
        if r["path"] in reasons:
            continue
        name = Path(r["path"]).name
        if (SCREENSHOT_NAME.search(name) or "screenshot" in r["comment"].lower()
                or (not r["make"] and (name.lower().endswith(".png") or (r["p_screen"] or 0) >= junk_prob))):
            reasons[r["path"]] = "screenshot"
        elif r["p_photo"] is not None and r["p_ground"] + r["p_pocket"] >= junk_prob:
            reasons[r["path"]] = f"junk:{r['p_ground'] + r['p_pocket']:.2f}"
        # CLIP の blurry 単体はシャープな写真でも高く出るので、シャープネスが境界付近のときだけ使う
        elif r["sharp"] < blur_max or (r["sharp"] < blur_max * 3 and (r["p_blurry"] or 0) >= junk_prob):
            reasons[r["path"]] = f"blur:{r['sharp']:.0f}"
    return reasons


def cmd_plan(root: Path, a):
    db = sqlite3.connect(work_dir(root) / "cache.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("select * from img order by taken, path") if (root / r["path"]).exists()]
    reasons = judge(rows, a.window, a.dup_dist, a.blur_max, a.junk_prob)
    wd = work_dir(root)
    with open(wd / "plan.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["path", "reason"])
        cw.writerows((r["path"], reasons[r["path"]]) for r in rows if r["path"] in reasons)
    by_path = {r["path"]: r for r in rows}

    def card(rel, why, checked):
        kind = why.partition(":")[0]
        return (f'<label class="card {kind}"><input type=checkbox{" checked" if checked else ""}'
                f' data-p="{html.escape(rel)}" data-r="{html.escape(why)}">'
                f'<img src="thumbs/{thumb_path(root, rel).name}">'
                f'<span>{html.escape(why if kind not in ("dup", "exact") else kind)} sharp:{by_path[rel]["sharp"]:.0f}'
                f'<br>{html.escape(Path(rel).name)}</span></label>')

    def keeper(rel):
        while reasons.get(rel, "").partition(":")[0] in ("dup", "exact"):
            rel = reasons[rel].partition(":")[2]
        return rel

    groups = {}
    for r in rows:
        if reasons.get(r["path"], "").partition(":")[0] in ("dup", "exact"):
            groups.setdefault(keeper(r["path"]), []).append(r["path"])
    cards = []
    for r in rows:
        rel = r["path"]
        if rel in groups:
            kind = reasons[groups[rel][0]].partition(":")[0]
            members = sorted([rel] + groups[rel], key=lambda m: (by_path[m]["taken"], m))
            cards.append(f'<div class="group {kind}">' + "".join(
                card(m, reasons.get(m, f"{kind}:{rel}"), m != rel) for m in members) + "</div>")
        elif rel in reasons and reasons[rel].partition(":")[0] not in ("dup", "exact"):
            cards.append(card(rel, reasons[rel], True))
    (wd / "review.html").write_text(REVIEW_HTML.replace("{{CARDS}}", "\n".join(cards)))
    counts = {}
    for v in reasons.values():
        counts[v.split(":")[0]] = counts.get(v.split(":")[0], 0) + 1
    print(f"{len(reasons)}/{len(rows)} 枚が候補 {counts}\n  {wd / 'review.html'}\n  {wd / 'plan.csv'}")


def cmd_apply(root: Path, plan: Path):
    trash = root / "_trash"
    with open(trash.parent / ".photoclean" / "applied.csv", "a", newline="") as log, open(plan) as f:
        lw = csv.writer(log)
        for row in csv.DictReader(f):
            src = root / row["path"]
            if not src.exists():
                continue
            dst_dir = trash / row["reason"].split(":")[0] / Path(row["path"]).parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Live Photo の動画や編集情報が同名で並んでいるので一緒に移動する
            sidecars = [p for p in src.parent.glob(glob.escape(src.stem) + ".*") if p.suffix.lower() in SIDECAR_EXTS]
            for p in [src] + sidecars:
                if p.exists():
                    shutil.move(p, dst_dir / p.name)
                    lw.writerow([str(p.relative_to(root)), str((dst_dir / p.name).relative_to(root)), row["reason"]])
    print(f"移動しました: {trash}（戻すときは .photoclean/applied.csv を参照）")


REVIEW_HTML = """<!doctype html><meta charset=utf-8><title>photoclean review</title>
<style>
body{font:12px system-ui;margin:16px;background:#111;color:#ddd}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}
.card{background:#222;padding:4px;border:3px solid #c33;cursor:pointer;position:relative}
.card:has(input:not(:checked)){border-color:#3a3;opacity:.6}
.card input{position:absolute;top:6px;left:6px}
.card img{width:100%;height:150px;object-fit:contain;display:block}
.group{grid-column:1/-1;display:flex;gap:8px;overflow-x:auto;padding:6px;background:#1a1a2a;border-radius:6px}
.group .card{flex:0 0 220px}
.group .card img{height:200px}
.card span{display:block;word-break:break-all}
header{position:sticky;top:0;background:#111;padding:8px 0}
</style>
<header>赤＝_trash へ移動 / 緑＝残す（クリックで切替）。重複は横一列が1グループで、初期状態は一番シャープな1枚を残す。
<button onclick="save()">plan.csv を保存</button>
<select onchange="filt(this.value)"><option value="">すべて</option><option>dup</option><option>exact</option><option>screenshot</option><option>junk</option><option>blur</option></select></header>
<div class=grid>{{CARDS}}</div>
<script>
function filt(k){document.querySelectorAll('.grid>*').forEach(c=>c.style.display=!k||c.classList.contains(k)?'':'none')}
function save(){
  const q=s=>'"'+s.replaceAll('"','""')+'"';
  const rows=[...document.querySelectorAll('input:checked')].map(i=>q(i.dataset.p)+','+q(i.dataset.r));
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob(['path,reason\\n'+rows.join('\\n')+'\\n'],{type:'text/csv'}));
  a.download='plan.csv';a.click();
}
</script>"""


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scan").add_argument("dir", type=Path)
    pp = sp.add_parser("plan")
    pp.add_argument("dir", type=Path)
    pp.add_argument("--window", type=int, default=20, help="近似重複を比較する直近の枚数")
    pp.add_argument("--dup-dist", type=int, default=8, help="pHash のハミング距離がこれ以下なら重複")
    pp.add_argument("--blur-max", type=float, default=20, help="シャープネス（512px での Laplacian 分散）がこれ未満ならブレ")
    pp.add_argument("--junk-prob", type=float, default=0.6, help="CLIP で誤撮影・ブレと判定する確率")
    ap_ = sp.add_parser("apply")
    ap_.add_argument("dir", type=Path)
    ap_.add_argument("plan", type=Path, nargs="?")
    a = ap.parse_args()
    root = a.dir.resolve()
    if a.cmd == "scan":
        cmd_scan(root)
    elif a.cmd == "plan":
        cmd_plan(root, a)
    else:
        cmd_apply(root, a.plan or work_dir(root) / "plan.csv")


if __name__ == "__main__":
    main()
