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
import re
import shutil
import sqlite3
import sys
from collections import Counter
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
SKIP_DIRS = {".photoclean", "_trash"}
SCREENSHOT_NAME = re.compile(r"screenshot|スクリーンショット|screen shot", re.I)
GROUP_KINDS = ("exact", "dup")
# 先頭の "photo" が通常写真。これ以外の確率の合計で誤撮影を判定する
PROMPTS = {
    "photo": "a nice photo",
    "ground": "a photo of the ground, floor or asphalt taken by accident",
    "pocket": "a dark accidental photo taken inside a pocket or bag",
    "blurry": "a very blurry out of focus photo",
    "screen": "a screenshot of a phone screen",
}
COLUMNS = ["path", "mtime", "size", "taken", "sha1", "phash", "sharp", "w", "h", "make", "comment"]
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


def open_db(root: Path) -> sqlite3.Connection:
    db = sqlite3.connect(work_dir(root) / "cache.db")
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    return db


def reason_kind(reason: str) -> str:
    return reason.partition(":")[0]


# ---- scan ----

def list_images(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTS and not SKIP_DIRS & set(p.relative_to(root).parts))


def read_exif(im: Image.Image, mtime: float) -> tuple[str, str, str]:
    exif = im.getexif()
    sub = exif.get_ifd(0x8769)
    taken = sub.get(36867) or exif.get(306)
    taken = taken.strip() if isinstance(taken, str) else datetime.fromtimestamp(mtime).strftime("%Y:%m:%d %H:%M:%S")
    make = str(exif.get(271) or "").strip("\x00 ")
    comment = sub.get(37510)
    comment = comment.decode("utf-8", "ignore") if isinstance(comment, bytes) else str(comment or "")
    return taken, make, comment


def sharpness(im: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze(args) -> dict | None:
    """画像1枚の特徴量を返し、サムネイルを保存する。読めない画像は None。"""
    root, rel = args
    p = root / rel
    try:
        st = p.stat()
        sha1 = hashlib.sha1(p.read_bytes()).hexdigest()
        im = Image.open(p)
        w, h = im.size
        taken, make, comment = read_exif(im, st.st_mtime)
        im.draft("RGB", (512, 512))  # JPEG はデコード時に縮小して高速化
        im = im.convert("RGB")
        im.thumbnail((512, 512))
        phash = str(imagehash.phash(im))
        sharp = sharpness(im)
        im.thumbnail((256, 256))
        im.save(thumb_path(root, rel), quality=80)
    except Exception as e:
        print(f"  読めませんでした: {rel} ({e})", file=sys.stderr)
        return None
    return dict(path=rel, mtime=st.st_mtime, size=st.st_size, taken=taken, sha1=sha1, phash=phash,
                sharp=sharp, w=w, h=h, make=make, comment=comment)


def clip_scores(root: Path, rels: list[str]):
    """(rel, PROMPTS の順の確率) を返す。"""
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
            yield from zip(batch, probs)
            print(f"  clip {min(i + 64, len(rels))}/{len(rels)}", file=sys.stderr)


def cmd_scan(root: Path):
    db = open_db(root)
    files = list_images(root)
    current = set(files)
    known = {r["path"]: (r["mtime"], r["size"]) for r in db.execute("select path, mtime, size from img")}
    db.executemany("delete from img where path=?", [(p,) for p in known if p not in current])
    todo = []
    for rel in files:
        st = (root / rel).stat()
        if known.get(rel) != (st.st_mtime, st.st_size):
            todo.append(rel)
    print(f"{len(todo)} 枚を解析", file=sys.stderr)
    with ProcessPoolExecutor() as ex:
        for n, res in enumerate(ex.map(analyze, [(root, r) for r in todo], chunksize=16), 1):
            if res:
                db.execute(f"insert or replace into img({','.join(COLUMNS)}) values({','.join('?' * len(COLUMNS))})",
                           [res[c] for c in COLUMNS])
            if n % 200 == 0:
                db.commit()
                print(f"  {n}/{len(todo)}", file=sys.stderr)
    db.commit()
    need = [r["path"] for r in db.execute("select path from img where p_photo is null")]
    cols = ",".join(f"p_{k}=?" for k in PROMPTS)
    for rel, probs in clip_scores(root, need):
        db.execute(f"update img set {cols} where path=?", (*probs, rel))
    db.commit()


# ---- plan ----

def exact_duplicates(rows: list[dict]) -> dict[str, str]:
    """SHA1 が同じものを、最初に出てきた1枚への参照として返す。"""
    first, out = {}, {}
    for r in rows:
        if r["sha1"] in first:
            out[r["path"]] = first[r["sha1"]]
        else:
            first[r["sha1"]] = r["path"]
    return out


def similar_groups(rows: list[dict], window: int, dup_dist: int) -> list[list[dict]]:
    """直近 window 枚だけと比較し、連鎖的に似ているものを1グループにまとめる。2枚以上のグループだけ返す。"""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [imagehash.hex_to_hash(r["phash"]) for r in rows]
    for i in range(len(rows)):
        for j in range(max(0, i - window), i):
            if hashes[i] - hashes[j] <= dup_dist:
                parent[find(i)] = find(j)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(i), []).append(r)
    return [g for g in groups.values() if len(g) > 1]


def pick_best(group: list[dict]) -> dict:
    return max(group, key=lambda r: (r["sharp"], r["w"] * r["h"]))


def classify(r: dict, blur_max: float, junk_prob: float) -> str | None:
    name = Path(r["path"]).name
    p = {k: r.get(f"p_{k}") or 0 for k in PROMPTS}
    if (SCREENSHOT_NAME.search(name) or "screenshot" in r["comment"].lower()
            or (not r["make"] and (name.lower().endswith(".png") or p["screen"] >= junk_prob))):
        return "screenshot"
    if p["ground"] + p["pocket"] >= junk_prob:
        return f"junk:{p['ground'] + p['pocket']:.2f}"
    # CLIP の blurry 単体はシャープな写真でも高く出るので、シャープネスが境界付近のときだけ使う
    if r["sharp"] < blur_max or (r["sharp"] < blur_max * 3 and p["blurry"] >= junk_prob):
        return f"blur:{r['sharp']:.0f}"
    return None


def judge(rows: list[dict], window: int, dup_dist: int, blur_max: float, junk_prob: float) -> dict[str, str]:
    """rows は撮影日時順。{path: reason} を返す。reason は "kind" または "kind:詳細"。"""
    reasons = {p: f"exact:{ref}" for p, ref in exact_duplicates(rows).items()}
    live = [r for r in rows if r["path"] not in reasons]
    for g in similar_groups(live, window, dup_dist):
        best = pick_best(g)
        reasons.update({r["path"]: f"dup:{best['path']}" for r in g if r is not best})
    for r in live:
        if r["path"] not in reasons and (why := classify(r, blur_max, junk_prob)):
            reasons[r["path"]] = why
    return reasons


def review_items(rows: list[dict], reasons: dict[str, str]) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """review.html に並べる単位を撮影順に返す。

    (kind, [(path, reason, 移動するか)]) の列。重複は残す1枚を含めたグループ、それ以外は1枚ずつ。
    """
    def keeper(rel):
        # exact の参照先がさらに dup のことがあるので、残す側までたどる
        while reason_kind(reasons.get(rel, "")) in GROUP_KINDS:
            rel = reasons[rel].partition(":")[2]
        return rel

    order = {r["path"]: i for i, r in enumerate(rows)}
    members = {}
    for r in rows:
        if reason_kind(reasons.get(r["path"], "")) in GROUP_KINDS:
            members.setdefault(keeper(r["path"]), []).append(r["path"])
    items = []
    for r in rows:
        rel = r["path"]
        if rel in members:
            kind = reason_kind(reasons[members[rel][0]])
            group = sorted([rel, *members[rel]], key=order.__getitem__)
            items.append((kind, [(m, reasons.get(m, f"{kind}:{rel}"), m != rel) for m in group]))
        elif rel in reasons and reason_kind(reasons[rel]) not in GROUP_KINDS:
            items.append((reason_kind(reasons[rel]), [(rel, reasons[rel], True)]))
    return items


def render_review(root: Path, rows: list[dict], reasons: dict[str, str]) -> str:
    sharp = {r["path"]: r["sharp"] for r in rows}

    def card(rel, why, checked):
        kind = reason_kind(why)
        label = kind if kind in GROUP_KINDS else why
        return (f'<label class="card {kind}"><input type=checkbox{" checked" if checked else ""}'
                f' data-p="{html.escape(rel)}" data-r="{html.escape(why)}">'
                f'<img src="thumbs/{thumb_path(root, rel).name}">'
                f'<span>{html.escape(label)} sharp:{sharp[rel]:.0f}<br>{html.escape(Path(rel).name)}</span></label>')

    parts = []
    for kind, cards in review_items(rows, reasons):
        body = "".join(card(*c) for c in cards)
        parts.append(f'<div class="group {kind}">{body}</div>' if kind in GROUP_KINDS else body)
    return REVIEW_HTML.replace("{{CARDS}}", "\n".join(parts))


def write_plan(path: Path, rows: list[dict], reasons: dict[str, str]):
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["path", "reason"])
        cw.writerows((r["path"], reasons[r["path"]]) for r in rows if r["path"] in reasons)


def cmd_plan(root: Path, window: int, dup_dist: int, blur_max: float, junk_prob: float) -> dict[str, str]:
    rows = [dict(r) for r in open_db(root).execute("select * from img order by taken, path")
            if (root / r["path"]).exists()]
    reasons = judge(rows, window, dup_dist, blur_max, junk_prob)
    wd = work_dir(root)
    write_plan(wd / "plan.csv", rows, reasons)
    (wd / "review.html").write_text(render_review(root, rows, reasons))
    counts = dict(Counter(reason_kind(v) for v in reasons.values()))
    print(f"{len(reasons)}/{len(rows)} 枚が候補 {counts}\n  {wd / 'review.html'}\n  {wd / 'plan.csv'}")
    return reasons


# ---- apply ----

def free_path(p: Path) -> Path:
    """p が既にあれば IMG_1_1.jpeg のように番号を付けて空いているパスを返す。"""
    out, n = p, 1
    while out.exists():
        out = p.with_name(f"{p.stem}_{n}{p.suffix}")
        n += 1
    return out


def sidecars(src: Path) -> list[Path]:
    """Live Photo の動画や編集情報など、同じ名前で並んでいる付属ファイル。"""
    return [p for p in src.parent.glob(glob.escape(src.stem) + ".*") if p.suffix.lower() in SIDECAR_EXTS]


def cmd_apply(root: Path, plan: Path) -> int:
    root = root.resolve()
    trash = root / "_trash"
    moved = 0
    with open(work_dir(root) / "applied.csv", "a", newline="") as log, open(plan, newline="") as f:
        lw = csv.writer(log)
        for row in csv.DictReader(f):
            src = (root / row["path"]).resolve()
            if not src.is_relative_to(root) or SKIP_DIRS & set(src.relative_to(root).parts):
                print(f"  対象フォルダの外なので飛ばしました: {row['path']}", file=sys.stderr)
                continue
            if not src.is_file():
                continue
            dst_dir = trash / reason_kind(row["reason"]) / src.parent.relative_to(root)
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in [src, *sidecars(src)]:
                dst = free_path(dst_dir / p.name)
                shutil.move(p, dst)
                lw.writerow([str(p.relative_to(root)), str(dst.relative_to(root)), row["reason"]])
            moved += 1
    print(f"{moved} 枚を移動しました: {trash}（戻すときは .photoclean/applied.csv を参照）")
    return moved


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scan").add_argument("dir", type=Path)
    pp = sp.add_parser("plan")
    pp.add_argument("dir", type=Path)
    pp.add_argument("--window", type=int, default=20, help="近似重複を比較する直近の枚数")
    pp.add_argument("--dup-dist", type=int, default=8, help="pHash のハミング距離がこれ以下なら重複")
    pp.add_argument("--blur-max", type=float, default=20, help="シャープネス（512px での Laplacian 分散）がこれ未満ならブレ")
    pp.add_argument("--junk-prob", type=float, default=0.6, help="CLIP で誤撮影・ブレと判定する確率")
    xp = sp.add_parser("apply")
    xp.add_argument("dir", type=Path)
    xp.add_argument("plan", type=Path, nargs="?")
    a = ap.parse_args(argv)
    root = a.dir.resolve()
    if a.cmd == "scan":
        cmd_scan(root)
    elif a.cmd == "plan":
        cmd_plan(root, a.window, a.dup_dist, a.blur_max, a.junk_prob)
    else:
        cmd_apply(root, a.plan or work_dir(root) / "plan.csv")


if __name__ == "__main__":
    main()
