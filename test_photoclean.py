import csv
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

import photoclean as pc

NORMAL = (0.9, 0.02, 0.02, 0.02, 0.04)


def row(path, phash="0000000000000000", sha1=None, sharp=100.0, make="Apple", comment="", w=4000, h=3000,
        taken="2023:01:01 00:00:00", p=NORMAL):
    d = dict(path=path, phash=phash, sha1=sha1 or path, sharp=sharp, w=w, h=h, make=make, comment=comment,
             taken=taken)
    d.update({f"p_{k}": v for k, v in zip(pc.PROMPTS, p or [None] * len(pc.PROMPTS))})
    return d


def distinct(n, prefix="z"):
    """pHash が互いに十分離れた n 行。"""
    rng = np.random.default_rng(n)
    return [row(f"{prefix}{i}.jpg", phash=f"{int(rng.integers(0, 2**63)):016x}") for i in range(n)]


def judge(rows, window=20, dup_dist=8, blur_max=20, junk_prob=0.6):
    return pc.judge(rows, window, dup_dist, blur_max, junk_prob)


# ---- judge: 完全一致 ----

def test_exact_keeps_first_and_ignores_window():
    rows = [row("a.jpg", sha1="x")] + distinct(50) + [row("b.jpg", sha1="x"), row("c.jpg", sha1="x")]
    r = judge(rows)
    assert r["b.jpg"] == "exact:a.jpg"
    assert r["c.jpg"] == "exact:a.jpg"
    assert "a.jpg" not in r


def test_exact_is_not_regrouped_as_dup():
    rows = [row("a.jpg", sha1="x", phash="ffff0000ffff0000"), row("b.jpg", sha1="x", phash="ffff0000ffff0000")]
    assert judge(rows) == {"b.jpg": "exact:a.jpg"}


# ---- judge: 近似重複 ----

def test_dup_keeps_sharpest():
    rows = [row("a.jpg", "ffff0000ffff0000", sharp=100), row("b.jpg", "ffff0000ffff0001", sharp=300)]
    assert judge(rows) == {"a.jpg": "dup:b.jpg"}


def test_dup_tie_breaks_on_resolution():
    rows = [row("a.jpg", "ffff0000ffff0000", w=100, h=100), row("b.jpg", "ffff0000ffff0000", sha1="b2")]
    assert judge(rows) == {"a.jpg": "dup:b.jpg"}


@pytest.mark.parametrize("dist,is_dup", [(8, True), (9, False)])
def test_dup_distance_boundary(dist, is_dup):
    b = f"{(1 << dist) - 1:016x}"
    rows = [row("a.jpg", "0000000000000000", sharp=200), row("b.jpg", b)]
    assert ("b.jpg" in judge(rows, dup_dist=8)) is is_dup


@pytest.mark.parametrize("gap,is_dup", [(19, True), (20, False)])
def test_dup_window_boundary(gap, is_dup):
    rows = [row("a.jpg", "ffff0000ffff0000", sharp=200)] + distinct(gap) + [row("b.jpg", "ffff0000ffff0000")]
    assert ("b.jpg" in judge(rows, window=20)) is is_dup


def test_dup_chains_into_one_group():
    # a と c は直接は遠いが、b を介してつながる
    rows = [row("a.jpg", "0000000000000000", sharp=10 ** 3), row("b.jpg", "000000000000003f"),
            row("c.jpg", "0000000000000fff")]
    assert judge(rows, dup_dist=6) == {"b.jpg": "dup:a.jpg", "c.jpg": "dup:a.jpg"}


def test_dup_takes_priority_over_other_reasons():
    rows = [row("a.jpg", "ffff0000ffff0000", sharp=200), row("Screenshot.png", "ffff0000ffff0000", make="")]
    assert judge(rows) == {"Screenshot.png": "dup:a.jpg"}


def test_empty():
    assert judge([]) == {}


# ---- judge: スクショ・誤撮影・ブレ ----

@pytest.mark.parametrize("kw,expect", [
    (dict(path="Screenshot 2024-01-01.jpg"), "screenshot"),
    (dict(path="スクリーンショット 1.jpg"), "screenshot"),
    (dict(comment="Screenshot"), "screenshot"),
    (dict(path="a.png", make=""), "screenshot"),
    (dict(path="a.png"), None),
    (dict(make="", p=(0.3, 0, 0, 0, 0.7)), "screenshot"),
    (dict(p=(0.3, 0, 0, 0, 0.7)), None),
    (dict(make=""), None),
])
def test_screenshot(kw, expect):
    r = row(**{"path": "a.jpg", **kw})
    assert judge([r]).get(r["path"]) == expect


@pytest.mark.parametrize("ground,pocket,is_junk", [(0.6, 0, True), (0.3, 0.3, True), (0.3, 0.29, False)])
def test_junk_threshold(ground, pocket, is_junk):
    r = row("a.jpg", p=(1 - ground - pocket, ground, pocket, 0, 0))
    assert judge([r]).get("a.jpg", "").startswith("junk") is is_junk


def test_without_clip_scores_only_sharpness_is_used():
    assert judge([row("a.jpg", p=None)]) == {}
    assert judge([row("a.jpg", sharp=5, p=None)]) == {"a.jpg": "blur:5"}


@pytest.mark.parametrize("sharp,blurry,is_blur", [
    (19, 0, True), (20, 0, False), (59, 0.7, True), (60, 0.7, False), (40, 0.5, False),
])
def test_blur(sharp, blurry, is_blur):
    r = row("a.jpg", sharp=sharp, p=(1 - blurry, 0, 0, blurry, 0))
    assert judge([r]).get("a.jpg", "").startswith("blur") is is_blur


def test_screenshot_before_junk_before_blur():
    assert judge([row("a.png", make="", sharp=1, p=(0, 1, 0, 0, 0))]) == {"a.png": "screenshot"}
    assert judge([row("a.jpg", sharp=1, p=(0, 1, 0, 0, 0))]) == {"a.jpg": "junk:1.00"}


# ---- review_items ----

def test_review_items_groups_with_keeper_unchecked():
    rows = [row("a.jpg"), row("b.jpg"), row("c.jpg"), row("d.jpg")]
    reasons = {"a.jpg": "dup:b.jpg", "c.jpg": "exact:a.jpg", "d.jpg": "blur:3"}
    assert pc.review_items(rows, reasons) == [
        ("dup", [("a.jpg", "dup:b.jpg", True), ("b.jpg", "dup:b.jpg", False), ("c.jpg", "exact:a.jpg", True)]),
        ("blur", [("d.jpg", "blur:3", True)]),
    ]


def test_review_items_exact_group():
    rows = [row("a.jpg"), row("b.jpg")]
    assert pc.review_items(rows, {"b.jpg": "exact:a.jpg"}) == [
        ("exact", [("a.jpg", "exact:a.jpg", False), ("b.jpg", "exact:a.jpg", True)])]


# ---- analyze ----

def scene(seed, size=(800, 600)):
    r = np.random.default_rng(seed)
    im = Image.new("RGB", size, tuple(int(x) for x in r.integers(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(40):
        x, y = int(r.integers(0, size[0])), int(r.integers(0, size[1]))
        d.rectangle([x, y, x + int(r.integers(20, 200)), y + int(r.integers(20, 200))],
                    fill=tuple(int(v) for v in r.integers(0, 255, 3)))
    return im


def save(im, path: Path, taken=None, make="Apple", comment=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if make:
        exif[271] = make
    sub = exif.get_ifd(0x8769)
    if taken:
        sub[36867] = taken
    if comment:
        sub[37510] = comment.encode()
    im.save(path, exif=exif) if path.suffix != ".png" else im.save(path)
    return path


def test_analyze_reads_exif_and_features(tmp_path):
    save(scene(1), tmp_path / "a.jpg", taken="2023:05:06 07:08:09", comment="Screenshot")
    f = pc.analyze((tmp_path, "a.jpg"))
    assert (f["taken"], f["make"], f["comment"], f["w"], f["h"]) == ("2023:05:06 07:08:09", "Apple", "Screenshot", 800, 600)
    assert len(f["phash"]) == 16 and len(f["sha1"]) == 40
    assert Image.open(pc.thumb_path(tmp_path, "a.jpg")).size == (256, 192)


def test_analyze_falls_back_to_mtime(tmp_path):
    save(scene(1), tmp_path / "a.jpg", make=None)
    f = pc.analyze((tmp_path, "a.jpg"))
    assert f["make"] == "" and len(f["taken"]) == 19


def test_analyze_sharpness_and_phash(tmp_path):
    save(scene(1), tmp_path / "a.jpg")
    scene(1).save(tmp_path / "b.jpg", quality=40)
    save(scene(1).filter(ImageFilter.GaussianBlur(20)), tmp_path / "c.jpg")
    a, b, c = (pc.analyze((tmp_path, n)) for n in ("a.jpg", "b.jpg", "c.jpg"))
    assert c["sharp"] < 20 < a["sharp"]
    assert pc.imagehash.hex_to_hash(a["phash"]) - pc.imagehash.hex_to_hash(b["phash"]) <= 8


def test_analyze_broken_file(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"not an image")
    assert pc.analyze((tmp_path, "x.jpg")) is None


# ---- scan / plan / apply ----

def fake_clip(root, rels):
    for r in rels:
        yield r, [0.05, 0.9, 0.02, 0.01, 0.02] if "ground" in r else list(NORMAL)


@pytest.fixture
def album(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "clip_scores", fake_clip)
    t = lambda m: f"2023:01:01 10:{m:02d}:00"
    for i in range(6):
        save(scene(i), tmp_path / f"IMG_{i}.jpg", taken=t(i * 2))
    save(scene(0).filter(ImageFilter.GaussianBlur(1)), tmp_path / "IMG_0b.jpg", taken=t(1))  # IMG_0 の近似
    (tmp_path / "IMG_0b.MOV").write_bytes(b"live")
    (tmp_path / "IMG_0bb.MOV").write_bytes(b"other")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "copy.jpg").write_bytes((tmp_path / "IMG_3.jpg").read_bytes())
    save(scene(9), tmp_path / "shot.png", make=None)
    save(scene(10), tmp_path / "ground.jpg", taken=t(20))
    save(scene(11).filter(ImageFilter.GaussianBlur(30)), tmp_path / "blurry.jpg", taken=t(30))
    (tmp_path / "broken.jpg").write_bytes(b"xx")
    (tmp_path / "notes.txt").write_text("x")
    save(scene(12), tmp_path / "_trash" / "old.jpg")
    return tmp_path


def db_paths(root):
    return {r[0] for r in sqlite3.connect(root / ".photoclean" / "cache.db").execute("select path from img")}


def plan(root):
    return pc.cmd_plan(root, window=20, dup_dist=8, blur_max=20, junk_prob=0.6)


def test_scan_caches_readable_images(album):
    pc.cmd_scan(album)
    assert db_paths(album) == {"IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg", "IMG_4.jpg", "IMG_5.jpg",
                               "IMG_0b.jpg", "sub/copy.jpg", "shot.png", "ground.jpg", "blurry.jpg"}
    row = sqlite3.connect(album / ".photoclean" / "cache.db").execute(
        "select p_ground from img where path='ground.jpg'").fetchone()
    assert row == (0.9,)


def test_scan_is_incremental_and_prunes(album, capfd, monkeypatch):
    pc.cmd_scan(album)
    capfd.readouterr()
    pc.cmd_scan(album)
    err = capfd.readouterr().err
    assert "1 枚を解析" in err and "broken.jpg" in err  # 読めない画像はキャッシュせず毎回試す

    (album / "IMG_5.jpg").unlink()
    save(scene(99), album / "IMG_4.jpg", taken="2023:01:01 10:08:00")
    clip_seen = []
    monkeypatch.setattr(pc, "clip_scores", lambda root, rels: (clip_seen.extend(rels), fake_clip(root, rels))[1])
    pc.cmd_scan(album)
    assert "2 枚を解析" in capfd.readouterr().err
    assert clip_seen == ["IMG_4.jpg"]
    assert "IMG_5.jpg" not in db_paths(album)


def test_plan_outputs(album):
    pc.cmd_scan(album)
    reasons = plan(album)
    assert reasons == {
        "IMG_0b.jpg": "dup:IMG_0.jpg",
        "sub/copy.jpg": "exact:IMG_3.jpg",
        "shot.png": "screenshot",
        "ground.jpg": "junk:0.92",
        "blurry.jpg": reasons["blurry.jpg"],
    }
    assert reasons["blurry.jpg"].startswith("blur:")
    wd = album / ".photoclean"
    assert dict(csv.reader(open(wd / "plan.csv"))) == {"path": "reason", **reasons}
    page = (wd / "review.html").read_text()
    assert page.count('<div class="group') == 2
    assert 'data-p="IMG_0.jpg" data-r="dup:IMG_0.jpg"><img' in page  # 残す側は未チェック


def test_plan_skips_files_deleted_after_scan(album):
    pc.cmd_scan(album)
    (album / "shot.png").unlink()
    assert "shot.png" not in plan(album)


def test_review_html_escapes_names(tmp_path):
    rows = [row('<b>"x".jpg', sharp=1)]
    page = pc.render_review(tmp_path, rows, pc.judge(rows, 20, 8, 20, 0.6))
    assert "<b>" not in page.split("<div class=grid>")[1]


def test_apply_moves_with_sidecars_and_logs(album):
    pc.cmd_scan(album)
    plan(album)
    assert pc.cmd_apply(album, album / ".photoclean" / "plan.csv") == 5
    trash = album / "_trash"
    assert sorted(str(p.relative_to(trash)) for p in trash.rglob("*") if p.is_file()) == [
        "blur/blurry.jpg", "dup/IMG_0b.MOV", "dup/IMG_0b.jpg", "exact/sub/copy.jpg", "junk/ground.jpg",
        "old.jpg", "screenshot/shot.png"]
    assert (album / "IMG_0bb.MOV").exists() and (album / "IMG_0.jpg").exists()
    log = list(csv.reader(open(album / ".photoclean" / "applied.csv")))
    assert ["IMG_0b.MOV", "_trash/dup/IMG_0b.MOV", "dup:IMG_0.jpg"] in log and len(log) == 6


def test_apply_never_overwrites(tmp_path):
    (tmp_path / "a.jpg").write_text("new")
    (tmp_path / "_trash" / "blur").mkdir(parents=True)
    (tmp_path / "_trash" / "blur" / "a.jpg").write_text("old")
    (tmp_path / "_trash" / "blur" / "a_1.jpg").write_text("old1")
    (tmp_path / "plan.csv").write_text("path,reason\na.jpg,blur:1\n")
    pc.cmd_apply(tmp_path, tmp_path / "plan.csv")
    assert (tmp_path / "_trash" / "blur" / "a.jpg").read_text() == "old"
    assert (tmp_path / "_trash" / "blur" / "a_2.jpg").read_text() == "new"


def test_apply_rejects_paths_outside_root(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    (tmp_path / "secret.jpg").write_text("x")
    (root / "_trash").mkdir()
    (root / "_trash" / "t.jpg").write_text("x")
    (root / "plan.csv").write_text("path,reason\n../secret.jpg,blur\n_trash/t.jpg,blur\nmissing.jpg,blur\n")
    assert pc.cmd_apply(root, root / "plan.csv") == 0
    assert (tmp_path / "secret.jpg").exists() and (root / "_trash" / "t.jpg").exists()


def test_apply_accepts_review_page_csv(tmp_path):
    """review.html の save() と同じ形式（全項目をダブルクォート）で保存した CSV を読める。"""
    (tmp_path / "a, b.jpg").write_text("x")
    (tmp_path / "plan.csv").write_text('path,reason\n"a, b.jpg","dup:c ""1"".jpg"\n')
    assert pc.cmd_apply(tmp_path, tmp_path / "plan.csv") == 1
    assert (tmp_path / "_trash" / "dup" / "a, b.jpg").exists()


def test_cli_end_to_end(album):
    pc.main(["scan", str(album)])
    pc.main(["plan", str(album), "--window", "0"])
    assert "IMG_0b.jpg" not in dict(csv.reader(open(album / ".photoclean" / "plan.csv")))
    pc.main(["apply", str(album)])
    assert (album / "_trash" / "screenshot" / "shot.png").exists()
