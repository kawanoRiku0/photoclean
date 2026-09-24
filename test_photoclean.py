from photoclean import judge


def row(path, phash, sha1=None, sharp=100.0, make="Apple", p=(0.9, 0.02, 0.02, 0.02, 0.04)):
    return dict(path=path, phash=phash, sha1=sha1 or path, sharp=sharp, w=4000, h=3000, make=make, comment="",
                p_photo=p[0], p_ground=p[1], p_pocket=p[2], p_blurry=p[3], p_screen=p[4])


def test_judge():
    rows = [
        row("a.jpg", "ffff0000ffff0000", sharp=200),
        row("b.jpg", "ffff0000ffff0001", sharp=150),   # a と近似 → a を残す
        row("c.jpg", "0f0f0f0f0f0f0f0f", sha1="a.jpg"),  # a と完全一致
        row("d.png", "1234123412341234", make=""),
        row("e.jpg", "abcdabcdabcdabcd", p=(0.1, 0.8, 0.05, 0.03, 0.02)),
        row("f.jpg", "5555aaaa5555aaaa", sharp=1),
    ] + [row(f"z{i}.jpg", f"{i * 7919:016x}"[::-1], sharp=100 + i) for i in range(30)]
    r = judge(rows, window=20, dup_dist=8, blur_max=20, junk_prob=0.6)
    assert r["b.jpg"] == "dup:a.jpg"
    assert r["c.jpg"] == "exact:a.jpg"
    assert r["d.png"] == "screenshot"
    assert r["e.jpg"].startswith("junk")
    assert r["f.jpg"].startswith("blur")
    assert "a.jpg" not in r


def test_window():
    far = [row("a.jpg", "ffff0000ffff0000")] + [row(f"z{i}.jpg", f"{(i + 1) * 104729:016x}") for i in range(25)]
    far.append(row("b.jpg", "ffff0000ffff0000"))
    assert "b.jpg" not in judge(far, window=20, dup_dist=0, blur_max=0, junk_prob=0.6)
