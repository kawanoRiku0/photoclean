# photoclean

写真フォルダから「重複」「ブレ」「スクショ」「地面やポケットの中を撮ってしまった誤撮影」を検出し、`_trash/` に移動するスクリプトです。

判定はすべてローカルで行います（pHash・Laplacian 分散・CLIP）。LLM に画像を見せないので、枚数が増えても API コストはかかりません。5,600 枚で約 1 分です（Apple M3 Pro）。

## 必要なもの

- [uv](https://docs.astral.sh/uv/)（依存ライブラリはスクリプト先頭のメタデータから自動で入ります）
- 初回のみ CLIP モデル（ViT-B-32、約 600MB）をダウンロードします

## 使い方

```sh
uv run photoclean.py scan  ~/Pictures/photos   # 特徴量を計算して .photoclean/ にキャッシュ
uv run photoclean.py plan  ~/Pictures/photos   # 判定して review.html と plan.csv を出力
open ~/Pictures/photos/.photoclean/review.html # 確認して「plan.csv を保存」
uv run photoclean.py apply ~/Pictures/photos ~/Downloads/plan.csv
```

1. `scan` は画像ごとの SHA1・pHash・シャープネス・EXIF・CLIP スコアを `.photoclean/cache.db` に保存します。2 回目以降は追加・変更されたファイルだけを解析します。
2. `plan` はキャッシュだけを使うので数秒で終わります。閾値を変えて何度でも再実行できます。
3. `review.html` では赤枠が移動、緑枠が残すです。クリックで切り替えます。重複は横一列が 1 グループで、どれを残すかを選べます。
4. `apply` は削除せず `_trash/<理由>/` に移動します。移動の記録は `.photoclean/applied.csv` に残ります。Live Photo の `.MOV` や `.AAE` が同名で並んでいれば一緒に移動します。

`apply` に CSV を渡さない場合は `.photoclean/plan.csv`（レビュー前の全候補）が使われます。

## 判定ルール

| 理由 | 条件 |
|---|---|
| `exact` | SHA1 が一致（フォルダ全体で比較） |
| `dup` | EXIF の撮影日時順に並べ、直近 `--window` 枚の中で pHash のハミング距離が `--dup-dist` 以下。似たもの同士を 1 グループにまとめ、一番シャープな 1 枚を残す |
| `screenshot` | ファイル名に screenshot 等を含む / EXIF の UserComment が Screenshot / カメラ機種が無く PNG または CLIP で画面と判定 |
| `junk` | CLIP の「地面」「ポケットの中」の確率の合計が `--junk-prob` 以上 |
| `blur` | シャープネス（512px に縮小した画像の Laplacian 分散）が `--blur-max` 未満。3 倍未満のときだけ CLIP の「ぼやけ」も使う |

## 閾値の調整

| オプション | 既定値 | 調整の目安 |
|---|---|---|
| `--window` | 20 | 比較する直近の枚数。連写が長いなら増やす |
| `--dup-dist` | 8 | 違う写真が重複に混ざるなら下げる（6 など） |
| `--blur-max` | 20 | ブレを見逃すなら上げる |
| `--junk-prob` | 0.6 | 誤撮影を見逃すなら下げる |

まず数百枚の連続したサンプルで試し、候補と閾値付近の値を見てから全体に適用するのがおすすめです。閾値付近の値はキャッシュから確認できます。

```sh
sqlite3 .photoclean/cache.db "select path, sharp, p_blurry from img order by sharp limit 10"
sqlite3 .photoclean/cache.db "select path, p_ground + p_pocket from img order by 2 desc limit 10"
```

## 対応形式

`.jpg` `.jpeg` `.png` `.heic` `.heif` `.webp`。動画は判定しません。

## テスト

```sh
uv run --python 3.12 --no-project --with pytest,pillow,pillow-heif,imagehash,opencv-python-headless,numpy pytest -q
```
