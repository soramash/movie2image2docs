# movie2image

Extract still images for each slide detected in a recorded presentation video using ffmpeg scene change detection.

This project provides a single script:
- extract_slides.py — a Python script that uses ffmpeg to detect slide changes and save one image per slide.

## Prerequisites
- macOS (other POSIX systems may also work)
- Python 3.9+ (3.8+ should also work)
- ffmpeg installed and available in PATH

Install ffmpeg (Homebrew):
```bash
brew install ffmpeg
```

## Quick start (recommended: vfr mode)
- Extract slide images from a single video using the default vfr mode:
```bash
python3 ./extract_slides.py "input.mov" --threshold 0.08 --out "slides-vfr-input"
```
- Process all .mov files in the current directory (zsh/bash):
```bash
for f in *.mov; do
  python3 ./extract_slides.py "$f" --threshold 0.08 --out "slides-vfr-${f%.*}"
done
```

Notes:
- Lower thresholds (e.g., 0.05–0.15) are more sensitive and yield more images.
- Output files are named like `slide-001.png`, `slide-002.png`, ... by default.
- First frame: The script includes the very first frame (n=0) by default so the opening slide is captured. Use `--no-include-first` if you want to skip it.

## Alternative: seek mode
Seek mode first detects slide-change timestamps, then seeks and saves one frame per change. This is useful if you want to control spacing between detections.
```bash
python3 ./extract_slides.py "input.mov" --mode seek --threshold 0.08 --min-gap 0.5 \
  --out "slides-seek-input"
```

## Options
- `--out <dir>`: Output directory (default: `slides`)
- `--threshold <0..1>`: Scene change threshold (default: `0.30`). Lower detects more slides.
- `--pattern <fmt>`: Filename pattern (printf style). Examples:
  - `slide-%03d.png` (default)
  - `frame-%04d.jpg` (save as JPEG)
- `--max-width <px>`: Optionally scale images to a maximum width while preserving aspect ratio.
- `--mode <vfr|seek>`: Extraction mode (default: `vfr`).
- `--include-first` / `--no-include-first`: Include or skip the very first frame (n=0). Default: include. In vfr mode this uses `select='eq(n,0)+gt(scene,THRESHOLD)'`; in seek mode it prepends `t=0.0`.
- `--min-gap <seconds>`: Minimum spacing between detections (seek mode only).

## Examples (generic)
- PNG output, vfr mode:
```bash
python3 ./extract_slides.py "input.mov" --threshold 0.10 --out "slides-vfr-input"
```
- JPG output with width limit:
```bash
python3 ./extract_slides.py "input.mov" --threshold 0.08 --max-width 1920 \
  --out "slides-vfr-input-jpg" --pattern "slide-%03d.jpg"
```
- Skip the very first frame (if you do not want the opening slide):
```bash
python3 ./extract_slides.py "input.mov" --threshold 0.10 --no-include-first --out "slides-vfr-input-no-first"
```
- Batch process multiple files (vfr mode):
```bash
for f in *.mov; do
  python3 ./extract_slides.py "$f" --threshold 0.10 --out "slides-vfr-${f%.*}"
done
```

## Troubleshooting
- First image missing: ensure `--include-first` (default) or avoid `--no-include-first`. For a raw ffmpeg command, include the first frame with:
  - `-vf "select='eq(n,0)+gt(scene,THRESHOLD)'" -vsync vfr`
- Too few images: lower `--threshold` (e.g., 0.08 → 0.05).
- Too many near-duplicates in seek mode: increase `--min-gap` (e.g., 1.0 → 2.0).
- `ffmpeg` not found: ensure `brew install ffmpeg` completed and `ffmpeg` is in PATH.

## Safety
- The script never modifies the input video.
- Outputs are written under the directory given via `--out`.

## 背景（Motivation）
研修資料がブラウザ上のスライドビューアで提供され、クリックでページをめくりながら学習する形式でした。これらのコンテンツはファイルとして配布されず、さらにサインイン用アカウントが約1ヶ月で無効化されるため、後日見直しや復習が難しいという課題がありました。

手作業で全スライドのスクリーンショットを撮るのは非効率なため、以下のワークフローで自動化する目的で本ツール群を作成しました。
- まず、最初のスライドから最後まで連続でページ送りする動画を1本作成
- 動画からスライド切り替わりを検出して、1ページ1枚の静止画を自動抽出（`extract_slides.py`）
- 抽出した画像の文字情報を OpenAI API でMarkdownとして抽出し、画像とともに1つのファイルへまとめる（例: Word/.docx via `word_from_sections.py`）

本リポジトリは「学習・記録」のために個人的に見返せる形に整えることを主目的としています。ご利用の際は、提供元のサービス利用規約・著作権・機密情報の取り扱いに必ず留意し、権利上問題のない範囲でのみ実行してください。
