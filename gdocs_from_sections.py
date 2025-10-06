#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import concurrent.futures
import io
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# OpenAI
try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore

# Google APIs
try:
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials as SACredentials  # noqa: E999
    from google.oauth2.credentials import Credentials as UserCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
except Exception as e:  # pragma: no cover
    pass

# Markdown
try:
    import markdown
except Exception as e:  # pragma: no cover
    markdown = None  # type: ignore

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


@dataclass
class OcrResult:
    image_path: Path
    markdown_text: str
    data_uri: str


def guess_mime(path: Path) -> str:
    return MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")


def image_to_data_uri(path: Path) -> str:
    mime = guess_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_section_dirs(root: Path, pattern: str) -> List[Path]:
    # glob with pattern relative to root
    return sorted([p for p in root.glob(pattern) if p.is_dir()])


def list_images(section_dir: Path) -> List[Path]:
    files = [p for p in sorted(section_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return files


def load_cached_md(cache_dir: Path, section: str, image_name: str) -> Optional[str]:
    cache_path = cache_dir / section / f"{image_name}.md"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def save_cached_md(cache_dir: Path, section: str, image_name: str, md: str) -> None:
    cache_path = cache_dir / section / f"{image_name}.md"
    ensure_dir(cache_path.parent)
    cache_path.write_text(md, encoding="utf-8")


def ocr_markdown_with_openai(image_path: Path, model: str, language_hint: str = "ja") -> str:
    if OpenAI is None:
        raise RuntimeError("openai SDK is not installed. Install dependencies from requirements-gdocs.txt")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")

    client = OpenAI()

    data_uri = image_to_data_uri(image_path)

    prompt = (
        "次の画像から可読な文字情報を正確にMarkdownで抽出してください。\n"
        "要件:\n"
        "- 出力は有効なMarkdownのみ（前置き/後置きの説明は禁止）\n"
        "- 見出し/強調/箇条書き/番号リスト/表は可能な限りMarkdownで再現\n"
        "- 不要なノイズや透かしは除外\n"
        "- 言語は原文のまま（{lang} を優先）\n"
        "- 画像に文字がない場合は (テキストなし) とだけ出力\n"
    ).format(lang=language_hint)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise OCR that outputs only Markdown."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI OCR failed for {image_path}: {e}")


def md_to_html(md: str) -> str:
    if markdown is None:
        raise RuntimeError("markdown package is not installed. Install dependencies from requirements-gdocs.txt")
    # Enable tables and sane_list handling
    return markdown.markdown(md, extensions=["tables", "sane_lists"])  # type: ignore


def build_section_html(title: str, results: List[OcrResult]) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"ja\">",
        "<head>",
        "  <meta charset=\"UTF-8\">",
        f"  <title>{title}</title>",
        "  <style>\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans JP', Arial, sans-serif; line-height: 1.6; padding: 24px; }\n"
        "    h1 { border-bottom: 1px solid #ddd; padding-bottom: 6px; }\n"
        "    img { max-width: 100%; height: auto; margin: 8px 0 16px; }\n"
        "    .item { margin: 40px 0; }\n"
        "    .filename { color: #555; font-size: 0.9em; }\n"
        "    table { border-collapse: collapse; }\n"
        "    table, th, td { border: 1px solid #aaa; }\n"
        "    th, td { padding: 6px 8px; }\n"
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
    ]

    for idx, r in enumerate(results, 1):
        md_html = md_to_html(r.markdown_text)
        parts += [
            "  <div class=\"item\">",
            f"    <h2>{idx:02d}. {r.image_path.name}</h2>",
            f"    <div class=\"filename\">{r.image_path}</div>",
            f"    <img src=\"{r.data_uri}\" alt=\"{r.image_path.name}\">",
            "    <h3>抽出テキスト（Markdownを反映）</h3>",
            f"    <div class=\"md\">{md_html}</div>",
            "  </div>",
        ]

    parts += ["</body>", "</html>"]
    return "\n".join(parts)


def get_drive_service() -> "googleapiclient.discovery.Resource":  # type: ignore[name-defined]
    """Obtain a Drive service using one of:
    - Application Default Credentials (gcloud auth application-default login)
    - Service Account via GOOGLE_SERVICE_ACCOUNT_JSON env var (path)
    - Installed App OAuth (credentials.json -> token.json)
    """
    creds = None
    # 1) Try ADC
    try:
        creds, _ = google.auth.default(scopes=SCOPES)  # type: ignore[name-defined]
        if creds and creds.expired and getattr(creds, "refresh_token", None):
            creds.refresh(Request())  # type: ignore[name-defined]
    except Exception:
        creds = None

    # 2) Service Account
    if not creds:
        sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if sa_path and Path(sa_path).exists():
            creds = SACredentials.from_service_account_file(sa_path, scopes=SCOPES)  # type: ignore[name-defined]

    # 3) Installed App OAuth
    if not creds:
        token_path = Path("token.json")
        if token_path.exists():
            try:
                creds = UserCredentials.from_authorized_user_file(str(token_path), SCOPES)  # type: ignore[name-defined]
            except Exception:
                creds = None
        if not creds or not getattr(creds, "valid", False):
            if creds and getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
                creds.refresh(Request())  # type: ignore[name-defined]
            else:
                cred_client_path = Path("credentials.json")
                if cred_client_path.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(str(cred_client_path), SCOPES)  # type: ignore[name-defined]
                    creds = flow.run_local_server(port=0)
                    token_path.write_text(creds.to_json(), encoding="utf-8")

    if not creds:
        raise RuntimeError(
            "Google Driveの認証情報が見つかりません。以下のいずれかで設定してください:\n"
            "- gcloud auth application-default login を実行\n"
            "- GOOGLE_SERVICE_ACCOUNT_JSON にサービスアカウントJSONのパスを指定\n"
            "- credentials.json を配置（OAuthクライアントID）→初回実行時にブラウザで認可\n"
        )

    service = build("drive", "v3", credentials=creds, cache_discovery=False)  # type: ignore[name-defined]
    return service


def upload_html_as_gdoc(service, title: str, html: str, parent_id: Optional[str]) -> Dict:
    meta: Dict[str, object] = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
    }
    if parent_id:
        meta["parents"] = [parent_id]

    bio = io.BytesIO(html.encode("utf-8"))
    media = MediaIoBaseUpload(bio, mimetype="text/html", resumable=False)

    file = (
        service.files()  # type: ignore[attr-defined]
        .create(body=meta, media_body=media, fields="id, name, webViewLink", supportsAllDrives=True)
        .execute()
    )
    return file


def process_section(
    section_dir: Path,
    model: str,
    cache_dir: Path,
    language_hint: str,
    max_workers: int,
) -> Tuple[str, List[OcrResult]]:
    images = list_images(section_dir)
    if not images:
        logging.warning(f"画像が見つかりません: {section_dir}")
        return (section_dir.name, [])

    results: List[OcrResult] = []

    def _one(img_path: Path) -> OcrResult:
        cached = load_cached_md(cache_dir, section_dir.name, img_path.name)
        if cached is None:
            md = ocr_markdown_with_openai(img_path, model=model, language_hint=language_hint)
            save_cached_md(cache_dir, section_dir.name, img_path.name, md)
        else:
            md = cached
        data_uri = image_to_data_uri(img_path)
        return OcrResult(image_path=img_path, markdown_text=md, data_uri=data_uri)

    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for r in ex.map(_one, images):
                results.append(r)
    else:
        for img in images:
            results.append(_one(img))

    return (section_dir.name, results)


def sanitize_title(name: str, prefix: Optional[str] = None) -> str:
    title = name
    title = re.sub(r"[\\/:*?\"<>|]", "_", title)
    if prefix:
        title = f"{prefix} {title}"
    return title


def main() -> None:
    parser = argparse.ArgumentParser(description="セクション画像からGoogleドキュメントを生成（OpenAI OCR使用）")
    parser.add_argument("--root", default=".", help="探索ルートディレクトリ（デフォルト: .）")
    parser.add_argument("--pattern", default="slides-vfr-IPA-20250930_section0*", help="セクションフォルダのglobパターン")
    parser.add_argument("--drive-parent-id", dest="parent_id", default=None, help="作成先のGoogle DriveフォルダID（任意）")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAIのビジョン対応モデル名（例: gpt-4o-mini）")
    parser.add_argument("--language", default="ja", help="OCRの言語ヒント（ja/enなど）")
    parser.add_argument("--max-workers", type=int, default=2, help="並列OCR数（初期値2）")
    parser.add_argument("--skip-upload", action="store_true", help="DriveにアップロードせずHTMLのみ出力")
    parser.add_argument("--title-prefix", default=None, help="ドキュメントタイトルの先頭に付ける文字列")
    parser.add_argument("--cache-dir", default=".cache/ocr-md", help="OCR結果キャッシュ保存先")
    parser.add_argument("--out-dir", default="build", help="--skip-upload時のHTML出力先")
    parser.add_argument("--log-level", default="INFO", help="ログレベル（DEBUG/INFO/WARN/ERROR）")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    root = Path(args.root).resolve()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(cache_dir)
    ensure_dir(out_dir)

    sections = find_section_dirs(root, args.pattern)
    if not sections:
        logging.error(f"セクションフォルダが見つかりません: root={root}, pattern={args.pattern}")
        sys.exit(1)

    service = None
    if not args.skip_upload:
        service = get_drive_service()

    summary: List[Tuple[str, Optional[str]]] = []

    for sec in sections:
        sec_name, results = process_section(
            section_dir=sec,
            model=args.model,
            cache_dir=cache_dir,
            language_hint=args.language,
            max_workers=args.max_workers,
        )
        title = sanitize_title(sec_name, args.title_prefix)
        html = build_section_html(title, results)

        if args.skip_upload:
            html_path = out_dir / f"{sec_name}.html"
            html_path.write_text(html, encoding="utf-8")
            logging.info(f"HTMLを書き出しました: {html_path}")
            summary.append((title, None))
        else:
            file = upload_html_as_gdoc(service, title=title, html=html, parent_id=args.parent_id)  # type: ignore[arg-type]
            link = file.get("webViewLink")  # type: ignore[assignment]
            logging.info(f"Googleドキュメントを作成しました: {title} → {link}")
            summary.append((title, link))

    print("\n=== 生成結果 ===")
    for title, link in summary:
        if link:
            print(f"- {title}: {link}")
        else:
            print(f"- {title}: (HTMLのみ出力)")


if __name__ == "__main__":
    main()
