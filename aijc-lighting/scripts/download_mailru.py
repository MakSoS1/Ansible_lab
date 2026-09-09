from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import normalize_extracted_tree, validate_dataset, write_validation_report

DEFAULT_URL = "https://cloud.mail.ru/public/GCsv/1BXmZPEBj"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AIIJC-dataset-downloader/1.0"


def extract_public_id(public_url: str) -> str:
    parsed = urllib.parse.urlparse(public_url)
    if parsed.scheme != "https" or parsed.netloc != "cloud.mail.ru":
        raise ValueError("public URL must use https://cloud.mail.ru")
    marker = "/public/"
    if marker not in parsed.path:
        raise ValueError("public URL must contain /public/")
    public_id = parsed.path.split(marker, 1)[1].strip("/")
    parts = public_id.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError("public URL does not contain a two-part public id")
    return "/".join(parts[:2])


def _extract_token(token_payload: dict | None) -> str | None:
    if not token_payload:
        return None
    body = token_payload.get("body")
    if isinstance(body, dict):
        value = body.get("token")
        if value:
            return str(value)
    if isinstance(body, str) and body:
        return body
    value = token_payload.get("token")
    return str(value) if value else None


def build_download_url(
    public_url: str,
    dispatcher_payload: dict,
    token_payload: dict | None = None,
) -> str:
    public_id = extract_public_id(public_url)
    endpoints = dispatcher_payload.get("body", {}).get("weblink_get", [])
    if not endpoints:
        raise ValueError("dispatcher payload does not contain body.weblink_get")
    storage_url = str(endpoints[0].get("url", "")).strip()
    if not storage_url:
        raise ValueError("dispatcher weblink_get endpoint is empty")

    parsed = urllib.parse.urlparse(storage_url)
    if "/public/" in parsed.path and "/weblink/get" not in parsed.path:
        base = f"{parsed.scheme}://{parsed.netloc}"
        url = f"{base}/public/{public_id}"
    else:
        url = f"{storage_url.rstrip('/')}/{public_id}"

    token = _extract_token(token_payload)
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={urllib.parse.quote(token)}"
    return url


def _request_bytes(url: str, referer: str | None = None, timeout: int = 60) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _request_json(url: str, referer: str | None = None) -> dict:
    return json.loads(_request_bytes(url, referer=referer).decode("utf-8"))


def _storage_url_from_html(html: str) -> str | None:
    patterns = [
        r'"weblink_get"\s*:\s*\[\s*\{.*?"url"\s*:\s*"([^"]+)"',
        r'weblink_get.*?"url"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.S)
        if match:
            return match.group(1).replace("\\/", "/")
    return None


def resolve_download_url(public_url: str) -> str:
    html = _request_bytes(public_url).decode("utf-8", errors="replace")
    storage = _storage_url_from_html(html)

    token_payload = None
    try:
        token_payload = _request_json(
            "https://cloud.mail.ru/api/v2/tokens/download",
            referer=public_url,
        )
    except Exception:
        token_payload = None

    if storage:
        dispatcher = {"body": {"weblink_get": [{"url": storage}]}}
        return build_download_url(public_url, dispatcher, token_payload)

    dispatcher = _request_json(
        "https://cloud.mail.ru/api/v2/dispatcher",
        referer=public_url,
    )
    return build_download_url(public_url, dispatcher, token_payload)


def download_file(public_url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_url = resolve_download_url(public_url)
    headers = {"User-Agent": USER_AGENT, "Referer": public_url, "Accept": "*/*"}
    request = urllib.request.Request(download_url, headers=headers)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(destination)
    return destination


def _extract_zip(zip_path: Path, target: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        prefix = zip_path.read_bytes()[:256]
        raise RuntimeError(
            "Mail.ru response is not a ZIP archive. First bytes: "
            + repr(prefix)
        )
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupted member in downloaded ZIP: {bad}")
        archive.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--zip-path", type=Path, default=PROJECT_ROOT / "data_освещённость.zip")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.zip_path.exists():
        print(f"Resolving and downloading {args.url}", flush=True)
        download_file(args.url, args.zip_path)
    else:
        print(f"Using existing archive {args.zip_path}", flush=True)

    extract_tmp = args.output_dir.parent / ".mailru-extract"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True)
    _extract_zip(args.zip_path, extract_tmp)
    layout = normalize_extracted_tree(extract_tmp)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(layout.root, args.output_dir)
    shutil.rmtree(extract_tmp)

    final_layout = normalize_extracted_tree(args.output_dir)
    report = validate_dataset(final_layout)
    write_validation_report(report, args.output_dir / "dataset_validation.json")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if not args.keep_zip and args.zip_path.exists():
        args.zip_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
