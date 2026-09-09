from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import normalize_extracted_tree, validate_dataset, write_validation_report

DEFAULT_URL = "https://cloud.mail.ru/public/GCsv/1BXmZPEBj"
DEFAULT_FILENAME = "data_освещённость.zip"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AIIJC-dataset-downloader/2.0"
ZIP_PREFIXES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


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


def extract_filename_from_html(html: str) -> str | None:
    patterns = [
        r"<title>\s*(.*?)\s*/\s*Облако\s+Mail\s*</title>",
        r'"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*\.zip)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.I | re.S)
        if not match:
            continue
        raw = match.group(1).replace("\\/", "/")
        if "\\u" in raw:
            try:
                raw = json.loads(f'"{raw}"')
            except Exception:
                pass
        value = html_lib.unescape(raw).strip()
        if value:
            return value
    return None


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


def _request_bytes(url: str, referer: str | None = None, timeout: int = 60) -> bytes:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "ru,en;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _request_json(url: str, referer: str | None = None) -> dict:
    return json.loads(_request_bytes(url, referer=referer).decode("utf-8"))


def _legacy_dispatcher_storage() -> str | None:
    try:
        text = _request_bytes("https://dispatcher.cloud.mail.ru/G", timeout=30).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return None
    match = re.search(r"https?://[^\s]+", text)
    return match.group(0).strip() if match else None


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def build_download_candidates(
    public_url: str,
    dispatcher_payload: dict,
    token_payload: dict | None = None,
    filename: str | None = None,
    extra_storage_urls: list[str] | None = None,
) -> list[str]:
    public_id = extract_public_id(public_url)
    endpoints = dispatcher_payload.get("body", {}).get("weblink_get", [])
    storages = [str(item.get("url", "")).strip() for item in endpoints if item.get("url")]
    storages.extend(extra_storage_urls or [])
    storages = _unique(storages)
    if not storages:
        raise ValueError("no Mail.ru weblink_get storage endpoints were resolved")

    token = _extract_token(token_payload)
    encoded_filename = urllib.parse.quote(filename, safe="") if filename else None
    candidates: list[str] = []

    def add(base: str, suffix: str) -> None:
        base = base.rstrip("/")
        suffix = suffix.lstrip("/")
        value = f"{base}/{suffix}"
        if token:
            value += ("&" if "?" in value else "?") + "key=" + urllib.parse.quote(token)
        if value not in candidates:
            candidates.append(value)

    for storage in storages:
        parsed = urllib.parse.urlparse(storage)
        # Some page snapshots expose a public host rather than /weblink/get.
        if "/public/" in parsed.path and "/weblink/get" not in parsed.path:
            base = f"{parsed.scheme}://{parsed.netloc}/public"
            add(base, public_id)
            if encoded_filename:
                add(base, f"{public_id}/{encoded_filename}")
            continue

        # Mail.ru has used both forms over time. Keep both: the first is the
        # classic direct link, the second is required by several public-file
        # configurations where omitting the visible filename returns 404.
        add(storage, public_id)
        if encoded_filename:
            add(storage, f"{public_id}/{encoded_filename}")

    return candidates


def build_download_url(
    public_url: str,
    dispatcher_payload: dict,
    token_payload: dict | None = None,
    filename: str | None = None,
) -> str:
    return build_download_candidates(
        public_url,
        dispatcher_payload,
        token_payload=token_payload,
        filename=filename,
    )[0]


def resolve_download_candidates(public_url: str) -> tuple[list[str], str]:
    html = _request_bytes(public_url).decode("utf-8", errors="replace")
    filename = extract_filename_from_html(html) or DEFAULT_FILENAME

    storage_urls: list[str] = []
    storage_html = _storage_url_from_html(html)
    if storage_html:
        storage_urls.append(storage_html)

    token_payload = None
    try:
        token_payload = _request_json(
            "https://cloud.mail.ru/api/v2/tokens/download",
            referer=public_url,
        )
    except Exception as exc:
        print(f"Mail.ru download token unavailable: {type(exc).__name__}: {exc}", flush=True)

    dispatcher: dict = {"body": {"weblink_get": []}}
    try:
        dispatcher_api = _request_json(
            "https://cloud.mail.ru/api/v2/dispatcher",
            referer=public_url,
        )
        api_endpoints = dispatcher_api.get("body", {}).get("weblink_get", [])
        dispatcher["body"]["weblink_get"].extend(api_endpoints)
    except Exception as exc:
        print(f"Mail.ru v2 dispatcher unavailable: {type(exc).__name__}: {exc}", flush=True)

    legacy = _legacy_dispatcher_storage()
    if legacy:
        storage_urls.append(legacy)

    if storage_urls:
        dispatcher["body"]["weblink_get"].extend({"url": x} for x in storage_urls)

    candidates = build_download_candidates(
        public_url,
        dispatcher,
        token_payload=token_payload,
        filename=filename,
    )
    return candidates, filename


def _masked_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = [(key, "***" if key.lower() == "key" else value) for key, value in query]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(safe_query), parsed.fragment)
    )


def _stream_zip_candidate(url: str, public_url: str, destination: Path) -> bool:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": public_url,
        "Accept": "application/zip,application/octet-stream,*/*",
        "Accept-Language": "ru,en;q=0.8",
    }
    request = urllib.request.Request(url, headers=headers)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            prefix = response.read(4)
            if prefix not in ZIP_PREFIXES:
                sample = prefix + response.read(124)
                print(
                    f"Rejected non-ZIP Mail.ru response from {_masked_url(url)}: {sample!r}",
                    flush=True,
                )
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as out:
                out.write(prefix)
                shutil.copyfileobj(response, out, length=4 * 1024 * 1024)
        tmp.replace(destination)
        return True
    except urllib.error.HTTPError as exc:
        print(f"Mail.ru candidate {_masked_url(url)} -> HTTP {exc.code}", flush=True)
        return False
    except Exception as exc:
        print(
            f"Mail.ru candidate {_masked_url(url)} failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    finally:
        if tmp.exists() and not destination.exists():
            tmp.unlink(missing_ok=True)


def download_file(public_url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates, filename = resolve_download_candidates(public_url)
    print(f"Resolved public filename: {filename}", flush=True)
    print(f"Trying {len(candidates)} Mail.ru direct-link variants", flush=True)
    for idx, url in enumerate(candidates, 1):
        print(f"[{idx}/{len(candidates)}] {_masked_url(url)}", flush=True)
        if _stream_zip_candidate(url, public_url, destination):
            print(f"Downloaded {destination} ({destination.stat().st_size} bytes)", flush=True)
            return destination
    raise RuntimeError(
        "All Mail.ru direct-link strategies failed. "
        "The public page was reachable, but none of the current weblink_get variants returned a ZIP."
    )


def _extract_zip(zip_path: Path, target: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        prefix = zip_path.read_bytes()[:256]
        raise RuntimeError(
            "Mail.ru response is not a ZIP archive. First bytes: " + repr(prefix)
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
    parser.add_argument("--zip-path", type=Path, default=PROJECT_ROOT / DEFAULT_FILENAME)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.zip_path.exists():
        print(f"Resolving and downloading {args.url}", flush=True)
        download_file(args.url, args.zip_path)
    else:
        print(f"Using existing archive {args.zip_path} ({args.zip_path.stat().st_size} bytes)", flush=True)

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
