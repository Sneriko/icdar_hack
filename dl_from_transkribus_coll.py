#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


TOKEN_URL = "https://account.readcoop.eu/auth/realms/readcoop/protocol/openid-connect/token"
API_BASE = "https://transkribus.eu/TrpServer/rest"


class TranskribusClient:
    def __init__(self, username: str, password: str, timeout: int = 60) -> None:
        self.username = username
        self.password = password
        self.timeout = timeout

        self.session = requests.Session()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

    def login(self) -> None:
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "transkribus-api-client",
        }
        r = self.session.post(TOKEN_URL, data=data, timeout=self.timeout)
        r.raise_for_status()
        tok = r.json()

        self.access_token = tok["access_token"]
        self.refresh_token = tok.get("refresh_token")
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
            }
        )

    def refresh_access_token(self) -> None:
        if not self.refresh_token:
            raise RuntimeError("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": "transkribus-api-client",
        }
        r = self.session.post(TOKEN_URL, data=data, timeout=self.timeout)
        r.raise_for_status()
        tok = r.json()

        self.access_token = tok["access_token"]
        self.refresh_token = tok.get("refresh_token", self.refresh_token)
        self.session.headers.update(
            {"Authorization": f"Bearer {self.access_token}"}
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        r = self.session.request(method, url, timeout=self.timeout, **kwargs)

        if r.status_code == 401 and self.refresh_token:
            self.refresh_access_token()
            r = self.session.request(method, url, timeout=self.timeout, **kwargs)

        r.raise_for_status()
        return r

    def get_json(self, path: str) -> Any:
        url = f"{API_BASE}/{path.lstrip('/')}"
        r = self._request("GET", url)
        return r.json()

    def get_binary(self, url: str) -> bytes:
        r = self._request("GET", url)
        return r.content

    def list_documents(self, collection_id: int | str) -> List[Dict[str, Any]]:
        data = self.get_json(f"collections/{collection_id}/list")
        if not isinstance(data, list):
            raise TypeError(f"Unexpected documents payload: {type(data)!r}")
        return data

    def get_fulldoc(self, collection_id: int | str, document_id: int | str) -> Dict[str, Any]:
        data = self.get_json(f"collections/{collection_id}/{document_id}/fulldoc")
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected fulldoc payload: {type(data)!r}")
        return data


def safe_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip() or "unnamed"


def pick_latest_transcript(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ts_list = (((page.get("tsList") or {}).get("transcripts")) or [])
    if not ts_list:
        return None

    def sort_key(ts: Dict[str, Any]) -> tuple:
        ts_id = ts.get("tsId")
        stamp = (
            ts.get("timestamp")
            or ts.get("timeStamp")
            or ts.get("created")
            or ts.get("lastModified")
            or ""
        )
        return (int(ts_id) if str(ts_id).isdigit() else -1, str(stamp))

    return sorted(ts_list, key=sort_key, reverse=True)[0]


def page_is_ground_truth(page: Dict[str, Any]) -> bool:
    ts = pick_latest_transcript(page)
    if not ts:
        return False
    return str(ts.get("status", "")).upper() == "GT"


def download_page_assets(
    client: TranskribusClient,
    page: Dict[str, Any],
    out_dir: Path,
    gt_only: bool,
    skip_existing: bool,
    save_meta: bool,
) -> bool:
    if gt_only and not page_is_ground_truth(page):
        return False

    img_url = page.get("url")
    img_name = page.get("imgFileName") or f"page_{page.get('pageNr', 'unknown')}.jpg"

    ts = pick_latest_transcript(page)
    xml_url = ts.get("url") if ts else None

    if not img_url:
        print(f"Skipping page without image URL: pageNr={page.get('pageNr')}", file=sys.stderr)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / safe_name(img_name)
    xml_path = img_path.with_suffix(".xml")

    if not (skip_existing and img_path.exists()):
        img_path.write_bytes(client.get_binary(img_url))

    if xml_url and not (skip_existing and xml_path.exists()):
        xml_path.write_bytes(client.get_binary(xml_url))

    if save_meta:
        meta = {
            "pageNr": page.get("pageNr"),
            "pageId": page.get("pageId"),
            "imgFileName": page.get("imgFileName"),
            "imgUrl": img_url,
            "transcript": ts,
            "is_ground_truth": page_is_ground_truth(page),
        }
        meta_path = img_path.with_suffix(".json")
        if not (skip_existing and meta_path.exists()):
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return True


def download_document(
    client: TranskribusClient,
    collection_id: int | str,
    document_id: int | str,
    out_root: Path,
    gt_only: bool,
    skip_existing: bool,
    save_meta: bool,
) -> int:
    fulldoc = client.get_fulldoc(collection_id, document_id)

    doc_title = (
        fulldoc.get("md", {}).get("title")
        or fulldoc.get("title")
        or f"doc_{document_id}"
    )
    doc_dir = out_root / safe_name(f"{document_id}_{doc_title}")

    pages = (((fulldoc.get("pageList") or {}).get("pages")) or [])
    count = 0
    for page in pages:
        if download_page_assets(
            client=client,
            page=page,
            out_dir=doc_dir,
            gt_only=gt_only,
            skip_existing=skip_existing,
            save_meta=save_meta,
        ):
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download page images and PAGE XML from a Transkribus collection."
    )

    parser.add_argument("--username", required=True, help="Transkribus username/email")
    parser.add_argument("--password", required=True, help="Transkribus password")
    parser.add_argument("--collection-id", required=True, type=int, help="Collection ID")

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where files will be saved",
    )
    parser.add_argument(
        "--doc-id",
        type=int,
        help="Optional single document ID to download instead of the whole collection",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Only download pages whose latest transcript has status GT",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Pause between documents, e.g. 0.1",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--save-meta",
        action="store_true",
        help="Save a JSON sidecar with page metadata next to each image",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = TranskribusClient(
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )
    client.login()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_pages_saved = 0

    if args.doc_id is not None:
        print(f"Downloading document {args.doc_id} from collection {args.collection_id}")
        saved = download_document(
            client=client,
            collection_id=args.collection_id,
            document_id=args.doc_id,
            out_root=out_root,
            gt_only=args.gt_only,
            skip_existing=args.skip_existing,
            save_meta=args.save_meta,
        )
        total_pages_saved += saved
        print(f"Saved {saved} page(s)")
    else:
        docs = client.list_documents(args.collection_id)
        print(f"Found {len(docs)} document(s) in collection {args.collection_id}")

        for i, doc in enumerate(docs, start=1):
            doc_id = doc.get("docId")
            title = doc.get("title", f"doc_{doc_id}")
            print(f"[{i}/{len(docs)}] {doc_id}: {title}")

            try:
                saved = download_document(
                    client=client,
                    collection_id=args.collection_id,
                    document_id=doc_id,
                    out_root=out_root,
                    gt_only=args.gt_only,
                    skip_existing=args.skip_existing,
                    save_meta=args.save_meta,
                )
                total_pages_saved += saved
                print(f"    saved {saved} page(s)")
            except requests.HTTPError as e:
                print(f"    ERROR: {e}", file=sys.stderr)

            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

    print(f"Done. Total saved pages: {total_pages_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())