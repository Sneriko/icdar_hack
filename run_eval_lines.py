"""
Run a trained TrOCR checkpoint on evaluation text lines from local PAGE/XML data.

Expected data layout under --data-path:
    <subset_a>/
      image_1.jpg
      image_2.jpg
      page/
        image_1.xml
        image_2.xml
        test_0050
    <subset_b>/
      ...

For every subset directory that has `page/<split-file>`, only those basenames listed
in that file are evaluated.

The script:
1) Finds eval pages from every `*/page/<split-file>` under --data-path.
2) Extracts line crops with polygon masking via `gt.Page.lines()` (same as training).
3) Applies training-time `normalize` and `reject` rules.
4) Runs TrOCR inference and writes one CSV row per retained line.
5) Saves every cropped line image.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from tqdm import tqdm

from gt import Page
from params import MODEL_MAX_LENGTH
from train import TrOCRModule, normalize, reject

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".bmp"}


def infer_model_base_model_id_from_checkpoint(checkpoint_path: Path) -> str | None:
    """
    Infer TrOCR base model family from checkpoint tensor shapes.
    Returns a HF model id if inference is possible, else None.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    probe_key = "model.decoder.model.decoder.layers.0.encoder_attn.k_proj.weight"
    probe = state_dict.get(probe_key)
    if probe is None or probe.ndim != 2:
        return None

    in_features = int(probe.shape[1])
    if in_features == 768:
        return "microsoft/trocr-base-handwritten"
    if in_features == 1024:
        return "microsoft/trocr-large-handwritten"
    return None


def normalize_basename(entry: str) -> str:
    """Turn split-file entries into a basename (strip path + suffix)."""
    raw = entry.strip()
    if not raw:
        return ""
    return Path(raw).stem


def find_image_for_basename(folder: Path, basename: str) -> Path | None:
    """Find best matching image file in `folder` for `basename` (case-insensitive)."""
    candidates = []
    base_lower = basename.lower()
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        if p.stem.lower() == base_lower:
            candidates.append(p)

    if not candidates:
        return None

    # Deterministic preference by extension then name.
    preferred_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".bmp"]
    return sorted(
        candidates,
        key=lambda p: (
            preferred_exts.index(p.suffix.lower())
            if p.suffix.lower() in preferred_exts
            else 999,
            p.name,
        ),
    )[0]


def iter_eval_pages(data_path: Path, split_filename: str) -> Iterable[tuple[Path, Page]]:
    """Yield `(subset_dir, Page)` for every eval basename in every subset split file."""
    for root, dirs, _files in os.walk(data_path):
        if "page" not in dirs:
            continue

        subset_dir = Path(root)
        page_dir = subset_dir / "page"
        split_path = page_dir / split_filename
        if not split_path.exists():
            continue

        with split_path.open("r", encoding="utf-8") as f:
            basenames = [normalize_basename(line) for line in f]
            basenames = [b for b in basenames if b]

        if not basenames:
            print(f"[WARN] Empty split file: {split_path}")
            continue

        for basename in basenames:
            xml_path = page_dir / f"{basename}.xml"
            image_path = find_image_for_basename(subset_dir, basename)

            if not xml_path.exists():
                print(f"[WARN] Missing XML for eval basename '{basename}': {xml_path}")
                continue
            if image_path is None:
                print(f"[WARN] Missing image for eval basename '{basename}' in {subset_dir}")
                continue

            yield subset_dir, Page(str(xml_path), str(image_path))


def save_line_image(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to trained Lightning checkpoint (.ckpt)")
    parser.add_argument("--data-path", required=True, help="Root path containing multiple subset/image directories")
    parser.add_argument(
        "--split-file",
        default="test_0050",
        help="Split filename expected in each subset's page/ directory (default: test_0050)",
    )
    parser.add_argument("--output-dir", default="evaluation_local", help="Output directory for crops and csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model-base-model-id",
        default=None,
        help=(
            "HF base model id used during training, e.g. "
            "'microsoft/trocr-base-handwritten' or 'microsoft/trocr-large-handwritten'. "
            "If omitted, the script attempts to infer it from checkpoint tensor shapes."
        ),
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    crop_root = output_dir / "crops"
    csv_path = output_dir / "line_predictions.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    inferred_model_id = infer_model_base_model_id_from_checkpoint(Path(args.checkpoint))
    model_id = args.model_base_model_id or inferred_model_id

    if model_id is not None:
        print(f"Using model base id: {model_id}")
        module = TrOCRModule.load_from_checkpoint(
            args.checkpoint, model_base_model_id=model_id
        )
    else:
        print(
            "[WARN] Could not infer model base id from checkpoint; "
            "falling back to default in params.py."
        )
        module = TrOCRModule.load_from_checkpoint(args.checkpoint)
    module.eval()
    module.model.to(device)

    rows: list[dict[str, str]] = []
    pending_images: list[Image.Image] = []
    pending_meta: list[dict[str, str]] = []

    pages_seen = 0
    subsets_seen = set()

    def flush_batch() -> None:
        if not pending_images:
            return

        pixel_values = module.processor(images=pending_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = module.model.generate(
                pixel_values,
                max_new_tokens=MODEL_MAX_LENGTH,
                num_beams=4,
                interpolate_pos_encoding=True,
            )

        preds = module.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred, meta in zip(preds, pending_meta):
            rows.append(
                {
                    "cropped_image_path": meta["cropped_image_path"],
                    "trocr_transcription": pred,
                    "ground_truth_transcription": meta["ground_truth_transcription"],
                    "line_key": meta["line_key"],
                    "page_xml_path": meta["page_xml_path"],
                    "source_image_path": meta["source_image_path"],
                }
            )

        pending_images.clear()
        pending_meta.clear()

    iterator = iter_eval_pages(data_path, args.split_file)
    for subset_dir, page in tqdm(iterator, desc="Processing eval pages", unit="page"):
        pages_seen += 1
        subsets_seen.add(str(subset_dir))

        rel_subset = subset_dir.resolve().relative_to(data_path)
        page_stem = Path(page.xml_path).stem

        for line_idx, (line_key, image, text) in enumerate(page.lines()):
            text_norm = normalize(text)
            if reject(image, text_norm):
                continue

            crop_path = crop_root / rel_subset / f"{page_stem}_line{line_idx:04d}.jpg"
            save_line_image(image, crop_path)

            pending_images.append(image)
            pending_meta.append(
                {
                    "cropped_image_path": str(crop_path),
                    "ground_truth_transcription": text_norm,
                    "line_key": line_key,
                    "page_xml_path": page.xml_path,
                    "source_image_path": page.image_path,
                }
            )

            if len(pending_images) >= args.batch_size:
                flush_batch()

    flush_batch()

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cropped_image_path",
                "trocr_transcription",
                "ground_truth_transcription",
                "line_key",
                "page_xml_path",
                "source_image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {pages_seen} eval pages from {len(subsets_seen)} subset folders.")
    print(f"Saved {len(rows)} line predictions to {csv_path}")


if __name__ == "__main__":
    main()
