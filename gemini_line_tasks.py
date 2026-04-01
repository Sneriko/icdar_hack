#!/usr/bin/env python3
"""Run multiple Gemini transcription/correction tasks over line_predictions.csv.

This script reads an input CSV containing:
- path to a text-line image
- a TrOCR transcription
- a ground-truth transcription

It runs five tasks with Gemini and writes:
1) an augmented CSV with new model outputs + per-line CER metrics
2) a summary CSV with corpus-level CER per task

Required environment variable:
  GEMINI_API_KEY

Example:
  python gemini_line_tasks.py \
      --input-csv datafolder/line_predictions.csv \
      --output-csv datafolder/line_predictions_gemini.csv \
      --summary-csv datafolder/line_predictions_gemini_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from jiwer import cer

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency 'google-genai'. Install it with: uv add google-genai"
    ) from exc


TASKS = {
    "task1_image_only": {
        "instruction": (
            "You are an OCR assistant. Read the provided text-line image and output only "
            "the exact transcription as plain text. Do not add explanations, quotes, or "
            "extra formatting."
        ),
        "use_image": True,
        "use_trocr": False,
        "use_guidelines": False,
    },
    "task2_image_plus_trocr": {
        "instruction": (
            "You are an OCR correction assistant. You are given a text-line image and a "
            "TrOCR transcription. Correct the TrOCR transcription using the image. Output "
            "only the corrected transcription as plain text."
        ),
        "use_image": True,
        "use_trocr": True,
        "use_guidelines": False,
    },
    "task3_trocr_only": {
        "instruction": (
            "You are a text normalization assistant. You are given a TrOCR transcription "
            "without the source image. Correct likely OCR mistakes using only the text. "
            "Output only the corrected transcription as plain text."
        ),
        "use_image": False,
        "use_trocr": True,
        "use_guidelines": False,
    },
    "task4_trocr_plus_guidelines": {
        "instruction": (
            "You are a transcription correction assistant. You are given a TrOCR "
            "transcription and transcription guidelines. Without image access, correct "
            "the transcription strictly according to the guidelines. Output only the "
            "corrected transcription as plain text."
        ),
        "use_image": False,
        "use_trocr": True,
        "use_guidelines": True,
    },
    "task5_image_plus_guidelines": {
        "instruction": (
            "You are a transcription assistant. You are given a text-line image, a TrOCR "
            "transcription, and transcription guidelines. Use the image and guidelines to "
            "produce the best corrected transcription. Output only the corrected "
            "transcription as plain text."
        ),
        "use_image": True,
        "use_trocr": True,
        "use_guidelines": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--guidelines-pdf", type=Path, default=Path("datafolder/tridis.pdf"))
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model name (e.g. gemini-2.5-pro, gemini-3.1-pro if available).",
    )
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--trocr-col", default="trocr_transcription")
    parser.add_argument("--gt-col", default="ground_truth")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for relative image paths (defaults to input CSV directory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for a quick test run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-row progress.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def build_parts(
    row: dict[str, str],
    image_path: Path,
    guideline_pdf: Path,
    task_cfg: dict[str, Any],
    trocr_col: str,
) -> list[types.Part | str]:
    parts: list[types.Part | str] = [task_cfg["instruction"]]

    if task_cfg["use_trocr"]:
        parts.append(f"TrOCR transcription: {row.get(trocr_col, '').strip()}")

    if task_cfg["use_guidelines"]:
        if not guideline_pdf.exists():
            raise FileNotFoundError(f"Guidelines PDF not found: {guideline_pdf}")
        parts.append("Transcription guidelines PDF is attached.")
        parts.append(
            types.Part.from_bytes(
                data=guideline_pdf.read_bytes(),
                mime_type="application/pdf",
            )
        )

    if task_cfg["use_image"]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        ext = image_path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        parts.append("Text-line image is attached.")
        parts.append(
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=mime,
            )
        )

    return parts


def run_task(
    client: genai.Client,
    model: str,
    row: dict[str, str],
    task_cfg: dict[str, Any],
    image_path: Path,
    guideline_pdf: Path,
    trocr_col: str,
) -> str:
    parts = build_parts(row, image_path, guideline_pdf, task_cfg, trocr_col)
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            max_output_tokens=512,
        ),
    )

    text = (response.text or "").strip()
    if not text:
        try:
            text = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            text = ""
    return text


def corpus_cer(truths: list[str], hyps: list[str]) -> float:
    if not truths:
        return 0.0
    return float(cer("\n".join(truths), "\n".join(hyps)))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise SystemExit("Input CSV has no rows.")

    base_dir = args.base_dir or args.input_csv.parent
    client = make_client()

    missing = [
        c
        for c in [args.image_col, args.trocr_col, args.gt_col]
        if c not in rows[0]
    ]
    if missing:
        raise SystemExit(
            f"Missing expected columns in CSV: {missing}. "
            f"Found columns: {list(rows[0].keys())}"
        )

    truths_by_task: dict[str, list[str]] = {task: [] for task in TASKS}
    hyps_by_task: dict[str, list[str]] = {task: [] for task in TASKS}

    for i, row in enumerate(rows, start=1):
        image_path = resolve_path(row[args.image_col], base_dir)
        gt = row.get(args.gt_col, "")

        for task_name, task_cfg in TASKS.items():
            pred_col = f"{task_name}_prediction"
            cer_col = f"{task_name}_cer"

            pred = run_task(
                client=client,
                model=args.model,
                row=row,
                task_cfg=task_cfg,
                image_path=image_path,
                guideline_pdf=args.guidelines_pdf,
                trocr_col=args.trocr_col,
            )
            row[pred_col] = pred
            row[cer_col] = f"{cer(gt, pred):.6f}"

            truths_by_task[task_name].append(gt)
            hyps_by_task[task_name].append(pred)

        if args.verbose:
            print(f"Processed {i}/{len(rows)}")

    fieldnames = list(rows[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    summary_json: dict[str, float] = {}
    for task_name in TASKS:
        total = corpus_cer(truths_by_task[task_name], hyps_by_task[task_name])
        summary_rows.append({"task": task_name, "corpus_cer": f"{total:.6f}"})
        summary_json[task_name] = total

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "corpus_cer"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
