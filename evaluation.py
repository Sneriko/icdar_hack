"""
Run model on evaluation set.
"""

import argparse
from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
import torch
from tqdm import tqdm
from train import TrOCRModule
import jiwer
import pandas as pd


TEST_SUITE = "Riksarkivet/eval_htr_out_of_domain_lines"


def collate_fn(batch, processor):
    pixel_values = processor(
        [sample["image"] for sample in batch], return_tensors="pt"
    ).pixel_values
    return pixel_values.to("cuda"), [sample["transcription"] for sample in batch]


def load_test_suite(processor):
    cache = snapshot_download(
        repo_id=TEST_SUITE, repo_type="dataset", allow_patterns="*.parquet"
    )
    dataloaders = {}
    for path in os.scandir(cache):
        if path.name.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=[path.path])
            dataloader = torch.utils.data.DataLoader(
                dataset["train"],
                batch_size=16,
                collate_fn=lambda batch: collate_fn(batch, processor),
            )
            dataloaders[path.name] = dataloader
    return dataloaders


def run_test_suite(checkpoint) -> str:
    """
    Run model on test suite

    Arguments:
        checkpoint: path to checkpoint to run

    Returns:
        path of result csv
    """

    m = TrOCRModule.load_from_checkpoint(checkpoint)
    dataloaders = load_test_suite(m.processor)

    lines = []
    for name, dataloader in dataloaders.items():
        for batch in tqdm(iter(dataloader), desc=name):
            batch, gts = batch
            outputs = m.model.generate(
                batch,
                max_new_tokens=23,
                num_beams=4,
                interpolate_pos_encoding=True,
                eos_token_id=1,
            )

            preds = m.processor.decode(outputs, skip_special_tokens=True)
            for gt, pred in zip(gts, preds):
                lines.append({
                    "dataset": name,
                    "gt": gt,
                    "htr": pred
                })

    df = pd.DataFrame(lines)
    df["cer"] = df.apply(lambda row: jiwer.cer(row["gt"], row["htr"]), axis=1)
    df["wer"] = df.apply(lambda row: jiwer.wer(row["gt"], row["htr"]), axis=1)

    path = checkpoint.replace(".ckpt", "")
    os.makedirs(path, exist_ok=True)
    output = os.path.join(path, "evaluation.csv")    
    df.to_csv(output)
    print("Wrote evaluation results to", output)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="checkpoint path")
    args = parser.parse_args()
    run_test_suite(args.checkpoint)
