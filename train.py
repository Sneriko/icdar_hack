"""
Train TrOCR.
"""

import argparse
import io
import torch
import pickle
from pathlib import Path

from PIL import Image
from jiwer import cer
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import lmdb

from tracking import get_logger
from augment import augment
from data import init_lmdb, train_test_split
from params import *


torch.set_float32_matmul_precision(TORCH_FLOAT32_MATMUL_PRECISION)


class TrOCRDataset(torch.utils.data.Dataset):
    def __init__(self, pages: list[str], do_augment: bool, processor: TrOCRProcessor):
        """
        Arguments:
            pages: The list of *page* keys that belong to this dataset.
        """

        self.env = lmdb.open(LMDB_DATA_DIRECTORY, readonly=True, map_size=LMDB_MAP_SIZE)
        self.keys = []
        with self.env.begin() as txn:
            for page in pages:
                keys = txn.get(page)
                keys = pickle.loads(keys)
                self.keys.extend(keys)
        self.env.close()
        self.env = None
        self.do_augment = do_augment
        self.processor = processor

    def __getitem__(self, idx):
        # Get sample from LMDB
        key = self.keys[idx]
        with self.env.begin() as txn:
            data = txn.get(key)
            image, text = pickle.loads(data)

        # Prepare image
        image = Image.open(io.BytesIO(image))
        if self.do_augment:
            image = augment(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Prepare labels
        text = text.decode("utf-8")
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=MODEL_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels, key

    def __len__(self):
        return len(self.keys)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.env = lmdb.open(
        LMDB_DATA_DIRECTORY,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        map_size=LMDB_MAP_SIZE,
    )
    # atexit.register(dataset.cleanup_environment)


def subset(key: bytes):
    """
    Return the subset `key` belongs to
    """
    key = key.decode("utf-8")
    key = key.removeprefix(DATA_PATH)
    return Path(key).parts[1].strip("/")


class TrOCRModule(lightning.LightningModule):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def _compute_loss(self, outputs, labels):
        logits = outputs.logits
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        return self.loss_fn(logits, labels)

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch
        outputs = self.model(
            pixel_values=pixel_values, labels=labels, interpolate_pos_encoding=True
        )
        loss = self._compute_loss(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, keys = batch
        outputs = self.model(pixel_values, labels=labels, interpolate_pos_encoding=True)
        loss = self._compute_loss(outputs, labels)
        self.log("validation_loss", loss)

        # Compute CER
        labels[labels == -100] = self.model.config.pad_token_id
        gt = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        outputs = self.model.generate(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
            max_new_tokens=MODEL_MAX_LENGTH,
        )
        preds = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for gt, pred, key in zip(gt, preds, keys):
            cer_ = cer(gt, pred)
            self.log("cer", cer_)
            self.log(f"cer_{subset(key)}", cer_)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=TRAIN_LEARNING_RATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train")
    parser.add_argument("experiment_name")
    parser.add_argument("--no-track", action="store_true")
    args = parser.parse_args()

    # Set up logging
    strict = not args.no_track
    logger = get_logger(args.experiment_name, strict)

    # Init processor
    processor = TrOCRProcessor.from_pretrained(MODEL_BASE_MODEL_ID, use_fast=True)
    processor.image_processor.size = MODEL_IMAGE_SIZE

    # Init datasets
    init_lmdb()
    train_keys, test_keys = train_test_split()
    train_dataset = TrOCRDataset(train_keys, do_augment=True, processor=processor)
    test_dataset = TrOCRDataset(test_keys, do_augment=False, processor=processor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        TRAIN_BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    print(f"Train dataset size: {len(train_dataset)} lines")
    print(f" Test dataset size: {len(test_dataset)} lines")

    # Init model
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_BASE_MODEL_ID)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.train()

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.experiment_name}",
        every_n_epochs=1,
        monitor="validation_loss",
        save_top_k=3,
        filename="{epoch}-{step}-{validation_loss:.4f}-{cer:.4f}",
    )

    # Define trainer
    trainer = lightning.Trainer(
        max_epochs=10,
        val_check_interval=0.001,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        num_sanity_val_steps=100,
        callbacks=[checkpoint_callback],
        limit_val_batches=200,
    )

    model = TrOCRModule(model, processor)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
