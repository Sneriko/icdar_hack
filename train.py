"""
Train TrOCR.
"""

import atexit
import argparse
import lightning as L

from jiwer import cer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tracking import get_logger
from data import init_lmdb, train_test_split
from params import *


torch.set_float32_matmul_precision("high")


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.keys)


class TrOCRModule(L.LightningModule):
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
        pixel_values, labels = batch
        outputs = self.model(
            pixel_values=pixel_values, labels=labels, interpolate_pos_encoding=True
        )
        loss = self._compute_loss(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, subset_names = batch
        outputs = self.model(pixel_values, labels=labels, interpolate_pos_encoding=True)
        loss = self._compute_loss(outputs, labels)
        self.log("validation_loss", loss)

        # Compute CER
        labels[labels == -100] = self.model.config.pad_token_id
        gt = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        outputs = self.model.generate(pixel_values=pixel_values, interpolate_pos_encoding=True)
        preds = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for gt, pred, subset in zip(gt, preds, subset_names):
            cer_ = cer(gt, pred)
            self.log(f"cer_{subset}", cer_)
            self.log("cer", cer_)

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

    # Init datasets
    init_lmdb()
    train_keys, test_keys = train_test_split()
    train_dataset = LMDBDataset(train_keys)
    test_dataset = LMDBDataset(test_keys)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)} lines")
    print(f" Test dataset size: {len(train_dataset)} lines")

    # Init processor
    processor = TrOCRProcessor.from_pretrained(MODEL_BASE_MODEL_ID, use_fast=True)
    processor.feature_extractor.size = MODEL_IMAGE_SIZE

    # Init model
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_BASE_MODEL_ID)
    model.config.max_length = MODEL_MAX_LENGTH
    model.config.num_beams = 1
    model.config.vocab_size = model.config.decoder.vocab_size
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
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=10,
        val_check_interval=1.0,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        num_sanity_val_steps=100,
        callbacks=[checkpoint_callback],
    )

    model = TrOCRModule(model, processor)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        test_dataloaders=test_dataloader,
    )
