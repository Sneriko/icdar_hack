"""
Train TrOCR.
"""

import argparse
import io
import torch
import pickle
from pathlib import Path
import random
import logging
import re
import unicodedata

from PIL import Image
from jiwer import cer
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import lmdb


from tracking import get_logger
from augment import augment
from data import init_lmdb, train_test_split
from params import *


torch.set_float32_matmul_precision(TORCH_FLOAT32_MATMUL_PRECISION)


validation_logger = logging.getLogger("validation")
validation_logger.addHandler(logging.FileHandler("validation.log"))
validation_logger.setLevel(logging.INFO)



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

        image = Image.open(io.BytesIO(image))
        text = text.decode("utf-8")
        text = normalize(text)

        if reject(image, text):
            return random.choice(self)

        # Prepare image        
        if self.do_augment:
            image = augment(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Prepare labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=MODEL_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
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


def subset(key: bytes):
    """
    Return the subset `key` belongs to
    """
    key = key.decode("utf-8")
    key = key.removeprefix(DATA_PATH)
    return Path(key).parts[1].strip("/")


def reject(image, text):
    """
    Return True if sample should be rejected.
    """
    contains_stopwords = any(stopword in text for stopword in DATA_STOPWORDS)
    too_small = image.height < 10 or image.width < 10
    return contains_stopwords or too_small


def normalize(text: str) -> str:
    """
    Normalize `text`
    """
    fractions = [
        ("½", "1/2"),
        ("↉", "0/3"),
        ("⅓", "1/3"),
        ("⅔", "2/3"),
        ("¼", "1/4"),
        ("¾", "3/4"),
        ("⅕", "1/5"),
        ("⅖", "2/5"),
        ("⅗", "3/5"),
        ("⅘", "4/5"),
        ("⅙", "1/6"),
        ("⅚", "5/6"),
        ("⅐", "1/7"),
        ("⅛", "1/8"),
        ("⅜", "3/8"),
        ("⅝", "5/8"),
        ("⅞", "7/8"),
        ("⅑", "1/9"),
        ("⅒", "1/10"),
    ]

    for a, b in fractions:
        # in mixed whole- and fraction numbers, there might not be a space between
        # the whole part and the fraction part, like this: 2½
        # when replacing the fractions, we need to insert a whitespace to preserve
        # the original number:  2½ -> 2 1/2 (and not 21/2!)
        # if there was a space (2 ½) this causes double whitespaces, but those will be
        # collapsed later on.
        text = text.replace(a, f" {b}")

    # No space before '¬', no repeated '¬'s
    text = re.sub(r"\s*¬", "¬", text)
    text = re.sub(r"¬+", r"¬", text)

    # Remove „
    text = text.replace("„", "")

    # Normalize fancy quotes
    text = text.replace("”", '"').replace("“", '"').replace("´", "'")

    # Make '·' and '‧' into regular '.'
    text = text.replace("·", ".").replace("‧", ".")

    # No tildes
    text = text.replace("~", "")

    # Replace more than three repeated punctuation marks or '…' with '...'
    # 'Den .... 13 .. Januarii' => 'Den ... 13 .. Januarii'
    text = re.sub(r"(\. ?)(\. ?)(\. ?)+", "... ", text)
    text = text.replace("…", "... ")

    # Replace '﹘' ('small em dash') with its 'large' version
    text = text.replace("﹘", " — ")
    text = text.replace("_,", "")

    # Replace all types of dashes with a em dash, IF they're surrounded by whitespace or start/end of line
    text = re.sub(r"(^| )(-|_|—|‒|―|–)( |$)", " — ", text)

    # Replace repeated dashes, dots or similar with a single em dash (TOGMF-style)
    # 'Carl _ _ _ Wikman. 16.' => 'Carl — Wikman. 16.'
    text = re.sub(r"((-|_|—|‒|―|–) ?)((-|_|—|‒|―|–) ?)+", " — ", text)

    # Replace all (possibly repeated) whitespace-like characters with a singe ' '
    text = re.sub(r"\s+", " ", text)

    # No leading or training whitespace or em dashes
    text = text.strip("— ")
    text = unicodedata.normalize("NFKC", text)    
    return text


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

            # print some samples to look at :)
            if random.random() < 0.1:
                validation_logger.info(f"{gt} | {pred} | {cer_}")

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
        monitor=TRAIN_EARLY_STOPPING_MONITOR,
        save_top_k=3,
        filename="{epoch}-{step}-{validation_loss:.4f}-{cer:.4f}",
    )

    early_stopping = EarlyStopping(
        monitor=TRAIN_EARLY_STOPPING_MONITOR,
        patience=TRAIN_EARLY_STOPPING_PATIENCE,
    )

    # Define trainer
    trainer = lightning.Trainer(
        max_epochs=TRAIN_MAX_EPOCHS,
        val_check_interval=0.5,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        num_sanity_val_steps=10,
        callbacks=[checkpoint_callback, early_stopping],
    )

    model = TrOCRModule(model, processor)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
