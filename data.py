"""
Scripts for creating and handling the LMDB.
"""

import lmdb
from PIL import Image
import pickle

import os
import io
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import random
from collections import defaultdict
from tqdm import tqdm
from gt import pages_from_path
import threading

from params import LMDB_DATA_DIRECTORY, LMDB_KEYS, LMDB_MAP_SIZE, TRAIN_SPLIT_SIZE


random.seed(0)


def encode_sample(key: str, image: Image, text: str) -> bytes | None:
    """
    Encode an (image, text) tuple into bytes for storing in LMDB.

    Arguments:
        image: sample image
        text: sample transcription

    Returns the sample as a byte string, or `None` if encoding failed.
    """
    image_bytes = io.BytesIO()
    try:
        image.save(image_bytes, format="JPEG")
    except ValueError:
        return None
    image_bytes = image_bytes.getvalue()
    text_bytes = text.encode("utf-8")
    return key.encode("utf-8"), pickle.dumps((image_bytes, text_bytes))


def get_pages():
    """
    Get all page keys stored in the LMDB.
    """
    env = lmdb.open(LMDB_DATA_DIRECTORY, map_size=LMDB_MAP_SIZE)
    with env.begin() as txn:
        keys = txn.get(LMDB_KEYS)
        keys = pickle.loads(keys) if keys else []
    env.close()
    return keys


class LinesDataset:
    def __init__(self, pages):
        self.env = lmdb.open(LMDB_DATA_DIRECTORY, map_size=LMDB_MAP_SIZE)

        self.keys = []
        with self.env.begin() as txn:
            for page in pages:
                keys = txn.get(page)
                keys = pickle.loads(keys)
                self.keys.extend(keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            data = txn.get(key)
            image, text = pickle.loads(data)

        image = Image.open(io.BytesIO(image))
        text = text.decode("utf-8")
        return image, text

    def __len__(self):
        return len(self.keys)


def _lmdb_writer(queue: Queue, pages_pbar):
    """
    Write samples from `queue` to the LMDB.
    """
    env = lmdb.open(LMDB_DATA_DIRECTORY, map_size=LMDB_MAP_SIZE)
    txn = env.begin(write=True)

    keys = txn.get(LMDB_KEYS)
    keys = pickle.loads(keys) if keys else []
    i = 0
    lines_pbar = tqdm(desc="Masked lines", position=1, unit=" lines")
    while 1:
        item = queue.get()
        if item is None:
            break

        line_keys = []
        page, lines = item
        for line in lines:
            item = encode_sample(*line)
            if item is None:
                continue

            key, value = item
            line_keys.append(key)
            txn.put(key, value)
            i += 1
            lines_pbar.update(1)

        page_key = page.key.encode("utf-8")
        keys.append(page_key)

        txn.put(page_key, pickle.dumps(line_keys))
        txn.put(LMDB_KEYS, pickle.dumps(keys))

        if i > 10_000:
            txn.commit()
            txn = env.begin(write=True)
            i = 0

        # One page done!
        pages_pbar.update(1)
        queue.task_done()

    txn.commit()
    env.close()
    queue.task_done()


def init_lmdb(source) -> list[bytes]:
    """
    Create a dataset from the given source.

    Returns:
        The list of keys for the created dataset.
    """

    keys = get_pages()
    to_process = []
    num_cached = 0
    for page in source:
        if page.key.encode("utf-8") in keys:
            num_cached += 1
        else:
            to_process.append(page)

    print(
        f"Found {len(source)} GT pages, ",
        f"{num_cached} already in LMDB, ",
        f"{len(to_process)} remains"
    )
    if not to_process:
        return

    pages_pbar = tqdm(
        desc="Processed pages", position=0, unit=" pages", total=len(to_process)
    )
    queue = Queue(100)  # max size, number of pages
    writer = threading.Thread(
        target=lambda: _lmdb_writer(queue, pages_pbar), daemon=True
    )
    writer.start()

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda page: queue.put((page, page.lines())), to_process)

    queue.put(None)  # Tell _lmdb_writer we're done
    queue.join()


def train_test_split():
    """
    Split the LMDB's keys into train and test.

    We use the TRAIN_SPLIT_SIZE but make sure that each subset gets at
    least one page into the evaluation split.
    """

    keys = get_pages()
    subsets = defaultdict(list)
    for key in keys:
        subsets[os.path.dirname(key)].append(key)

    train = []
    test = []
    for subset in subsets.values():
        n = int((1 - TRAIN_SPLIT_SIZE) * len(subset))
        n = max(1, n)
        random.shuffle(subset)
        test.extend(subset[:n])
        train.extend(subset[n:])
    return train, test
