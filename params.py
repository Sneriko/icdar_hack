"""
Hyperparameters for training the Swedish lion.
"""

DATA_PATH = "/mnt/work/GT-Lejonet_HTR_202602"
DATA_STOPWORDS = [  # reject samples containing these 'words'
    "??",   # transcriber uncertainty
    "(?)",  # as above, example from JLF: 'dukar, 1 armkläde, 2 harm lijner (?), 1 par toflor,'
    "[",    # used as commentary or annotations, example from JLF: 'Nilsson ibidem. Desse bekänn[e]r och tilstå hafwa'
    "]",    # as above

]

LMDB_DATA_DIRECTORY = ".data"
LMDB_KEYS = b"__keys__"
LMDB_MAP_SIZE = 200 * 1024**3

MODEL_BASE_MODEL_ID = "microsoft/trocr-base-handwritten"
MODEL_IMAGE_SIZE = {"height": 192, "width": 1024}
MODEL_MAX_LENGTH = 128

TRAIN_LEARNING_RATE = 1e-5
TRAIN_BATCH_SIZE = 128
TRAIN_MAX_EPOCHS = 50
TRAIN_EARLY_STOPPING_PATIENCE = 5
TRAIN_EARLY_STOPPING_MONITOR = "validation_loss"

TEST_SPLIT = "test_0050"
AUGMENTATION_PROBABILITY = 0.5

TORCH_FLOAT32_MATMUL_PRECISION = "high"
