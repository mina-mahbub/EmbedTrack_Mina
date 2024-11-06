"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from embedtrack.train.run_training_pipeline import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    run_pipeline,
)

import os
from pathlib import Path

# data configs

PROJECT_PATH = "/home/MinaHossain/EmbedTrack"  # /home/MinaHossain/EmbedTrack

RAW_DATA_PATH = os.path.join(PROJECT_PATH, "ctc_raw_data/train")
DATA_PATH_DEST = os.path.join(PROJECT_PATH, "data")
MODEL_PATH = os.path.join(PROJECT_PATH, "models")

USE_SILVER_TRUTH = True
TRAIN_VAL_SEQUNCES = ["01"]
TRAIN_VAL_SPLIT = 0.10


DATA_SETS = [   "Cell-Data-P2" ]


MODEL_NAME = "test4" # Change this everytime you make a brand new model

N_EPOCHS = 15
# Adam optimizer; normalize images; OneCycle LR sheduler; N epochs



for data_set in DATA_SETS:
    if data_set == "Fluo-N2DH-SIM+":
        use_silver_truth = False
    else:
        use_silver_truth = USE_SILVER_TRUTH

    data_config = DataConfig(
        RAW_DATA_PATH,
        data_set,
        DATA_PATH_DEST,
        use_silver_truth=use_silver_truth,
        train_val_sequences=TRAIN_VAL_SEQUNCES,
        train_val_split=TRAIN_VAL_SPLIT,
    )

    # train configs
    MODEL_SAVE_DIR = os.path.join(
        MODEL_PATH,
        data_set,
        MODEL_NAME,
    )
    if data_set == "Cell-Data-P2":
        CROP_SIZE = 512
        TRAIN_BATCH_SIZE = 8
        VAL_BATCH_SIZE = 8
        DISPLAY_IT = 1000

    else:
        CROP_SIZE = 512
        TRAIN_BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        DISPLAY_IT = 200

    CENTER = "centroid"  # 'centroid', 'approximate-medoid', 'medoid'
    ### we tried  "medoid" on test2 and test3
    RESUME_TRAINING = True  # we changed it to TRUE from FALSE to resume the training from previous training. ############ Mina
    TRAIN_SIZE = None  # train on full train data set; otherwise still training on full data set but only use a fraction of the data per epoch
    VAL_SIZE = None  # train on full train data set; otherwise still training on full data set but only use a fraction of the data per epoch
    VIRTUAL_TRAIN_BATCH_MULTIPLIER = 1
    VIRTUAL_VAL_BATCH_MULTIPLIER = 1
    DISPLAY = True

    train_config = TrainConfig(
        MODEL_SAVE_DIR,
        crop_size=CROP_SIZE,
        center=CENTER,
        resume_training=RESUME_TRAINING,
        train_size=TRAIN_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        virtual_train_batch_multiplier=VIRTUAL_TRAIN_BATCH_MULTIPLIER,
        val_size=VAL_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        virtual_val_batch_multiplier=VIRTUAL_VAL_BATCH_MULTIPLIER,
        n_epochs=N_EPOCHS,
        display=DISPLAY,
        display_it=DISPLAY_IT,
        learning_rate=5e-5
    )

    # model config
    INPUT_CHANNELS = 1
    N_SEG_CLASSES = [4, 1]
    N_TRACK_CLASSES = 2

    model_config = ModelConfig(INPUT_CHANNELS, N_SEG_CLASSES, N_TRACK_CLASSES)

    run_pipeline(data_config, train_config, model_config)
    plt.close("all")

