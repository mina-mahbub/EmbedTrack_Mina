# """
# Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
# Licensed under MIT License
# """

# import os
# from datetime import datetime
# from pathlib import Path
# from time import time

# from embedtrack.infer.infer_ctc_data import inference

# # FILE_PATH = Path(__file__)
# # PROJECT_PATH = os.path.join(*FILE_PATH.parts[:-3])
# PROJECT_PATH = "/home/MinaHossain/EmbedTrack" #"/home/MinaHossain/EmbedTrack"

# RAW_DATA_PATHS = [os.path.join(PROJECT_PATH, "ctc_raw_data/challenge"),
#                   os.path.join(PROJECT_PATH, "ctc_raw_data/train")]
# MODEL_PATH = os.path.join(PROJECT_PATH, "models")
# RES_PATH = os.path.join(PROJECT_PATH, "results")

# DATA_SETS = [
#     # "Fluo-N2DH-SIM+",
#     # "Fluo-C2DL-MSC",
#     # "Fluo-N2DH-GOWT1",
#     # "PhC-C2DL-PSC",
#     # "BF-C2DL-HSC",
#     # "Fluo-N2DL-HeLa",
#     # "BF-C2DL-MuSC",
#     # "DIC-C2DH-HeLa",
#      "Cell-Data-P2",
#     # "PhC-C2DH-U373"
# ]


# CALC_CTC_METRICS = False


# # Adam optimizer; normalize images; OneCycle LR sheduler; N epochs
# MODEL_NAME = "test2" # Change this everytime you make a brand new model  #"adam_norm_onecycle_15"
# BATCH_SIZE = 16 #32
# runtimes = {}
# for raw_data_path in RAW_DATA_PATHS:
#     for data_set in DATA_SETS:
#         for data_id in ["01", "02"]:
#             img_path = os.path.join(raw_data_path, data_set, data_id)

#             model_dir = os.path.join(MODEL_PATH, data_set, MODEL_NAME)
#             # print(model_dir)
#             if not os.path.exists(model_dir):
#                 print(f"no trained model for data set {data_set}")
#                 continue

#             # time stamps
#             timestamps_trained_models = [
#                 datetime.strptime(time_stamp, "%Y-%m-%d---%H-%M-%S")
#                 for time_stamp in os.listdir(model_dir)
#             ]
            
#             timestamps_trained_models.sort()
#             last_model = timestamps_trained_models[-1].strftime("%Y-%m-%d---%H-%M-%S")
#             model_path = os.path.join(model_dir, last_model, "best_iou_model.pth")
#             config_file = os.path.join(model_dir, last_model, "config.json")
#             t_start = time()
#             inference(img_path, model_path, config_file, batch_size=BATCH_SIZE)
#             t_end = time()

#             run_time = t_end - t_start
#             print(f"Inference Time {img_path}: {run_time}s")




"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""

import os
from datetime import datetime
from pathlib import Path
from time import time

from embedtrack.infer.infer_ctc_data import inference

PROJECT_PATH = "/home/MinaHossain/EmbedTrack"

# RAW_DATA_PATHS = [os.path.join(PROJECT_PATH, "ctc_raw_data/challenge"),
#                   os.path.join(PROJECT_PATH, "ctc_raw_data/train")]
RAW_DATA_PATHS = [os.path.join(PROJECT_PATH, "ctc_raw_data/challenge")]
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
# RES_PATH = os.path.join(PROJECT_PATH, "results")

DATA_SETS = [ "Cell-Data-P2"]

CALC_CTC_METRICS = False

MODEL_NAME = "test8"    ### ### Change the name every time we run inferernce for a model that was trained #### 
BATCH_SIZE = 8
runtimes = {}

for raw_data_path in RAW_DATA_PATHS:
    for data_set in DATA_SETS:
        for data_id in ["02"]:          #["01", "02"]:
            img_path = os.path.join(raw_data_path, data_set, data_id)
            model_dir = os.path.join(MODEL_PATH, data_set, MODEL_NAME)

            if not os.path.exists(model_dir):
                print(f"No trained model for data set {data_set}")
                continue

            # Collect valid timestamps, skipping non-timestamp files
            timestamps_trained_models = []
            for time_stamp in os.listdir(model_dir):
                if os.path.isdir(os.path.join(model_dir, time_stamp)):
                    try:
                        timestamps_trained_models.append(datetime.strptime(time_stamp, "%Y-%m-%d---%H-%M-%S"))
                    except ValueError:
                        print(f"Skipping non-timestamp directory: {time_stamp}")
                        continue

            if timestamps_trained_models:
                timestamps_trained_models.sort()
                last_model = timestamps_trained_models[-1].strftime("%Y-%m-%d---%H-%M-%S")
                model_path = os.path.join(model_dir, last_model, "best_iou_model.pth")
                config_file = os.path.join(model_dir, last_model, "config.json")
            else:
                # Check if best_iou_model.pth and config.json are directly in model_dir
                print("Its running here anyways")
                model_path = os.path.join(model_dir, "best_iou_model.pth")
                config_file = os.path.join(model_dir, "config.json")

                if not (os.path.exists(model_path) and os.path.exists(config_file)):
                    print(f"No valid model or configuration found in {model_dir}")
                    continue

            # Start inference and time it
            t_start = time()
            inference(img_path, model_path, config_file, batch_size=BATCH_SIZE)
            t_end = time()

            run_time = t_end - t_start
            print(f"Inference Time {img_path}: {run_time}s")


