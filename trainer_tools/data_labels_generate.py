
import glob
import torch
import datetime  # 用于获取当前时间，以便在日志中加上时间戳
import os
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
import re
from tqdm import tqdm


class DataLabelsGenerate:
    def __init__(self):
        print("DataLabelsGenerate __init__")
        # self.save_files_path = "F:\\EEGDataset\\MASS\\data_anno_savefiles"
        # self.save_files_path = "G:\\Research\\EEG_Project\\data\\MASS\\SS3\\data_anno_savefiles"

        # self.save_files = glob.glob(os.path.join(self.save_files_path, "*.h5"))

    def generate_data_labels(self, file_names, batch_size):
        while True:
            for file_name in file_names:
                with h5py.File(file_name, 'r') as f:
                    eeg_data_group = f['data']
                    channel_names = list(eeg_data_group.keys())
                    data_list = []

                    for channel in channel_names:
                        data = eeg_data_group[channel][:]
                        data_tensor = torch.tensor(data, dtype=torch.float32)
                        data_list.append(data_tensor.unsqueeze(1))

                    combined_data = torch.cat(data_list, dim=1)  # (样本数, 通道数, 高, 宽)
                    labels = torch.tensor(f['labels']['stage_labels'][:], dtype=torch.long)

                # 注意：文件已关闭，现在 safe to yield
                indices = np.arange(len(combined_data))
                for i in range(0, len(combined_data), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_data = combined_data[batch_indices]
                    batch_labels = labels[batch_indices]

                    yield {
                        'batch_data': batch_data,
                        'labels': batch_labels,
                        'file_name': file_name
                    }

