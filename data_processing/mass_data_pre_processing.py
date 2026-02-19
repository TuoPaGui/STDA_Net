import os
import glob
import h5py
import numpy as np
import datetime
import warnings
import mne
from tqdm import tqdm
from scipy.interpolate import interp1d
from tools import data_labels_generate

subset_name = "SS3"

def log_message(message):
    log_dir = f"F:/EEGDataset/MASS/{subset_name}/data_anno_savefiles"
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    log_file = os.path.join(log_dir, '数据预处理日志.txt')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{now} - {message}\n")

class DataPreprocessing:
    def __init__(self):
        print("DataPreprocessing __init__")
        base_path = f"F:/EEGDataset/MASS/{subset_name}"
        self.data_files_path = os.path.join(base_path, "data")
        self.anno_files_path = os.path.join(base_path, "annoations")
        self.save_files_path = os.path.join(base_path, "data_anno_savefiles")

        # 确保所有关键路径存在
        os.makedirs(self.data_files_path, exist_ok=True)
        os.makedirs(self.anno_files_path, exist_ok=True)
        os.makedirs(self.save_files_path, exist_ok=True)

        self.data_files_names = []
        self.anno_files_names = []
        self.save_files_names = []

        self.raw_data = None
        self.raw_annotations = None

        self.second_time = 30

        self.sampling_frequency = 100
        self.labels = []
        self.data = {}
        self.channel_names = []
        # 多通道
        self.desired_channel_names = ['EEG C4','EEG O2', 'EEG F4']

        self.data_labels_generator = data_labels_generate.DataLabelsGenerate()

    def save_files(self, index):
        with h5py.File(self.save_files_names[index], 'w') as f:
            eeg_group = f.create_group('data')
            for ch in self.channel_names:
                eeg_group.create_dataset(f'{ch}_data', data=self.data[ch])    # 转换为uV
            labels_group = f.create_group('labels')
            labels_group.create_dataset('stage_labels', data=self.labels)

    def get_files_names(self):
        self.data_files_names = glob.glob(os.path.join(self.data_files_path, '*PSG.edf'))
        base_names = [os.path.basename(f)[:-8] for f in self.data_files_names]

        self.anno_files_names = []
        self.save_files_names = []

        for bn in base_names:
            if subset_name.upper() == "SS2":
                anno_pattern = f"{bn} Base.edf"
            else:
                anno_pattern = f"{bn} Annotations.edf"
            anno_matches = glob.glob(os.path.join(self.anno_files_path, anno_pattern))
            if not anno_matches:
                log_message(f"[⚠️] 找不到注释文件: {anno_pattern}")
                continue

            self.anno_files_names.append(anno_matches[0])
            self.save_files_names.append(os.path.join(self.save_files_path, f"{bn}.h5"))

        self.data_files_names = self.data_files_names[:len(self.anno_files_names)]
        self.files_counts = len(self.data_files_names)

    def get_data_labels(self):
        """直接获取原始数据和标签，不移除伪影"""
        self.data.clear()
        self.labels.clear()

        # 有效睡眠阶段映射
        stage_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4,
        }

        sfreq = self.sampling_frequency
        total_len = int(self.raw_data.times[-1] * sfreq)

        # Step 1️⃣ 初始化标签序列
        label_seq = np.full(total_len, -1, dtype=int)

        # Step 2️⃣ 根据注释填充睡眠阶段标签
        for onset, duration, desc in zip(self.raw_annotations.onset,
                                         self.raw_annotations.duration,
                                         self.raw_annotations.description):
            start = int(onset * sfreq)
            end = int((onset + duration) * sfreq)
            if desc in stage_map:
                label_seq[start:end] = stage_map[desc]

        # Step 3️⃣ 提取所有有标签的数据点（不移除伪影）
        valid_idx = np.where(label_seq != -1)[0]
        if len(valid_idx) == 0:
            log_message("❌ 没有有效标签数据，跳过该文件")
            return

        # Step 4️⃣ 获取有效数据（不移除伪影）
        label_seq = label_seq[valid_idx]
        valid_data = {}
        for ch in self.channel_names:
            full_data = self.raw_data.get_data(picks=ch)[0, :]
            valid_data[ch] = full_data[valid_idx]

        # Step 5️⃣ 拼接成连续数据，按指定时间分段
        seg_len = int(self.second_time * sfreq)
        total_segments = len(label_seq) // seg_len

        if total_segments == 0:
            print("⚠️ 有效数据不足一个完整段，跳过")
            return

        # 截取完整的段
        label_seq = label_seq[:total_segments * seg_len].reshape(-1, seg_len)
        for ch in self.channel_names:
            valid_data[ch] = valid_data[ch][:total_segments * seg_len].reshape(-1, seg_len)

        # Step 6️⃣ 多数投票生成每段标签
        self.labels = []
        for segment_labels in label_seq:
            # 直接使用多数投票，不考虑伪影
            label = np.bincount(segment_labels).argmax()
            self.labels.append(label)

        # Step 7️⃣ 存储数据
        for ch in self.channel_names:
            self.data[ch] = valid_data[ch]

        assert len(self.labels) == self.data[self.channel_names[0]].shape[0], "标签与数据段不一致！"

        log_message(f"✅ 拼接后保留段数: {len(self.labels)}")
        print(f"✅ 拼接后最终有效段: {len(self.labels)}，每段 {seg_len} 个采样点")

    def data_preprocessing(self):
        self.get_files_names()
        for i, data_file in enumerate(self.data_files_names):
            save_path = self.save_files_names[i]
            if os.path.exists(save_path):
                log_message(f"{save_path} 已存在，跳过")
                print(f"{save_path} 已存在，跳过")
                continue

            fif_name = data_file.replace(".edf", "_raw.fif")
            if not os.path.exists(fif_name):
                self.raw_data = mne.io.read_raw_edf(data_file, preload=True, verbose=True)
                self.raw_data.save(fif_name, overwrite=True)
            else:
                self.raw_data = mne.io.read_raw_fif(fif_name, preload=True)

            self.raw_data.resample(self.sampling_frequency)
            self.raw_data.filter(l_freq=0.3, h_freq=40.0, verbose=False)  # EEG常见频段范围
            self.raw_data.notch_filter(freqs=49, verbose=False)  # 工频噪声

            log_message("✅ 已完成滤波处理（Notch + Band-pass）")

            self.channel_names = [ch for ch in self.raw_data.info['ch_names']
                                  if any(d in ch for d in self.desired_channel_names)]

            log_message(f"处理文件：{data_file}")

            self.raw_annotations = mne.read_annotations(self.anno_files_names[i])

            log_message(f"采样率：{self.sampling_frequency}")
            log_message(f"持续时间：{self.raw_data.times[-1]}")

            self.get_data_labels()
            self.save_files(i)
            log_message("\n")

if __name__ == "__main__":
    processor = DataPreprocessing()
    processor.data_preprocessing()

