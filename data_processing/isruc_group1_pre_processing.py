import os
import numpy as np
import mne
import h5py
from tqdm import tqdm
import re

# ======== 配置路径 ========
subset_path = r"F:\EEGDataset\ISRUC_3\data"
save_path = r"F:\EEGDataset\ISRUC_3\data_anno_savefiles"
os.makedirs(save_path, exist_ok=True)

# ======== 参数设置 ========
EEG_BASES = ['C3-', 'C4-', 'O1-', 'O2-', 'F3-', 'F4-']   # 固定 6 个 EEG
EOG_PRI_BASES = ['LOC-', 'ROC-']                          # 优先使用
EOG_FALLBACK_BASES = ['E1-', 'E2-']                       # 不足再补
sampling_frequency = 100
epoch_duration = 30  # 每段长度（秒）
samples_per_epoch = epoch_duration * sampling_frequency

def match_pattern_for_base(base: str):
    """给定基础名（如 'C3-'），返回用于匹配 raw.ch_names 的正则。允许 -A1/-A2/-M1/-M2 或无参考后缀。"""
    b = base.strip('-')
    return rf'^{re.escape(b)}(?:-(?:A1|A2|M1|M2))?$'

def find_actual_channel(raw_names, base):
    """在 raw.ch_names 里找到与 base 匹配的真实通道名（含参考后缀），找不到返回 None。"""
    pat = re.compile(match_pattern_for_base(base), flags=re.IGNORECASE)
    for name in raw_names:
        if pat.fullmatch(name):
            return name
    return None

def load_labels(label_file):
    """ISRUC 标签 0~5 映射为 0~4（N3/N4 合并为 3，R 为 4）"""
    mapping = {0:0, 1:1, 2:2, 3:3, 4:3, 5:4}
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = int(line)
                if value in mapping:
                    labels.append(mapping[value])
            except ValueError:
                continue
    print(f"映射后有效标签数量: {len(labels)}")
    return labels

def save_h5(filename, eeg_names, eog_names, data_dict, labels):
    order_names = list(eeg_names) + list(eog_names)
    order_types = ['EEG'] * len(eeg_names) + ['EOG'] * len(eog_names)

    assert len(order_names) == 8 and len(order_types) == 8

    with h5py.File(filename, 'w') as f:
        data_group = f.create_group('data', track_order=True)

        sdt = h5py.string_dtype(encoding='utf-8')
        data_group.attrs['channel_order'] = np.array(order_names, dtype=sdt)
        data_group.attrs['channel_type']  = np.array(order_types, dtype=sdt)

        # 关键：前缀索引，确保任何工具显示都是 0..7 的顺序
        for i, ch_name in enumerate(order_names):
            ds_name = f'{i:02d}_{ch_name}_data'
            data_group.create_dataset(ds_name, data=data_dict[ch_name])

        label_group = f.create_group('labels', track_order=True)
        label_group.create_dataset('stage_labels', data=np.array(labels, dtype=np.int64))

def process_one_subject(subject_folder):
    subject_id = os.path.basename(subject_folder)
    edf_path = os.path.join(subject_folder, f"{subject_id}.edf")
    label_path = os.path.join(subject_folder, f"{subject_id}_1.txt")
    save_file = os.path.join(save_path, f"{subject_id}.h5")

    if os.path.exists(save_file):
        print(f"{subject_id} 已存在，跳过")
        return

    if not os.path.exists(edf_path) or not os.path.exists(label_path):
        print(f"{subject_id} 缺少 EDF 或 TXT 文件，跳过")
        return

    print(f"\n处理 {subject_id}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(sampling_frequency)

    raw_names = raw.ch_names

    # —— 先定位 6 个 EEG —— #
    eeg_actual = []
    for base in EEG_BASES:
        name = find_actual_channel(raw_names, base)
        if name is None:
            print(f"{subject_id} 缺少 EEG 通道 {base}，跳过")
            return
        eeg_actual.append(name)

    # —— 再定位 2 个 EOG：先尝试 LOC/ROC，不足用 E1/E2 补 —— #
    eog_actual = []
    for base in EOG_PRI_BASES:
        name = find_actual_channel(raw_names, base)
        if name:
            eog_actual.append(name)

    if len(eog_actual) < 2:
        for base in EOG_FALLBACK_BASES:
            if len(eog_actual) >= 2:
                break
            name = find_actual_channel(raw_names, base)
            if name and name not in eog_actual:
                eog_actual.append(name)

    if len(eog_actual) < 2:
        print(f"{subject_id} 未能凑齐 2 个 EOG（优先 LOC/ROC，备用 E1/E2），跳过")
        return

    # 用于统一切片、保存（保存时按“先 EEG 后 EOG”写入）
    ordered_channels = eeg_actual + eog_actual
    print(f"{subject_id} 通道顺序（前6 EEG，后2 EOG）：{ordered_channels}")

    # 获取数据（如需微伏：* 1e6）
    data = {}
    for ch in ordered_channels:
        data[ch] = raw.get_data(picks=ch).flatten()  # * 1e6 转微伏可在此乘

    # 加载与裁剪标签
    labels = load_labels(label_path)
    total_epochs = min(len(data[ordered_channels[0]]) // samples_per_epoch, len(labels))
    print(f"{subject_id} 可用数据段数: {total_epochs}")
    if total_epochs == 0:
        print(f"{subject_id} 数据或标签太短，跳过")
        return

    # 切分为 (epochs, samples_per_epoch)
    for ch in ordered_channels:
        data[ch] = data[ch][:total_epochs * samples_per_epoch].reshape(total_epochs, samples_per_epoch)
        print(f"{subject_id} 通道 {ch} 形状: {data[ch].shape}，前10个采样值: {data[ch][0][:10]}")

    labels = labels[:total_epochs]

    # 保存（同层级下以通道名命名，顺序先 EEG 后 EOG）
    save_h5(save_file, eeg_actual, eog_actual, data, labels)
    print(f"{subject_id} 保存成功，文件: {save_file}")

def main():
    folders = sorted(
        [os.path.join(subset_path, d) for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))],
        key=lambda x: int(os.path.basename(x))
    )
    for folder in tqdm(folders, desc="Processing Subjects"):
        process_one_subject(folder)

if __name__ == '__main__':
    main()
