import h5py
import math
#
def calculate_steps_per_epoch(file_names,batch_size):
    total_steps = 0
    for file in file_names:
        with h5py.File(file, 'r') as f:
            # 直接读取标签数据的样本数
            stage_labels = f['labels']['stage_labels']
            file_samples = stage_labels.shape[0]
            # 计算当前文件需要的步数（向上取整）
            steps_per_file = math.ceil(file_samples / batch_size)
            total_steps += steps_per_file
    return total_steps