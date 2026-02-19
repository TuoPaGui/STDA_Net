import os

def log_message(message):
    log_file_path = os.path.join("G:\\Research\\EEG_Project\\Template\\CodeDir", '模型训练日志.txt')
    with open(log_file_path, 'a') as log_file:  # 以追加模式打开文件
        log_file.write(f"{message}\n")  # 写入时间戳和消息

def log_confusion_matrix_with_metrics(cm_percent, pr_list, f1_list, re_list, class_names, title):
    num_classes = len(class_names)
    col_width = 9  # 每列固定宽度

    # 构建表头
    header = "Predicted".ljust(col_width)
    header += "".join(name.rjust(col_width) for name in class_names)
    header += "".join(metric.rjust(col_width) for metric in ["PR", "F1", "RE"])

    log_message(f"\n[{title}_Confusion Matrix]")
    log_message(header)

    # 每行真实标签对应的混淆+指标
    for i in range(num_classes):
        row = class_names[i].ljust(col_width)
        row += "".join(f"{cm_percent[i][j]:>{col_width}.2f}" for j in range(num_classes))
        row += f"{pr_list[i]:>{col_width}.2f}{f1_list[i]:>{col_width}.2f}{re_list[i]:>{col_width}.2f}"
        log_message(row)