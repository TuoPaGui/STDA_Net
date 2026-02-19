from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,f1_score, cohen_kappa_score

def extract_prf_from_report(report_dict, class_labels):
    precision = []
    recall = []
    f1 = []
    for label in class_labels:
        if str(label) in report_dict:
            precision.append(report_dict[str(label)]['precision'])
            recall.append(report_dict[str(label)]['recall'])
            f1.append(report_dict[str(label)]['f1-score'])
        else:
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)
    return precision, recall, f1

# 将混淆矩阵格式化为字符串，支持百分比或整数显示
def format_confusion_matrix(matrix, is_percent=False):
    formatted_lines = []
    for row in matrix:
        if is_percent:
            line = "\t".join([f"{value:.2f}" for value in row])
        else:
            line = "\t".join([str(int(value)) for value in row])
        formatted_lines.append(line)
    return "\n".join(formatted_lines)

# 计算分类任务的各类指标（准确率、F1、Kappa、混淆矩阵、分类报告等）
def calculate_metrics(preds, labels):
    # ========= 数据校验 =========
    if len(preds) == 0 or len(labels) == 0:
        raise ValueError("预测值或标签为空")
    if len(preds) != len(labels):
        raise ValueError(f"数据长度不一致: preds({len(preds)}) vs labels({len(labels)})")

    # ========= 混淆矩阵 =========
    confusion_mat = confusion_matrix(labels, preds)
    confusion_mat_percent = confusion_mat / confusion_mat.sum(axis=1, keepdims=True) * 100

    # ========= 分类报告 =========
    report_dict = classification_report(labels, preds, output_dict=True)
    class_labels = sorted(set(labels) | set(preds))  # 保证所有类都被覆盖

    # ========= 提取每类指标 =========
    precision, recall, f1 = extract_prf_from_report(report_dict, class_labels)

    # ========= 返回结构 =========
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
        'kappa': cohen_kappa_score(labels, preds),
        'classification_report': classification_report(labels, preds),

        # 美观打印
        'confusion_matrix_percent_str': format_confusion_matrix(confusion_mat_percent, is_percent=True),
        'confusion_matrix_count_str': format_confusion_matrix(confusion_mat),

        # 数值结果（便于绘图或保存）
        'confusion_matrix_percent': confusion_mat_percent,
        'confusion_matrix_count': confusion_mat,

        # 每类精度、召回率、F1
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_labels': class_labels
    }