from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)

def extract_prf_from_report(report_dict, class_labels):
    # Extract per-class precision/recall/f1; fill missing classes with 0.0
    precision, recall, f1 = [], [], []
    for label in class_labels:
        key = str(label)
        if key in report_dict:
            precision.append(report_dict[key]["precision"])
            recall.append(report_dict[key]["recall"])
            f1.append(report_dict[key]["f1-score"])
        else:
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)
    return precision, recall, f1

def format_confusion_matrix(matrix, is_percent=False):
    # Tab-separated rows for clean console printing
    lines = []
    for row in matrix:
        if is_percent:
            lines.append("\t".join(f"{v:.2f}" for v in row))
        else:
            lines.append("\t".join(str(int(v)) for v in row))
    return "\n".join(lines)

def calculate_metrics(preds, labels):
    # Basic validation
    if len(preds) == 0 or len(labels) == 0:
        raise ValueError("preds or labels is empty")
    if len(preds) != len(labels):
        raise ValueError(f"Length mismatch: preds({len(preds)}) vs labels({len(labels)})")

    # Confusion matrix (counts + row-normalized percent)
    cm = confusion_matrix(labels, preds)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    # Classification report (dict + text)
    report_dict = classification_report(labels, preds, output_dict=True)
    class_labels = sorted(set(labels) | set(preds))

    # Per-class metrics
    precision, recall, f1_per_class = extract_prf_from_report(report_dict, class_labels)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "kappa": cohen_kappa_score(labels, preds),
        "classification_report": classification_report(labels, preds),

        "confusion_matrix_percent_str": format_confusion_matrix(cm_percent, is_percent=True),
        "confusion_matrix_count_str": format_confusion_matrix(cm),

        "confusion_matrix_percent": cm_percent,
        "confusion_matrix_count": cm,

        "precision": precision,
        "recall": recall,
        "f1_score": f1_per_class,
        "class_labels": class_labels,
    }
