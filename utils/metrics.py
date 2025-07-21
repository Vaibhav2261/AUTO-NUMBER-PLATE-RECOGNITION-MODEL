from sklearn.metrics import precision_score, recall_score

def calculate_ocr_accuracy(true_texts, predicted_texts):
    correct = 0
    for t, p in zip(true_texts, predicted_texts):
        if t.strip().upper() == p.strip().upper():
            correct += 1
    return correct / len(true_texts) if true_texts else 0

def compute_detection_metrics(y_true, y_pred, iou_threshold=0.5):
    # Basic mAP mockup: use precision_score and recall_score for now
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }
