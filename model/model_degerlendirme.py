import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, roc_curve, auc, precision_recall_curve, log_loss
from sklearn.preprocessing import label_binarize
import joblib
import seaborn as sns


def extract_face(img, detector, target_size=(224, 224)):
    results = detector.detect_faces(img)
    if not results:
        new_face = crop_and_resize(img, target_w=224, target_h=224)
    else:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        adj_h = 10
        new_y1 = max(y1 - adj_h, 0)
        new_y2 = min(y1 + height + adj_h, img.shape[0])
        new_height = new_y2 - new_y1
        adj_w = int((new_height - width) / 2)
        new_x1 = max(x1 - adj_w, 0)
        new_x2 = min(x2 + adj_w, img.shape[1])
        new_face = img[new_y1:new_y2, new_x1:new_x2]
    sqr_img = cv2.resize(new_face, target_size)
    return sqr_img


def crop_and_resize(image, target_w=224, target_h=224):
    if image.ndim == 2:
        img_h, img_w = image.shape
    elif image.ndim == 3:
        img_h, img_w, channels = image.shape
    target_aspect_ratio = target_w / target_h
    input_aspect_ratio = img_w / img_h
    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio * target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w, resize_h))
        crop_left = int((resize_w - target_w) / 2)
        crop_right = crop_left + target_w
        new_img = img[:, crop_left:crop_right]
    elif input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w / input_aspect_ratio)
        img = cv2.resize(image, (resize_w, resize_h))
        crop_top = int((resize_h - target_h) / 4)
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    else:
        new_img = cv2.resize(image, (target_w, target_h))
    return new_img


def load_data(data_path):
    X_test_rgb = joblib.load(os.path.join(data_path, 'X_test_rgb.joblib'))
    y_test_rgb = joblib.load(os.path.join(data_path, 'y_test_rgb.joblib'))
    return X_test_rgb, y_test_rgb


def evaluate_model(model, X_test, y_test, label_map, output_folder):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auc_score = roc_auc_score(label_binarize(y_true, classes=list(label_map.keys())), y_pred_probs,
                                  average='weighted', multi_class='ovo')
    except Exception as e:
        auc_score = None
    logloss = log_loss(y_true, y_pred_probs)
    class_report = classification_report(y_true, y_pred, target_names=list(label_map.values()), zero_division=0)
    with open(os.path.join(output_folder, 'model_metrics.txt'), 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("==========================\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Weighted): {precision:.4f}\n")
        f.write(f"Recall (Weighted): {recall:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1:.4f}\n")
        f.write(f"Log Loss: {logloss:.4f}\n")
        if auc_score is not None:
            f.write(f"AUC Score (Weighted): {auc_score:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(label_map.values()), yticklabels=list(label_map.values()),
                cmap='Blues')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.title('Confusion Matrix - Counts')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_counts.png'))
    plt.close()
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=list(label_map.values()),
                yticklabels=list(label_map.values()), cmap='Blues')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.title('Confusion Matrix - Normalized')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_normalized.png'))
    plt.close()
    y_true_bin = label_binarize(y_true, classes=list(label_map.keys()))
    n_classes = y_true_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {label_map[i]} (AUC = {roc_auc_dict[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_folder, 'roc_curves.png'))
    plt.close()
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall_curve, precision_curve, label=f'Precision-Recall curve of class {label_map[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_folder, 'precision_recall_curves.png'))
    plt.close()
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Log Loss': logloss
    }
    if auc_score is not None:
        metrics['AUC Score'] = auc_score
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    for index, value in enumerate(metrics.values()):
        plt.text(index, value + 0.01, f"{value:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'evaluation_metrics.png'))
    plt.close()


def plot_metrics(history, output_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'training_validation_loss.png'))
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'training_validation_accuracy.png'))
    plt.close()


if __name__ == "__main__":
    data_path = r"C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\FaceShape Dataset\pickles"
    model_path = r"C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\model\face_shape_model_vgg16_rgb.h5"
    output_folder = r"C:\Users\duggy\OneDrive\Belgeler\Github\AIProject\model\evaluation_results"
    label_map = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    X_test_rgb, y_test_rgb = load_data(data_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    model = load_model(model_path)
    evaluate_model(model, X_test_rgb, y_test_rgb, label_map, output_folder)
