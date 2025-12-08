import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_all_preds(model, loader, device=device):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    return all_preds, all_probs, all_targets


def plot_history(history):
    epochs = history['epoch']

    # Loss curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=history["train_loss"], marker="o", label="Train Loss")
    sns.lineplot(x=epochs, y=history["val_loss"], marker="o", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Accuracy curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=history["train_acc"], marker="o", label="Train Acc")
    sns.lineplot(x=epochs, y=history["val_acc"], marker="o", label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


def evaluate_model(model, test_loader, class_names=None):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Confusion Matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # F1, precision, recall
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")

    # ROC-AUC (one-vs-rest)
    try:
        auc_macro = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
        print(f"ROC-AUC (macro): {auc_macro:.4f}")
        print(f"ROC-AUC (weighted): {auc_weighted:.4f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    return y_true, y_pred, y_probs
