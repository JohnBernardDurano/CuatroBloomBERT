from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.show()


def plot_training_history(history):
    acc_train = history["train_acc"]
    acc_val = history["val_acc"]
    loss_train = history["train_loss"]
    loss_val = history["val_loss"]

    epochs = range(1, len(acc_train) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(epochs, acc_train, label="Training Accuracy", marker="o")
    axes[0].plot(epochs, acc_val, label="Validation Accuracy", marker="o")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(epochs, loss_train, label="Training Loss", marker="o")
    axes[1].plot(epochs, loss_val, label="Validation Loss", marker="o")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
