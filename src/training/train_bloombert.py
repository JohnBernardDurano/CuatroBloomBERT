import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as nlpaw

from src.model.bloombert import BloomBERT
from src.dataset.process_data import BloomsDataset
from src.helper.plots_helper import plot_confusion_matrix
from src.dataset.augment import augment_data


def train_model_bloombert(
    df, tokenizer, config, class_weights=None, test_size=0.2, augment=False
):
    X_train, X_val, y_train, y_val = train_test_split(
        df["Text"],
        df["Label"],
        test_size=test_size,
        random_state=1234,
        stratify=df["Label"],
    )

    train = pd.DataFrame({"Text": X_train, "Label": y_train})
    val = pd.DataFrame({"Text": X_val, "Label": y_val})

    if augment:
        import nltk

        nltk.download("wordnet")
        nltk.download("averaged_perceptron_tagger_eng")

        aug = nlpaw.SynonymAug(aug_src="wordnet", aug_max=3)
        max_count = max(train["Label"].value_counts())
        print("Oversampling training data")
        print("-- Initial training data distribution")
        print(train["Label"].value_counts().sort_index())
        train = augment_data(train, aug, target_count=max_count)
        print("-- Augmented training data distribution")
        print(train["Label"].value_counts().sort_index())

    train_encodings = tokenizer(list(train["Text"]), truncation=True, padding=True)
    val_encodings = tokenizer(list(val["Text"]), truncation=True, padding=True)

    train_dataset = BloomsDataset(train_encodings, list(train["Label"]))
    val_dataset = BloomsDataset(val_encodings, list(val["Label"]))

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # ------------------------
    # Model, Loss, Optimizer
    # ------------------------
    model = BloomBERT(output_dim=6).to(config["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(
        config["device"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.1
    )

    # Freeze BERT encoder at the start
    for param in model.bert.parameters():
        param.requires_grad = False

    # ------------------------
    # Training loop
    # ------------------------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    best_model = None

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        correct_train, total_train = 0, 0

        if epoch == 5:
            for layer in model.bert.transformer.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if epoch == 30:
            for layer in model.bert.transformer.layer[-3:]:
                for param in layer.parameters():
                    param.requires_grad = True

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["labels"].to(config["device"])

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config["device"])
                attention_mask = batch["attention_mask"].to(config["device"])
                labels = batch["labels"].to(config["device"])

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    plot_confusion_matrix(
        all_labels,
        all_preds,
        labels=["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"],
        title="Confusion Matrix - Final Validation",
    )

    return best_model, history, best_val_acc
