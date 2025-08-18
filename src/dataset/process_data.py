import pandas as pd
from torch.utils.data import Dataset
import re
import numpy as np
import torch


class BloomsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(classes)
    weights = total_samples / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation but keep words/numbers
    text = re.sub(r"[^\w\s]", "", text)
    # 3. Normalize whitespace (remove multiple spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text)
    # 4. Strip leading/trailing spaces
    text = text.strip()

    return text


def load_dataset(filepath: str, clean=False):
    df = pd.read_csv(filepath)

    if clean:
        df["Text"] = df["Text"].apply(clean_text)

    # Encode categories to based on blooms level
    category_map = {
        "Remember": 0,
        "Understand": 1,
        "Apply": 2,
        "Analyse": 3,
        "Evaluate": 4,
        "Create": 5,
    }

    df["Label"] = df["Label"].map(category_map)
    return df
