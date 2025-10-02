"""
Protein Secondary Structure Prediction with ESM and CNN-LSTM Ensemble

This module implements a deep learning pipeline for residue-level secondary structure
prediction using pretrained ESM transformer embeddings, LSTM layers, and a CNN decoder.
It includes dataset loading, tokenization, training with Ray Tune hyperparameter search,
ensemble evaluation, and submission file generation.

Dependencies:
- PyTorch
- HuggingFace Transformers
- Ray Tune (with HyperOpt)
- BioPython, scikit-learn, matplotlib, pandas

Author: Pierce
"""

# Standard Library
import argparse
import json
import os
import random
import tempfile
from typing import Optional

# Third-party Libraries
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from Bio import SeqIO
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# HuggingFace Transformers
from transformers import AutoTokenizer, EsmModel

# Ray (Tuning and Configs)
from ray import tune
from ray.air import session
from ray.tune import Tuner, TuneConfig, with_resources, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import Checkpoint

# Seed Initialization
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

_tokenizer = None
def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    return _tokenizer

def load_esm_model(device=DEVICE):
    return EsmModel.from_pretrained(ESM_MODEL_NAME, add_pooling_layer=False).to(device)

class ResidueLevelDataset(Dataset):
    """
    Dataset for protein sequences and residue-level structure labels.

    Loads sequences from a FASTA file and per-residue labels from a TSV file.
    Optionally tokenizes sequences using a provided tokenizer.
    """

    def __init__(
        self,
        fasta_path: str,
        annotation_path: str,
        label_order: Optional[list[str]] = None,
        tokenizer: Optional[object] = None,
        max_length: int = 1022
    ) -> None:
        """
        Initialize with paths to FASTA and annotation files.

        Args:
            fasta_path (str): Path to FASTA file.
            annotation_path (str): Path to annotation TSV file.
            label_order (list[str], optional): Order of structure labels.
            tokenizer (optional): Tokenizer to convert sequences.
            max_length (int): Max sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_order = label_order or ['H', 'B', 'E', 'G', 'I', 'P', 'T', 'S', '.']
        self.label_to_idx = {
            label: i for i, label in enumerate(self.label_order)
        }

        self.sequences = {
            record.id: str(record.seq)
            for record in SeqIO.parse(fasta_path, "fasta")
        }

        df = pd.read_csv(annotation_path, sep='\t', comment='#').dropna()

        self.data = {}
        for _, row in df.iterrows():
            full_id = row["id"]
            pdb_id, _, pos = full_id.split('_')
            pos = int(pos)
            label = row["secondary_structure"]

            if pdb_id not in self.data:
                self.data[pdb_id] = {}
            self.data[pdb_id][pos] = self.label_to_idx[label]

        self.pdb_ids = [
            pid for pid in self.sequences
            if pid in self.data and len(self.sequences[pid]) <= max_length
        ]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Return dict with sequence, label tensor, ID, and optional input IDs.

        Args:
            idx (int): Index of sample.

        Returns:
            dict: Sequence data and labels.
        """
        pid = self.pdb_ids[idx]
        sequence = self.sequences[pid]
        residue_labels = self.data[pid]

        labels = torch.full(
            (len(sequence),),
            self.label_to_idx['.'],
            dtype=torch.long
        )
        for i in range(len(sequence)):
            pos = i + 1
            if pos in residue_labels:
                labels[i] = residue_labels[pos]

        input_ids = None
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                sequence,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0]

        return {
            "sequence": sequence,
            "labels": labels,
            "sequence_id": pid,
            "input_ids": input_ids
        }


class ESMForSecondaryStructure(nn.Module):
    """
    Neural network model for residue-level secondary structure prediction
    using ESM embeddings, LSTM, and CNN layers.

    Args:
        esm_model (EsmModel): Pretrained ESM model.
        num_labels (int): Number of output structure labels.
        dropout_rate (float): Dropout probability.
        cnn_sizes (list[int], optional): Number of filters for each CNN layer.
    """

    def __init__(
        self,
        esm_model,
        num_labels: int,
        dropout_rate: float = 0.1,
        cnn_sizes: Optional[list[int]] = None
    ) -> None:
        super().__init__()
        self.esm = esm_model
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = esm_model.config.hidden_size

        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=40,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        cnn_sizes = cnn_sizes or [64, 48, 32, 16]
        self.cnn = nn.Sequential(
            nn.Conv1d(80, cnn_sizes[0], kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(cnn_sizes[0], cnn_sizes[1], kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(cnn_sizes[1], cnn_sizes[2], kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(cnn_sizes[2], cnn_sizes[3], kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(cnn_sizes[3], num_labels, kernel_size=1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [B, T].
            attention_mask (torch.Tensor): Attention mask [B, T].

        Returns:
            torch.Tensor: Logits for each residue position [B, T, C].
        """
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        hidden_states = self.dropout(hidden_states)
        rnn_out, _ = self.rnn(hidden_states)  # [B, T, 80]
        cnn_input = rnn_out.permute(0, 2, 1)  # [B, 80, T]
        logits = self.cnn(cnn_input).permute(0, 2, 1)  # [B, T, C]

        return logits


def esm_collate_fn(
    batch: list,
    tokenizer,
    ignore_index: int = -100,
    config: Optional[dict] = None
) -> tuple[dict, torch.Tensor]:
    """
    Collate function for batching sequences and labels for ESM models.

    Pads and tokenizes sequences with the provided tokenizer and aligns
    per-residue labels, inserting ignore indices for special tokens.

    Args:
        batch (list): List of samples from the dataset.
        tokenizer: HuggingFace tokenizer to tokenize sequences.
        ignore_index (int): Index to use for ignored/padded label positions.
        config (dict, optional): Configuration with optional "max_length".

    Returns:
        tuple: (tokenized_inputs, padded_labels)
            tokenized_inputs (dict): Tokenized tensors with attention mask.
            padded_labels (torch.Tensor): Padded label tensor [B, T].
    """
    sequences = [item["sequence"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    max_length = config.get("max_length", 1024) if config else 1024

    tokenized = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )

    max_len = tokenized["input_ids"].shape[1]
    padded_labels = torch.full(
        (len(sequences), max_len),
        ignore_index,
        dtype=torch.long
    )

    for i, labels in enumerate(labels_list):
        seq_len = min(len(labels), max_len - 2)  # Account for [CLS] + [EOS]
        padded_labels[i, 1:seq_len + 1] = labels[:seq_len]

    return tokenized, padded_labels


FASTA_PATH = "/home/pie_crusher/CS121/Project1/sequences.fasta"
LABEL_PATH = "/home/pie_crusher/CS121/Project1/train.tsv"

def build_data_loaders(
    batch_size: int,
    fasta_path: str,
    label_path: str,
    tokenizer,
    dataset_class,
    config: dict = None,
    fold: int = 0,
    num_folds: int = 5):
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
    df = pd.read_csv(label_path, sep='\t')
    df["pdb_id"] = df["id"].apply(lambda x: x.split("_")[0])

    LABEL_ORDER = ['H', 'B', 'E', 'G', 'I', 'P', 'T', 'S', '.']
    sequence_residue_counts = {}
    for pdb_id, group in df.groupby("pdb_id"):
        if pdb_id not in sequences:
            continue
        labels = group["secondary_structure"].values
        residue_counter = Counter(labels)
        sequence_residue_counts[pdb_id] = residue_counter

    sequence_ids = list(sequence_residue_counts.keys())

    multi_hot_labels = []
    for seq_id in sequence_ids:
        counts = sequence_residue_counts[seq_id]
        total = sum(counts.values())
        label_vector = [counts.get(cls, 0) / total for cls in LABEL_ORDER]
        multi_hot_labels.append(label_vector)

    print(f"Generated multi-label proportion vectors for {len(sequence_ids)} sequences.")

    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    splits = list(mskf.split(sequence_ids, multi_hot_labels))

    train_seq_ids = set([sequence_ids[i] for i in splits[fold][0]])
    val_seq_ids = set([sequence_ids[i] for i in splits[fold][1]])

    full_dataset = dataset_class(fasta_path, label_path, tokenizer=tokenizer, max_length=config.get("max_length", 1022))

    id_to_idx = {item["sequence_id"]: idx for idx, item in enumerate(full_dataset)}

    train_indices = [id_to_idx[seq_id] for seq_id in train_seq_ids if seq_id in id_to_idx]
    val_indices = [id_to_idx[seq_id] for seq_id in val_seq_ids if seq_id in id_to_idx]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    collate = lambda batch: esm_collate_fn(batch, tokenizer, config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    test_loader = None
    if "test.tsv" in os.listdir(".") and "sequences.fasta" in os.listdir("."):
        test_df = pd.read_csv("test.tsv", sep='\t')
        label_to_idx = {label: i for i, label in enumerate(LABEL_ORDER)}

        test_ids = test_df["id"].tolist()
        test_label_map = {}
        for full_id in test_ids:
            pdb_id, _, pos = full_id.split("_")
            pos = int(pos)
            if pdb_id not in test_label_map:
                test_label_map[pdb_id] = {}
            test_label_map[pdb_id][pos] = label_to_idx["."]

        sequences = {
            record.id: str(record.seq)
            for record in SeqIO.parse("sequences.fasta", "fasta")
            if record.id in test_label_map
        }

        class TestResidueDataset(Dataset):
            def __init__(self):
                self.pdb_ids = list(sequences.keys())

            def __len__(self):
                return len(self.pdb_ids)

            def __getitem__(self, idx):
                pid = self.pdb_ids[idx]
                sequence = sequences[pid]
                pos_to_label = test_label_map[pid]

                labels = torch.full(
                    (len(sequence),),
                    label_to_idx["."],
                    dtype=torch.long
                )
                for i in range(len(sequence)):
                    pos = i + 1
                    if pos in pos_to_label:
                        labels[i] = pos_to_label[pos]

                input_ids = tokenizer(
                    sequence,
                    return_tensors="pt",
                    add_special_tokens=False
                )["input_ids"][0]

                return {
                    "sequence": sequence,
                    "labels": labels,
                    "sequence_id": pid,
                    "input_ids": input_ids,
                }

        test_dataset = TestResidueDataset()
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate
        )

    print(f"DataLoaders ready with multi-label stratified K-Fold (Fold {fold+1}/{num_folds}).")
    return train_loader, val_loader, test_loader


def compute_token_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> tuple[float, float, float, float]:
    """
    Compute accuracy, precision, recall, and F1 for token classification.

    Args:
        preds (torch.Tensor): Predicted label indices.
        labels (torch.Tensor): True label indices.
        ignore_index (int): Label index to ignore in metrics.

    Returns:
        tuple: Accuracy, precision, recall, and F1 score (macro averaged).
    """
    mask = labels != ignore_index
    true = labels[mask].cpu().numpy().flatten()
    pred = preds[mask].cpu().numpy().flatten()
    acc = (true == pred).mean()
    precision = precision_score(true, pred, average='macro', zero_division=0)
    recall = recall_score(true, pred, average='macro', zero_division=0)
    f1 = f1_score(true, pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

def plot_training_curves(history: dict) -> None:
    """
    Plot and save training and validation curves.

    Args:
        history (dict): Dictionary with keys "train_loss", "val_loss",
                        "val_acc", and "val_f1" mapping to lists of metrics.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.plot(epochs, history["val_f1"], label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

def train_model(
    model: nn.Module,
    config: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    start_epoch: int = 0,
    resume_checkpoint: Optional[Checkpoint] = None,
    allow_resume: bool = True
) -> dict:
    """
    Train a model and evaluate on validation set with optional checkpointing.

    Args:
        model (nn.Module): Model to train.
        config (dict): Training configuration including learning rate, patience, etc.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader or None): Validation data loader.
        start_epoch (int): Epoch to start/resume from.
        resume_checkpoint (Checkpoint or None): Optional Ray Tune checkpoint.
        allow_resume (bool): If True, resume from saved local checkpoint if available.

    Returns:
        dict: Best validation metrics and full training history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0)
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler(device="cuda")

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    final_epoch = config.get("num_epochs")
    patience = config.get("patience", 3)
    patience_counter = 0

    trial_id = os.environ.get("RAY_TRIAL_ID", "final")
    checkpoint_path = f"checkpoints/best_model_{trial_id}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    fold_id = config.get("fold", 0)
    writer = SummaryWriter(log_dir=f"runs/fold_{fold_id}")

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    if resume_checkpoint is not None:
        with resume_checkpoint.as_directory() as ckpt_dir:
            ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pth"), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
            start_epoch = ckpt.get("epoch", start_epoch) + 1
            print(f"Resumed from Ray checkpoint: epoch {start_epoch}")

    elif allow_resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            print("\u26a0\ufe0f Warning: Optimizer state not restored.")
        best_val_loss = checkpoint.get("val_loss", best_val_loss)
        best_val_f1 = checkpoint.get("val_f1", best_val_f1)
        best_val_acc = checkpoint.get("val_accuracy", best_val_acc)
        start_epoch = checkpoint.get("epoch", start_epoch) + 1
        print(f"Resumed from local checkpoint: epoch {start_epoch}")

    for epoch in range(start_epoch, config["num_epochs"]):
        final_epoch = epoch + 1
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits = model(**inputs)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = labels.to(device)

                    with autocast(device_type="cuda"):
                        logits = model(**inputs)
                        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                        preds = torch.argmax(logits, dim=-1)

                    val_loss += loss.item()
                    preds, labels = preds.cpu(), labels.cpu()
                    for i in range(preds.size(0)):
                        mask = labels[i] != -100
                        all_preds.extend(preds[i][mask].tolist())
                        all_labels.extend(labels[i][mask].tolist())

            avg_val_loss = val_loss / len(val_loader)
            acc, _, _, f1 = compute_token_metrics(
                torch.tensor(all_preds), torch.tensor(all_labels)
            )
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(acc)
            history["val_f1"].append(f1)

            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/Validation", acc, epoch)
            writer.add_scalar("F1/Validation", f1, epoch)

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_val_loss = avg_val_loss
                best_val_acc = acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "final_epoch": final_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": best_val_acc,
                    "val_f1": best_val_f1,
                    "config": config,
                    "history": history
                }, checkpoint_path)

                if session.get_session():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss
                        }, os.path.join(tmpdir, "checkpoint.pth"))
                        session.report(
                            {
                                "val_loss": best_val_loss,
                                "val_accuracy": best_val_acc,
                                "val_f1": best_val_f1,
                                "final_epoch": final_epoch
                            },
                            checkpoint=Checkpoint.from_directory(tmpdir)
                        )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {final_epoch}")
                    break

    writer.close()

    return {
        "val_f1": best_val_f1,
        "val_loss": best_val_loss,
        "val_accuracy": best_val_acc,
        "final_epoch": final_epoch,
        "history": history
    }


def load_model_and_config(checkpoint_path: str = "checkpoints/best_model_final.pth"):
    """
    Load a trained model and its config from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        tuple: (model, config) where model is the trained model in eval mode,
            and config is the training configuration dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    esm_model = load_esm_model(device)

    num_labels = config.get("num_labels", 9)

    model = ESMForSecondaryStructure(
        esm_model=esm_model,
        num_labels=num_labels,
        dropout_rate=config.get("dropout_rate", 0.1)
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, config

def train_tune(config: dict) -> None:
    """
    Train a model with Ray Tune using the provided hyperparameter config.

    This function supports resuming from previous checkpoints and reports metrics
    such as validation loss, accuracy, and F1 score back to Ray Tune.

    Args:
        config (dict): Configuration dictionary including learning rate, batch size,
                    dropout, CNN sizes, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm_backbone = load_esm_model(device)

    model = ESMForSecondaryStructure(
        esm_model=esm_backbone,
        num_labels=9,
        dropout_rate=config.get("dropout_rate", 0.1),
        cnn_sizes=config.get("cnn_sizes", [64, 48, 32, 16])
    ).to(device)

    config["cnn_sizes"] = config.get("cnn_sizes", [64, 48, 32, 16])

    checkpoint = session.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_data = torch.load(
                os.path.join(ckpt_dir, "checkpoint.pth"),
                map_location=device,
                weights_only=False
            )
            model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
            start_epoch = ckpt_data.get("epoch", 0) + 1
            
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    fold = 0
    train_loader, val_loader, _ = build_data_loaders(
        batch_size=config["batch_size"],
        fasta_path=FASTA_PATH,
        label_path=LABEL_PATH,
        tokenizer=tokenizer,
        dataset_class=ResidueLevelDataset,
        fold=fold,
        num_folds=config.get("num_folds", 5),
        config=config
    )


    result = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch
    )

    model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pth")
        torch.save({
            "epoch": result["final_epoch"],
            "model_state_dict": model.state_dict(),
            "config": config
        }, ckpt_path)

        session.report(
            {
                "val_loss": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
                "val_f1": result["val_f1"],
                "final_epoch": result["final_epoch"]
            },
            checkpoint=Checkpoint.from_directory(tmpdir)
        )

def train_ensemble_on_full_data(config: dict) -> None:
    """
    Train a 5-fold ensemble on different train/val splits using residue-aware stratified K-Fold.

    Each model is trained on a unique fold's training set and validated on its validation split.
    A unique random seed is set per fold to encourage initialization diversity and improve ensemble generalization.

    Args:
        config (dict): Best hyperparameter configuration.
    """
    print("Training 5-fold ensemble using K-Fold splitting and per-fold random seeds...")

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    for fold in range(config.get("num_folds", 5)):
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        random.seed(42 + fold)

        print(f"\nTraining Ensemble Model {fold + 1}/5 (Fold {fold})")

        train_loader, val_loader, _ = build_data_loaders(
            batch_size=config["batch_size"],
            fasta_path=FASTA_PATH,
            label_path=LABEL_PATH,
            tokenizer=tokenizer,
            dataset_class=ResidueLevelDataset,
            fold=fold,
            num_folds=config.get("num_folds", 5),
            config=config
        )

        model = ESMForSecondaryStructure(
            esm_model=load_esm_model(),
            num_labels=config.get("num_labels", 9),
            dropout_rate=config.get("dropout_rate", 0.1),
            cnn_sizes=config.get("cnn_sizes", [64, 48, 32, 16])
        )

        result = train_model(
            model,
            config,
            train_loader,
            val_loader,
            allow_resume=False
        )

        save_path = f"checkpoints/final_model_fold_{fold}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {**config, "final_epoch": result["final_epoch"]},
            "history": result["history"]
        }, save_path)

        print(f"Saved fold {fold} model to: {save_path}")


def evaluate_ensemble_on_val_loader(val_loader):
    """
    Evaluate a 5-fold ensemble model on a validation data loader.

    Args:
        val_loader (DataLoader): DataLoader for validation sequences.

    Returns:
        tuple: (accuracy, precision, recall, F1 score) for ensemble predictions.
    """
    device = DEVICE
    print("Evaluating 5-fold ensemble on validation set...")

    ensemble_models = []
    for fold in range(5):
        ckpt_path = f"checkpoints/final_model_fold_{fold}.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint["config"]

        model = ESMForSecondaryStructure(
            esm_model=load_esm_model(device),
            num_labels=config.get("num_labels", 9),
            dropout_rate=config.get("dropout_rate", 0.1),
            cnn_sizes=config.get("cnn_sizes", [64, 48, 32, 16])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        ensemble_models.append(model)

    ensemble_preds = []
    all_labels_flat = []

    for _, (inputs, labels) in enumerate(val_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.cpu()
        B, _ = labels.shape

        batch_logits = []
        with torch.no_grad():
            for model in ensemble_models:
                with autocast(device_type="cuda"):
                    logits = model(**inputs)
                batch_logits.append(logits.detach().cpu())

        avg_logits = torch.stack(batch_logits).mean(dim=0)  # [B, T, C]
        preds = torch.argmax(avg_logits, dim=-1)  # [B, T]

        for i in range(B):
            mask = labels[i] != -100
            pred_i = preds[i][mask]
            label_i = labels[i][mask]
            ensemble_preds.extend(pred_i.tolist())
            all_labels_flat.extend(label_i.tolist())

    acc, precision, recall, f1 = compute_token_metrics(
        torch.tensor(ensemble_preds), torch.tensor(all_labels_flat)
    )

    print("\n5-Fold Ensemble Validation Evaluation")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels_flat, ensemble_preds, digits=4))

    return acc, precision, recall, f1

def generate_submission_csv(
    fasta_path: str = "sequences.fasta",
    test_path: str = "test.tsv",
    output_path: str = "predictions.csv"
) -> None:
    """
    Generate a submission CSV by predicting secondary structure labels for test data.

    Args:
        fasta_path (str): Path to the input FASTA file.
        test_path (str): Path to the test TSV file.
        output_path (str): Path to save the output predictions TSV.
    """
    print(f"Generating prediction TSV for Codabench at {output_path}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()

    sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse(fasta_path, "fasta")
    }

    test_df = pd.read_csv(test_path, sep="\t")
    test_df["secondary_structure"] = "."

    models = []
    for fold in range(5):
        ckpt = torch.load(f"checkpoints/final_model_fold_{fold}.pth", map_location=device)
        esm = load_esm_model(device)
        model = ESMForSecondaryStructure(
            esm_model=esm,
            num_labels=9,
            dropout_rate=ckpt["config"].get("dropout_rate", 0.1)
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    label_order = ['H', 'B', 'E', 'G', 'I', 'P', 'T', 'S', '.']
    idx_to_label = dict(enumerate(label_order))

    grouped = test_df.groupby(lambda x: test_df.iloc[x]["id"].split("_")[0])
    for pdb_id, group in grouped:
        if pdb_id not in sequences:
            continue

        sequence = sequences[pdb_id]
        tokenized = tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=True
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            logits_all = []
            for model in models:
                logits = model(**tokenized)
                logits_all.append(logits[0].detach().cpu())

            avg_logits = torch.stack(logits_all).mean(dim=0)
            pred_indices = torch.argmax(avg_logits, dim=-1).tolist()

        pred_indices = pred_indices[1:-1]  # Remove CLS/EOS tokens
        pred_labels = [idx_to_label[i] for i in pred_indices]

        for i in group.index:
            _, _, pos = test_df.at[i, "id"].split("_")
            pos = int(pos) - 1
            if pos < len(pred_labels):
                test_df.at[i, "secondary_structure"] = pred_labels[pos]
            else:
                test_df.at[i, "secondary_structure"] = "."

    test_df.to_csv(output_path, index=False, sep="," if output_path.endswith(".csv") else "\t")
    print(f"Submission saved to {output_path}")

    test_df[["id", "secondary_structure"]].to_csv(output_path, sep="\t", index=False)
    print(f"Done. TSV file saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-debug",
        action="store_true",
        help="Skip tuning and run directly with a preset config"
    )
    args = parser.parse_args()

    if not args.debug:
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 5e-3),
            "batch_size": tune.choice([1,2]),
            "dropout_rate": tune.uniform(0.1, 0.3),
            "max_length": tune.choice([512, 768, 1024]),
            "weight_decay": tune.choice([0.0, 0.005, 0.01]),
            "patience": tune.choice([2, 3, 5])
        }
        FIXED_NUM_EPOCHS = 5
        trainable_with_resources = with_resources(train_tune, {"cpu": 2, "gpu": 1})

        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=100,
            grace_period=1,
            reduction_factor=3
        )

        search_alg = HyperOptSearch(metric="val_f1", mode="max")

        tuner = Tuner(
            trainable_with_resources,
            param_space={**search_space, "num_epochs": FIXED_NUM_EPOCHS},
            tune_config=TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=15,
                max_concurrent_trials=4,
                metric="val_f1",
                mode="max"
            ),
            run_config=RunConfig(name="tune_hyperopt_50x")
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="val_f1", mode="max")
        print("Final epoch:", best_result.metrics["final_epoch"])
        best_config = best_result.config
        best_config["num_epochs"] = best_result.metrics["final_epoch"]

        best_checkpoint_path = os.path.join(best_result.checkpoint.path, "checkpoint.pth")
        FINAL_PATH = "checkpoints/best_model.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(torch.load(best_checkpoint_path, map_location="cpu", weights_only=False), FINAL_PATH)

        with open("checkpoints/best_model_config.json", "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)

        print("\nBest checkpoint saved to:", os.path.abspath(FINAL_PATH))
        print("Best Hyperparameters:", best_config)

    else:
        ckpt = torch.load("checkpoints/best_model.pth")
        print(ckpt["config"].get("cnn_sizes"))
        print("Debug mode: skipping hyperparameter tuning")
        best_config = {
            "learning_rate": 0.0002,
            "batch_size": 1,
            "dropout_rate": 0.25,
            "max_length": 1024,
            "weight_decay": 0.01,
            "patience": 5,
            "num_epochs": 5,
            "cnn_sizes": [64, 48, 32, 16],
            "num_labels": 9
        }
        FINAL_PATH = "checkpoints/best_model.pth"

    print("\nTraining on full dataset with best config")
    train_ensemble_on_full_data(best_config)

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    metrics_list = []
    for fold in range(5):
        print(f"\nEvaluating Ensemble on Fold {fold} Validation Set")
        _, val_loader, _ = build_data_loaders(
            batch_size=best_config["batch_size"],
            fasta_path=FASTA_PATH,
            label_path=LABEL_PATH,
            tokenizer=tokenizer,
            dataset_class=ResidueLevelDataset,
            fold=fold,
            num_folds=5,
            config=best_config
        )
        acc, precision, recall, f1 = evaluate_ensemble_on_val_loader(val_loader)
        metrics_list.append((acc, precision, recall, f1))

    avg_metrics = np.mean(metrics_list, axis=0)
    print(f"\nAverage Metrics across 5 folds:\nAccuracy: {avg_metrics[0]:.4f}, Precision: {avg_metrics[1]:.4f}, Recall: {avg_metrics[2]:.4f}, F1: {avg_metrics[3]:.4f}")

    with open("checkpoints/best_result.txt", "w", encoding="utf-8") as f:
        if not args.debug:
            f.write(f"Best Validation F1 (Ray Tune): {best_result.metrics['val_f1']:.4f}\n")
        f.write(f"Validation Accuracy (Ensemble): {acc:.4f}\n")
        f.write(f"Validation Precision:            {precision:.4f}\n")
        f.write(f"Validation Recall:               {recall:.4f}\n")
        f.write(f"Validation F1:                   {f1:.4f}\n")
        f.write("Weights path: checkpoints/final_model_fold_*.pth\n")
        f.write("Config path:  checkpoints/best_model_config.json\n")

    checkpoint = torch.load(FINAL_PATH, map_location="cpu", weights_only=False)
    if "history" in checkpoint:
        plot_training_curves(checkpoint["history"])

    generate_submission_csv(
        fasta_path="sequences.fasta",
        test_path="test.tsv",
        output_path="prediction.csv"
    )
