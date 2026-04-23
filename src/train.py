"""
Boilerplate training script for machine learning experiments.

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

# Optional imports (uncomment if needed)
# import torch
# from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def build_model() -> Any:
    """Create and return the model."""
    # model = YourModel()
    model = None
    return model


def build_dataloader(batch_size: int):
    """Create and return the dataloader."""
    # dataset = YourDataset()
    # return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return None


def train_one_epoch(model, dataloader, epoch: int) -> Dict[str, float]:
    """Run one training epoch."""
    # model.train()
    # for batch in dataloader:
    #     ...
    metrics = {"loss": 0.0}
    print(f"Epoch {epoch}: {metrics}")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    model = build_model()
    dataloader = build_dataloader(cfg.batch_size)

    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, dataloader, epoch)

    print("Training complete")


if __name__ == "__main__":
    main()
