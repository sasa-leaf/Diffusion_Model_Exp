from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.utils import save_image


def save_loss(losses, config):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label=f"Loss ({config.PRED_MODE}-pred)")
    plt.title(f"Training Loss - Mode: {config.PRED_MODE}, Patch: {config.PATCH_SIZE}")
    plt.xlabel("Steps")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        Path(config.OUTPUT_DIR)
        / f"{config.PRED_MODE}_{config.PATCH_SIZE}_loss_curve.png"
    )
    plt.close()


def save_images(images, epoch, config, nrow=8):
    # [-1, 1] -> [0, 1]
    save_path = (
        Path(config.OUTPUT_DIR)
        / f"{config.PRED_MODE}_{config.PATCH_SIZE}_sample_epoch_{epoch:03d}.png"
    )
    save_image(images, save_path, nrow=nrow, normalize=True, value_range=(-1, 1))
