import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.s_20251129_noise_pred_vs_data_pred.model import SimpleJiT
from src.s_20251129_noise_pred_vs_data_pred.utils import save_images, save_loss


# ==========================================
# 実験設定
# ==========================================
class Config:
    """
    PRED_MODE: 'x', 'eps', 'v' のいずれか
    - 'x': データ予測 (論文推奨)
    - 'eps': ノイズ予測 (従来手法)
    - 'v': 速度予測 (Flow Matching標準)

    PATCH_SIZE: 4 (低次元) または 32 (高次元)
    - 4: 従来のViT設定。どのモードでも学習可能。
    - 32: 画像全体を1パッチとする設定。'eps'は崩壊し、'x'は成功するはず。
    """

    PRED_MODE = "eps"
    PATCH_SIZE = 32

    # 学習ハイパーパラメータ
    IMG_SIZE = 32
    BATCH_SIZE = 128
    LR = 3e-4
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # モデルサイズ
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 4

    # ディレクトリ
    repo_path = Path(__file__).parents[3]
    relative_path = Path(__file__).parent.relative_to(repo_path)

    DATA_DIR = repo_path / "data"
    OUTPUT_DIR = repo_path / "output" / Path(*relative_path.parts[1:])


# ==========================================
# ユーティリティ関数
# ==========================================
def get_dataloader(config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]に正規化
        ]
    )
    dataset = datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )


# ==========================================
# 学習とサンプリング
# ==========================================
def run():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"=== Experiment Start ===")
    print(f"Mode: {config.PRED_MODE}-prediction")
    print(f"Patch Size: {config.PATCH_SIZE} (Input Dim: {config.PATCH_SIZE**2 * 3})")
    print(f"Device: {config.DEVICE}")

    dataloader = get_dataloader(config)
    model = SimpleJiT(config).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)

    losses = []

    # --- Training Loop ---
    model.train()
    for epoch in range(config.EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        epoch_loss = 0

        for x, _ in pbar:
            x = x.to(config.DEVICE)  # Ground Truth Data (x_1)
            batch_size = x.shape[0]

            # --- Flow Matching Logic ---
            # t ~ [0, 1], z_0 ~ N(0, I)
            # 論文 Eq(1): z_t = t * x + (1-t) * eps
            # t=1: Data, t=0: Noise

            t = torch.rand(batch_size, device=config.DEVICE)
            epsilon = torch.randn_like(x)

            # tを (N, 1, 1, 1) にブロードキャスト
            t_expand = t.view(-1, 1, 1, 1)

            # 補間 (Noisy Input)
            z_t = t_expand * x + (1 - t_expand) * epsilon

            # モデル予測
            pred = model(z_t, t)

            # ターゲットの計算
            if config.PRED_MODE == "x":
                target = x
            elif config.PRED_MODE == "eps":
                target = epsilon
            elif config.PRED_MODE == "v":
                # v = dx/dt = x - eps
                target = x - epsilon
            else:
                raise ValueError(f"Unknown mode: {config.PRED_MODE}")

            loss = nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            losses.append(loss.item())

        # 定期的にロスをプロット
        save_loss(losses, config)

        # --- Sampling (Evaluation) ---
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Euler Solver for ODE
                # z_0 (Noise) -> z_1 (Data)
                # dz/dt = v

                num_steps = 50
                dt = 1.0 / num_steps

                # 初期ノイズ
                z = torch.randn(
                    16, 3, config.IMG_SIZE, config.IMG_SIZE, device=config.DEVICE
                )

                for i in range(num_steps):
                    # 現在の時刻 t (0 -> 1)
                    t_val = i / num_steps
                    t_batch = torch.full((16,), t_val, device=config.DEVICE)

                    # 予測
                    model_output = model(z, t_batch)

                    # 予測値を速度 v に変換
                    if config.PRED_MODE == "v":
                        v_pred = model_output
                    elif config.PRED_MODE == "x":
                        # x_pred = model_output
                        # v = (x - z_t) / (1-t)
                        t_denom = max(1 - t_val, 1e-5)
                        v_pred = (model_output - z) / t_denom
                    elif config.PRED_MODE == "eps":
                        # eps_pred = model_output
                        # v = (z_t - eps) / t
                        t_denom = max(t_val, 1e-5)
                        v_pred = (z - model_output) / t_denom

                    # Euler Update
                    z = z + v_pred * dt

                save_images(z, epoch + 1, config)
            model.train()

    print("Experiment Finished.")


if __name__ == "__main__":
    run()
