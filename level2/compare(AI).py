"""Level 2 对比实验：MLP vs LeNet(CNN)，数据集 Fashion-MNIST。

一次运行完成"准确率 / 参数量 / 收敛速度 / 错误样本"四角度对比。
公平性保证：两个模型共用同一次 get_dataloaders() 的数据划分，
并使用相同的优化器、学习率、epochs、batch size。

产物:
  run/mlp_history.json / run/lenet_history.json   训练历史
  run/mlp_best.pt    / run/lenet_best.pt          权重
  run/compare_convergence.png                     收敛速度对比曲线
  run/compare_errors.png                          错误样本对比图
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_dataloaders, MEAN, STD, CLASSES
from model import LeNet, count_parameters
from train import train_one_epoch, evaluate

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = "run"


class MLP(nn.Module):
    """与 Level 1 相同结构的 MLP，仅用于对照实验。"""

    def __init__(self, input_dim=784, hidden_dims=(256, 128),
                 num_classes=10, dropout=0.2):
        super().__init__()
        layers = [nn.Flatten()]
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, loaders, epochs, tag):
    """训练一个模型，返回历史记录；权重和历史都带 tag 保存。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion,
                                     optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, DEVICE)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{tag}] Epoch {epoch:2d}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(RUN_DIR, f"{tag}_best.pt"))

    with open(os.path.join(RUN_DIR, f"{tag}_history.json"),
              "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history, best_val_acc


@torch.no_grad()
def collect_errors(model, loader, device, max_num=8):
    """收集模型预测错误的样本 (图片, 真实标签, 预测标签)。"""
    model.eval()
    errors = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        mask = pred != labels
        for img, true, p in zip(images[mask], labels[mask], pred[mask]):
            if len(errors) < max_num:
                errors.append((img.cpu(), true.item(), p.item()))
        if len(errors) >= max_num:
            break
    return errors


def plot_convergence(histories, save_path):
    """收敛速度对比：val_acc + val_loss 双图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"MLP": "tab:blue", "LeNet": "tab:red"}
    for tag, history in histories.items():
        epochs = range(1, len(history["val_acc"]) + 1)
        axes[0].plot(epochs, history["val_acc"], "o-",
                     color=colors[tag], label=tag)
        axes[1].plot(epochs, history["val_loss"], "o-",
                     color=colors[tag], label=tag)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val accuracy")
    axes[0].set_title("Convergence speed: val accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val loss")
    axes[1].set_title("Convergence speed: val loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_errors(errors_by_model, save_path):
    """错误样本对比：每行一个模型，共 max_num 列。"""
    tags = list(errors_by_model.keys())
    n_cols = max(len(v) for v in errors_by_model.values())
    fig, axes = plt.subplots(2, n_cols, figsize=(2 * n_cols, 5))
    for row, tag in enumerate(tags):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(errors_by_model[tag]):
                img, true, pred = errors_by_model[tag][col]
                # 反归一化后显示
                img = torch.clamp(img * STD + MEAN, 0, 1).squeeze(0)
                ax.imshow(img, cmap="gray")
                ax.set_title(f"真:{CLASSES[true][:6]}\n预:{CLASSES[pred][:6]}",
                             fontsize=9, color="red")
    fig.suptitle("错误样本对比  (上: MLP   下: LeNet)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"使用设备: {DEVICE}")

    # 关键：只调用一次，两个模型共用完全相同的 train/val/test 划分
    loaders = get_dataloaders(batch_size=BATCH_SIZE)

    models = {
        "mlp": MLP().to(DEVICE),
        "lenet": LeNet().to(DEVICE),
    }
    for tag, model in models.items():
        print(f"{tag} 参数量: {count_parameters(model):,}")

    # 1~3. 训练并记录收敛历史（同一数据、同一超参数）
    histories, results = {}, {}
    for tag, model in models.items():
        history, best_val_acc = train_model(model, loaders, EPOCHS, tag)
        histories[tag] = history
        results[tag] = best_val_acc
        print(f"{tag} 最佳验证准确率: {best_val_acc:.4f}")

    # 4. 测试集准确率 + 错误样本
    errors = {}
    print("\n========== 汇总 ==========")
    for tag, model in models.items():
        test_acc = evaluate(model, loaders["test"],
                            nn.CrossEntropyLoss(), DEVICE)[1]
        errors[tag] = collect_errors(model, loaders["test"], DEVICE)
        print(f"{tag:<6} 参数量 {count_parameters(model):>8,} | "
              f"最佳验证 {results[tag]:.4f} | 测试集 {test_acc:.4f}")

    plot_convergence(histories, os.path.join(RUN_DIR, "compare_convergence.png"))
    plot_errors(errors, os.path.join(RUN_DIR, "compare_errors.png"))
    print("\n产物已保存到 run/: compare_convergence.png, compare_errors.png")


if __name__ == "__main__":
    main()