import os 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import MLP,count_parameters

#hyper parameters:
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_DIMS = (256,128)
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = "run"

#python 修饰器：这整个函数都是梯度关闭
@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval()
    total_loss , correct, total = 0.0, 0, 0
    for images,labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs,labels).item()*labels.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
    return total_loss/total, correct/total #返回平均loss和正确率

#AI
def plot_curves(history, save_path):
    """画两张图: Loss 曲线 + 验证集准确率曲线, 保存到文件。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "o-", label="train loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], "o-", label="val accuracy")
    axes[1].axhline(0.90, color="r", ls="--", label="target 90%")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Validation accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

#训练
def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    running_loss = 0.
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*labels.size(0)

    return running_loss/ len(loader.dataset)

def train(model, loaders, criterion, optimizer, device, epochs, run_dir="run"):
    history={"train_loss":[],"val_loss":[],"val_acc":[]}
    best_val_acc = 0.

    for epoch in range(1,epochs+1):
        train_loss = train_one_epoch(model,loaders["train"],criterion,optimizer,device)
        val_loss, val_acc = evaluate(model,loaders,criterion,device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),os.path.join(run_dir,"best.pt"))

    torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))
    plot_curves(history, os.path.join(run_dir, "loss_curve.png"))
    return history, best_val_acc

def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"使用设备: {DEVICE}")

    loaders = get_dataloaders(batch_size=BATCH_SIZE)
    model = MLP(hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    _, best_val_acc = train(model, loaders, criterion, optimizer,
                            DEVICE, EPOCHS, RUN_DIR)
    print(f"训练完成! 最佳验证集准确率: {best_val_acc:.4f}")
    print(f"产物已保存到: {RUN_DIR}/")


if __name__ == "__main__":
    main()
