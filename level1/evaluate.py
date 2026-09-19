import os
import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import MLP
RUN_DIR = "run"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total

def main():
    print(f"使用设备: {DEVICE}")

    loaders = get_dataloaders(batch_size=256)
    model = MLP().to(DEVICE)
    ckpt = os.path.join(RUN_DIR, "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"已加载权重: {ckpt}")

    test_acc = evaluate_test(model, loaders["test"], DEVICE)
    print(f"测试集准确率: {test_acc:.4f}")
    print("验收标准: >= 90%  ->  " +
          ("通过" if test_acc >= 0.90 else "未达标"))

if __name__ == "__main__":
    main()
