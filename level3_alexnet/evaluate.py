
import os
import torch
import torch.nn as nn

from dataset import get_dataloaders, CLASSES
from model import AlexNet

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


@torch.no_grad()
def per_class_accuracy(model, loader, device):
    """返回每个类别的准确率列表，用于定位模型弱点。"""
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        for label, p in zip(labels.tolist(), pred.tolist()):
            class_total[label] += 1
            class_correct[label] += p == label
    return [c / t if t else 0.0 for c, t in zip(class_correct, class_total)]


def main():
    print(f"使用设备: {DEVICE}")

    loaders = get_dataloaders(batch_size=256)
    model = AlexNet().to(DEVICE)
    ckpt = os.path.join(RUN_DIR, "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"已加载权重: {ckpt}")

    test_acc = evaluate_test(model, loaders["test"], DEVICE)
    print(f"测试集准确率: {test_acc:.4f}")

    print("\n各类别准确率:")
    for name, acc in zip(CLASSES,
                         per_class_accuracy(model, loaders["test"], DEVICE)):
        bar = "#" * int(acc * 20)
        print(f"  {name:<13} {acc*100:5.1f}%  {bar}")

    print("\n验收标准: >= 90%  ->  " +
          ("通过" if test_acc >= 0.90 else "未达标"))


if __name__ == "__main__":
    main()