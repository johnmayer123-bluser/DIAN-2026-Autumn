
import argparse
import os
import torch
from PIL import Image
from torchvision import transforms

from dataset import MEAN, STD
from model import MLP

RUN_DIR = "run"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, image_path, device):
    img = Image.open(image_path).convert("L")   # 转灰度
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    x = transform(img).unsqueeze(0).to(device)  
    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)       
    label = probs.argmax(dim=1).item()
    return label, probs[0, label].item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="图片路径")
    args = parser.parse_args()

    model = MLP().to(DEVICE)
    ckpt = os.path.join(RUN_DIR, "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    label, conf = predict(model, args.image, DEVICE)
    print(f"预测结果: 数字 {label} (置信度 {conf:.4f})")

if __name__ == "__main__":
    main()