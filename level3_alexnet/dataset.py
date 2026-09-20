import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

MEAN, STD = 0.2860, 0.3530

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

IMG_SIZE = 224   


def build_transform(train=False):
    """AlexNet 需要 224×224 输入；数据增强只在训练集生效。"""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomCrop(IMG_SIZE, padding=4),   
            transforms.RandomHorizontalFlip(),            
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_dataloaders(batch_size=128, val_ratio=1/6, data_dir="data"):
    train_full = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=build_transform(train=True))
    val_full = datasets.FashionMNIST(data_dir, train=True, download=False,
                                     transform=build_transform(train=False))
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True,
                                     transform=build_transform(train=False))

    val_size = int(len(train_full) * val_ratio)
    train_set, _ = random_split(
        train_full, [len(train_full) - val_size, val_size])
    _, val_set = random_split(
        val_full, [len(val_full) - val_size, val_size])

    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            pin_memory=pin),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          pin_memory=pin),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False,
                           pin_memory=pin),
    }
    return loaders