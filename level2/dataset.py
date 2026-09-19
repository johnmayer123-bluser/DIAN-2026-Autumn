import torch 
from torch.utils.data import DataLoader ,random_split
from torchvision import transforms, datasets

MEAN , STD = 0.2860, 0.3530

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def build_transform():
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def get_dataloaders(batch_size=128, val_ratio=1/6, data_dir="data"):
    transform = build_transform()
    train_full = datasets.FashionMNIST(data_dir,train=True,download=True,transform=transform)
    test_set = datasets.FashionMNIST(data_dir,train=False,download=True,transform=transform)

    val_size = int(len(train_full)*val_ratio)
    train_set, val_set = random_split(train_full,[len(train_full) - val_size, val_size])
    
    pin = torch.cuda.is_available()
    loaders = {
            "train": DataLoader(train_set, batch_size=batch_size,shuffle=True,pin_memory=pin),
            "test" : DataLoader(test_set,shuffle=False,batch_size=batch_size,pin_memory=pin),
            "val" : DataLoader(val_set,batch_size=batch_size,shuffle=True,pin_memory=pin),
        }
    return loaders
    
    