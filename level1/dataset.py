import torch 
from torch.utils.data import DataLoader, random_split
from torchvision import datasets , transforms 

MEAN, STD = 0.1307, 0.3081

def build_transform():
    #处理数据
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=MEAN,std=STD)])


def get_dataloaders(batch_size=128,val_ratio=1/6,data_dir="data"):
    transform = build_transform()
    train_full = datasets.MNIST(data_dir, train=True, transform= transform, download=True)
    test_set = datasets.MNIST(data_dir, train=False,transform=transform, download=True)

    val_size = int(len(train_full)*val_ratio)
    train_set, val_set = random_split(train_full,[len(train_full) - val_size, val_size])

    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size,shuffle=True,pin_memory=pin),
        "test" : DataLoader(test_set,shuffle=False,batch_size=batch_size,pin_memory=pin),
        "val" : DataLoader(val_set,batch_size=batch_size,shuffle=True,pin_memory=pin),
    }
    return loaders

#为了让训练流程清楚，我写了如下验证函数
if __name__ == "__main__":
    loaders = get_dataloaders()
    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"{name}: {len(loader.dataset)} 张, "
              f"batch 形状: images {tuple(images.shape)}, labels {tuple(labels.shape)}")


