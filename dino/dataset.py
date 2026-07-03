import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from datasets import load_dataset

# per-channel CIFAR-100 statistics
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
IMG_SIZE = 32
KERNEL_SIZE = 3

def global_transform1(s=1.0):
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=s*0.4, contrast=s*0.4, saturation=s*0.4, hue=s*0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(KERNEL_SIZE, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

def global_transform2(s=1.0):
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=s*0.4, contrast=s*0.4, saturation=s*0.4, hue=s*0.1),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            T.GaussianBlur(KERNEL_SIZE, sigma=(0.1, 2.0))
        ], p=0.1),
        T.ToTensor(),
        T.RandomSolarize(threshold=0.6, p=0.2), # Работает с тензором [0, 1]
        T.Normalize(MEAN, STD)
    ])

def local_transform(s=1.0):
    return T.Compose([
        # Делаем кроп маленькой зоны (например, от 10% до 40% площади), но возвращаем к размеру 32x32
        T.RandomResizedCrop(IMG_SIZE, scale=(0.1, 0.4)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=s*0.4, contrast=s*0.4, saturation=s*0.4, hue=s*0.1),
        T.RandomApply([
            T.GaussianBlur(KERNEL_SIZE, sigma=(0.1, 2.0)),
            T.RandomGrayscale(p=0.2),
        ], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

def eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])


class _HFImagesTrain(Dataset):
    def __init__(self, hf_split, num_crops=4):
        self.hf = hf_split
        self.num_local_crops = num_crops - 2 # Если всего 4, то 2 глобальных и 2 локальных
        
        self.g1 = global_transform1()
        self.g2 = global_transform2()
        self.l  = local_transform()

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        img = self.hf[int(i)]['img']
        
        # Генерируем локальные кропы
        local_crops = [self.l(img) for _ in range(self.num_local_crops)]
        
        # Возвращаем строго по сигнатуре твоей тренировочной петли: x_g1, x_g2, x_locals
        return self.g1(img), self.g2(img), local_crops


class _HFImagesEval(Dataset):
    """Датасет для kNN валидации: возвращает чистую картинку и её метку (label)"""
    def __init__(self, hf_split):
        self.hf = hf_split
        self.transform = eval_transform()

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        item = self.hf[int(i)]
        img = self.transform(item['img'])
        label = item['fine_label'] # В датасете uoft-cs/cifar100 таргет лежит в fine_label
        return img, label


def make_dataloaders(num_workers=8, batch_size=512, num_crops=4):
    # Загружаем датасет из Hugging Face
    ds = load_dataset("uoft-cs/cifar100")
    
    train_hf = _HFImagesTrain(ds["train"], num_crops=num_crops)
    memory_hf = _HFImagesEval(ds["train"])
    test_hf = _HFImagesEval(ds["test"])

    # shuffle=True для трейна, shuffle=False для валидации (для корректной kNN оценки)
    train_loader = DataLoader(train_hf, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    memory_loader = DataLoader(memory_hf, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_hf, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    return train_loader, memory_loader, test_loader


if __name__ == "__main__":
    # Быстрый тест корректности размерностей батча
    train_loader, test_loader = make_dataloaders(num_workers=2, batch_size=4, num_crops=4)
    batch = next(iter(train_loader))
    global_view_1, global_view_2, local_views = batch 
    
    print("Global 1 shape:", global_view_1.shape)
    print("Global 2 shape:", global_view_2.shape)
    print("Local crops count:", len(local_views))
    print("Local crop 1 shape:", local_views[0].shape)
    
    eval_batch = next(iter(test_loader))
    print("Eval img shape:", eval_batch[0].shape, "Eval labels shape:", eval_batch[1].shape)