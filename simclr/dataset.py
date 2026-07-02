"""
SimCLR data pipeline for CIFAR-100 @ 32x32.   ***TASK: fill in the TODOs.***

Two things live here:
  * the stochastic augmentation t ~ T that turns one image into a random view, and a
    `TwoCropTransform` that applies it twice to make a positive pair (x_i, x_j);
  * dataloaders: a contrastive loader (pairs, for pretraining) and two eval loaders
    (a memory/train bank and a test set, both lightly transformed) for kNN.

Augmentation policy to implement (SimCLR Sec 3 / Appendix A, adapted to 32x32):
  random resized crop + horizontal flip  ->  color jitter (p=0.8)  ->  grayscale (p=0.2).
Gaussian blur is dropped -- it barely helps on 32x32 images.

CIFAR-100 is read from the (already cached) HuggingFace `uoft-cs/cifar100`.
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from datasets import load_dataset

# per-channel CIFAR-100 statistics
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
IMG_SIZE = 32


def simclr_transform(size=IMG_SIZE, s=0.5):

    transforms = T.Compose([
        T.RandomResizedCrop(size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2* s)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
    return transforms
    


def eval_transform(size=IMG_SIZE):
    """Deterministic transform for kNN (no random augmentation): ToTensor + Normalize."""

    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class TwoCropTransform:
    """Apply the SimCLR augmentation twice -> a positive pair (view1, view2)."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)

        return x1, x2

class _HFImages(Dataset):
    """Wrap a HuggingFace CIFAR-100 split; apply `transform` to the PIL image."""

    def __init__(self, hf_split, transform):
        self.hf = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        ex = self.hf[int(i)]
        return self.transform(ex["img"]), ex["fine_label"]


def make_dataloaders(batch_size, num_workers=8, s=0.5):
    ds = load_dataset("uoft-cs/cifar100")
    train, test = ds['train'], ds['test']

    simclr_transforms = simclr_transform(s=s)
    train_transform = TwoCropTransform(simclr_transforms)
    
    train_ds = _HFImages(train, train_transform)
    contrastive_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    memory_ds = _HFImages(train, transform=eval_transform())
    memory_loader = DataLoader(memory_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    test_ds = _HFImages(test, transform=eval_transform())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return contrastive_loader, memory_loader, test_loader
   