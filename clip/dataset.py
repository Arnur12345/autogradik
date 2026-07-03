"""
Data pipeline for CLIP-with-class-embeddings on CIFAR-100 @ 32x32.
                                                        ***TASK: fill in the TODOs.***

CLIP trains on (image, label) pairs -- one augmented view per image. The label plays the
role of the "text": the class encoder turns it into an embedding the image is matched
against. So the train loader yields (image, label) with augmentation, test yields none.

Batch composition -- two regimes:
  * REQUIRED (one_per_class=False): plain shuffling; a class may appear several times in a
    batch. The CLIP loss handles that with same-class (multi-positive) targets.
  * OPTIONAL (one_per_class=True): implement `UniqueClassBatchSampler` so every batch holds
    DISTINCT classes (batch_size <= num_classes, e.g. 64). Then there are no same-class
    collisions and the loss reduces to the standard diagonal CLIP target.

CIFAR-100 is read from the (already cached) HuggingFace `uoft-cs/cifar100`.
"""
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from datasets import load_dataset
import torch

# per-channel CIFAR-100 statistics
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
IMG_SIZE = 32


def train_transform(s=1.0):
    """Standard CIFAR augmentation.
    
    TODO: 
    """
    transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomApply(T.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    return transforms


def eval_transform():
    """Deterministic transform for evaluation: normalize only."""
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


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


class UniqueClassBatchSampler(Sampler):
    """OPTIONAL TASK. Yield batches of `batch_size` indices whose labels are all DISTINCT.

    With one example per class per batch, the CLIP similarity matrix has no same-class
    collisions and the loss becomes the plain diagonal target. Requires batch_size <= #classes.
    """

    def __init__(self, labels, batch_size, seed=0):
        self.labels = list(labels)
        self.batch_size = batch_size
        self.seed = seed
        # group sample indices by class
        self.by_class = {}
        for idx, c in enumerate(self.labels):
            self.by_class.setdefault(c, []).append(idx)
        assert batch_size <= len(self.by_class), "batch_size must be <= number of classes"

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        shuffled_by_class = {}
        for c, idx in self.by_class.items():
            idx_tensor = torch.tensor(idx)
            shuffled_indices = idx_tensor[torch.randperm(len(idx), generator=g)].tolist()
            shuffled_by_class[c] = shuffled_indices

        while True:
            remaining_classes = [c for c, idx in shuffled_by_class.items() if len(idx) > 0]
            if len(remaining_classes) < self.batch_size:
                break

            chosen_idx = torch.randperm(len(remaining_classes), generator=g)[:self.batch_size]
            chosen_classes = [remaining_classes[i] for i in chosen_idx]

            batch = []
            for c in chosen_classes:
                batch.append(shuffled_by_class[c].pop())

            yield batch
    

    def __len__(self):
        return len(self.labels) // self.batch_size


def make_dataloaders(batch_size, num_workers=8, one_per_class=False):
    """Return (train_loader, test_loader), each yielding (image, label).

    TODO:
      * ds = load_dataset("uoft-cs/cifar100"); train/test = ds["train"], ds["test"]
      * wrap with _HFImages: train uses train_transform(), test uses eval_transform()
      * test_loader: shuffle=False.
      * train_loader:
          - if one_per_class: DataLoader(train, batch_sampler=UniqueClassBatchSampler(
                train_hf["fine_label"], batch_size), ...)          # OPTIONAL path
          - else:             DataLoader(train, batch_size=batch_size, shuffle=True,
                drop_last=True, ...)
        (pin_memory=True, num_workers as given.)
    """
    ds = load_dataset("uoft-cs/cifar100")
    train, test = ds['train'], ds['test']

    train_ds = _HFImages(train, train_transform())
    if one_per_class:
        train_loader = DataLoader(train, batch_sampler=UniqueClassBatchSampler(train_ds['fine_label', batch_size]), shuffle=True)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    test_ds = _HFImages(test, eval_transform())
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader

