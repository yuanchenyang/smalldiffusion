import torch
import csv
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N):
        t = tmin + torch.linspace(0, 1, N) * tmax
        self.vals = torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]

class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, delimiter='\t', scale=50, offset=50):
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]

class ColSelectDataloader:
    def __init__(self, loader, col_name):
        self.loader = loader
        self.col_name = col_name
        self._iter_loader = None
    def __iter__(self):
        self._iter_loader = iter(self.loader)
        return self
    def __next__(self):
        return next(self._iter_loader)[self.col_name]

def get_hf_dataloader(dataset_name='fashion_mnist', split='train', batch_size=128, train_transforms=[]):
    augs = transforms.Compose(train_transforms + [
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    def transform(examples):
        examples['pixel_values'] = [augs(image.convert('L')) for image in examples['image']]
        del examples['image']
        return examples

    dataset = load_dataset(dataset_name, split=split)
    transformed_dataset = dataset.with_transform(transform)
    loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
    return ColSelectDataloader(loader, 'pixel_values')
