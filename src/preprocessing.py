import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229,0.224,0.225])
    ])

def load_full_dataset(data_dir):
    transform = get_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset

def split_dataset(dataset, train_ratio=0.7):
    total_len = len(dataset)
    train_size = int(train_ratio * total_len)
    remaining = total_len - train_size
    val_size = remaining // 2
    test_size = remaining - val_size  

    train_dataset, temp_dataset = random_split(dataset, [train_size, remaining])
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    print(f"Total: {total_len}, Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


    return train_dataset, val_dataset, test_dataset  
def make_dataloaders(train_ds, val_ds, test_ds, train_bs=128, eval_bs=32):
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False)

    return train_loader, val_loader, test_loader


def make_dataloaders(train_ds, val_ds, test_ds, train_bs= 128, eval_bs= 32):
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=eval_bs, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=eval_bs, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_class_distribution(dataset):
    class_counts = defaultdict(int)
    for _, label in dataset.samples:
        class_counts[label] += 1

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = [idx_to_class[i] for i in class_counts]
    counts = [class_counts[i] for i in class_counts]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

def find_corrupted_images(dataset):
    corrupted = []
    for img_path, _ in dataset.samples:
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            corrupted.append(img_path)

    return corrupted

def show_grid(dataset, idx_to_class, samples_per_class=4):
    classes = list(idx_to_class.values())
    num_classes = len(classes)
    fig, axs = plt.subplots(num_classes, samples_per_class, figsize=(12, 3 * num_classes))

    for cls_idx, cls_name in enumerate(classes):
        imgs = [s[0] for s in dataset.samples if s[1] == cls_idx][:samples_per_class]

        for j, path in enumerate(imgs):
            img = Image.open(path).convert("RGB")
            axs[cls_idx, j].imshow(img)
            axs[cls_idx, j].set_title(cls_name)
            axs[cls_idx, j].axis("off")

    plt.tight_layout()
    plt.show()