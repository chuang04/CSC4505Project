
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.cuda.amp import GradScaler
from preprocessing import load_full_dataset, split_dataset, make_dataloaders
from model import ConvNextClassifier  
from model_utils import get_all_preds, plot_history, evaluate_model

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_acc, n_batches = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches

def evaluate(model, loader, criterion, device, scaler):
    model.eval()
    total_loss, total_acc, n_batches = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda'):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


data_dir = os.path.join("..", "dataset", "1", "HMU-GC-HE-30K", "all_image")
full_dataset = load_full_dataset(data_dir)

train_ds, val_ds, test_ds = split_dataset(full_dataset)  # <- now returns datasets
train_loader, val_loader, test_loader = make_dataloaders(train_ds, val_ds, test_ds)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvNextClassifier(num_classes=8, pretrained=True, freeze_backbone=True, dropout=0.3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler()


EPOCHS = 10
results = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, scaler)

    results.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

    print(f"Epoch {epoch:02d} | "
          f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"../model/ConvNext_checkpoint_epoch{epoch}.pth")

# Save final model
torch.save(model.state_dict(), "../model/ConvNext_Final.pth")

# Save training history
history = pd.DataFrame(results)
history.to_csv("../model/ConvNext_training_history.csv", index=False)


plot_history(history)

class_names = full_dataset.classes
y_true, y_pred, y_probs = evaluate_model(model, test_loader, class_names)