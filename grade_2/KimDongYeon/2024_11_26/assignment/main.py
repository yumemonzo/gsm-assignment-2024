import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import get_transform, get_datasets, split_dataset
from model import SimpleCNN
from trainer import Trainer
from utils import ensure_dir_exists


def main():
    # hyper-parameters
    output_dir = "./outputs"
    train_ratio = 0.8
    batch_size = 64
    lr = 0.0001
    epochs = 30

    ensure_dir_exists(output_dir)
    transform = get_transform()

    train_dataset, test_dataset = get_datasets(transform=transform)
    train_dataset, valid_dataset = split_dataset(dataset=train_dataset, split_size=train_ratio)
    print(f"train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)} | test_dataset: {len(test_dataset)}\n")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    model = SimpleCNN().to(device)
    print(f"model: {model}\n")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, device, save_dir=output_dir)
    trainer.training(num_epochs=epochs)

    test_loss, test_acc = trainer.test(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
