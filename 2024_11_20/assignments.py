import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import get_transform, get_datasets, split_dataset
from model import SimpleCNN
from trainer import Trainer


def main():
    transform = get_transform()

    train_dataset, test_dataset = get_datasets(transform=transform)
    train_dataset, valid_dataset = split_dataset(dataset=train_dataset, split_size=0.8)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, device, save_path="best_model.pth")
    trainer.training(num_epochs=30)

    test_loss, test_acc = trainer.test(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
