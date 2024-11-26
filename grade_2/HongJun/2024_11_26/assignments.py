import logging
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataloader import get_transform, get_datasets, split_dataset
from model import SimpleCNN
from trainer import Trainer

@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    OmegaConf.to_yaml(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger = logging.getLogger("training")
    logger.setLevel(logging.DEBUG)

    transform = get_transform()

    train_dataset, test_dataset = get_datasets(transform=transform)
    train_dataset, valid_dataset = split_dataset(dataset=train_dataset, split_size=cfg.data.train_ratio)
    logger.info(f"train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)} | test_dataset: {len(test_dataset)}\n")

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}\n")

    model = SimpleCNN().to(device)
    logger.info(f"model: {model}\n")

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, device, save_dir=output_dir)
    trainer.training(num_epochs=cfg.train.num_epochs, logger=logger)

    test_loss, test_acc = trainer.test(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
