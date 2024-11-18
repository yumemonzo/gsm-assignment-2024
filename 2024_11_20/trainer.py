import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, device, save_path, log_dir="runs"):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.lowest_loss = float('inf')
        self.writer = SummaryWriter(log_dir)  # TensorBoard SummaryWriter 초기화

    def train(self):
        self.model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        accuracy = correct / len(self.train_loader.dataset)
        return total_loss / len(self.train_loader), accuracy

    def valid(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
        accuracy = correct / len(self.valid_loader.dataset)
        return total_loss / len(self.valid_loader), accuracy
    
    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
        accuracy = correct / len(test_loader.dataset)
        return total_loss / len(test_loader), accuracy

    def training(self, num_epochs, logger):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.valid()

            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            logger.info(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

            # TensorBoard 기록: 동일한 그래프에 기록
            self.writer.add_scalars('Loss', {'Train': train_loss, 'Valid': valid_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'Train': train_acc, 'Valid': valid_acc}, epoch)

            if valid_loss < self.lowest_loss:
                self.lowest_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"New best model saved with Validation Loss: {valid_loss:.4f}")

        # TensorBoard Writer 닫기
        self.writer.close()
