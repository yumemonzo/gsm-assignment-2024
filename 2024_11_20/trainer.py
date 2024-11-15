import torch


class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, device, save_path):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.lowest_loss = float('inf')

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

    def training(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.valid()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

            if valid_loss < self.lowest_loss:
                self.lowest_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved with Validation Loss: {valid_loss:.4f}")
