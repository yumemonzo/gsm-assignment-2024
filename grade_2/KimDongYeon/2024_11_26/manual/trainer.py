"""
이 파일은 PyTorch를 사용하여 모델 학습 및 평가를 관리하는 Trainer 클래스를 정의합니다.
Trainer 클래스는 모델 학습, 검증, 테스트, 모델 저장 기능을 제공합니다.

Author: yumemonzo@gmail.com
Date: 2024-11-26
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer


class Trainer:
    """
    PyTorch 모델 학습 및 평가를 관리하는 Trainer 클래스.

    Attributes:
        model (Module): 학습할 PyTorch 모델.
        train_loader (DataLoader): 학습 데이터 로더.
        valid_loader (DataLoader): 검증 데이터 로더.
        criterion (Module): 손실 함수.
        optimizer (Optimizer): 최적화 알고리즘.
        device (torch.device): 사용 장치 (CPU 또는 GPU).
        save_dir (str): 모델 저장 디렉토리.
        lowest_loss (float): 현재까지 검증 데이터에서 기록된 최저 손실 값.
    """

    def __init__(self, model: Module, train_loader: DataLoader, valid_loader: DataLoader,
                 criterion: Module, optimizer: Optimizer, device: torch.device, save_dir: str) -> None:
        """
        Trainer 클래스 초기화 메서드.

        Args:
            model (Module): 학습할 PyTorch 모델.
            train_loader (DataLoader): 학습 데이터 로더.
            valid_loader (DataLoader): 검증 데이터 로더.
            criterion (Module): 손실 함수.
            optimizer (Optimizer): 최적화 알고리즘.
            device (torch.device): 사용 장치 (CPU 또는 GPU).
            save_dir (str): 모델 저장 디렉토리.
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.lowest_loss = float('inf')

    def train(self) -> tuple[float, float]:
        """
        학습 데이터셋을 사용하여 모델을 학습.

        Returns:
            tuple[float, float]: 평균 학습 손실과 학습 정확도.
        """
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

    def valid(self) -> tuple[float, float]:
        """
        검증 데이터셋을 사용하여 모델을 평가.

        Returns:
            tuple[float, float]: 평균 검증 손실과 검증 정확도.
        """
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

    def test(self, test_loader: DataLoader) -> tuple[float, float]:
        """
        테스트 데이터셋을 사용하여 모델을 평가.

        Args:
            test_loader (DataLoader): 테스트 데이터 로더.

        Returns:
            tuple[float, float]: 평균 테스트 손실과 테스트 정확도.
        """
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

    def training(self, num_epochs: int) -> None:
        """
        학습 및 검증 과정을 관리하고 최적의 모델을 저장.

        Args:
            num_epochs (int): 총 학습 에포크 수.
        """
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.valid()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

            if valid_loss < self.lowest_loss:
                self.lowest_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
                print(f"New best model saved with Validation Loss: {valid_loss:.4f}")
