"""
이 파일은 PyTorch를 사용한 모델 학습을 관리하는 Trainer 클래스를 정의합니다.
Trainer 클래스는 모델 학습, 검증, 테스트, TensorBoard 기록, 모델 저장 기능을 제공합니다.

Author: yumemonzo@gmail.com
Date: 2024-11-18
"""

import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    모델 학습 및 평가를 관리하는 Trainer 클래스입니다.

    Attributes:
        model (torch.nn.Module): 학습할 PyTorch 모델.
        train_loader (torch.utils.data.DataLoader): 학습 데이터 로더.
        valid_loader (torch.utils.data.DataLoader): 검증 데이터 로더.
        criterion (torch.nn.Module): 손실 함수.
        optimizer (torch.optim.Optimizer): 최적화 알고리즘.
        device (torch.device): 모델과 데이터를 사용할 장치 (CPU 또는 GPU).
        save_dir (str): 학습 중 생성된 모델 및 TensorBoard 기록을 저장할 디렉토리.
        lowest_loss (float): 현재까지 검증 데이터에서 기록된 최저 손실 값.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard 기록 도구.
    """

    def __init__(
        self: "Trainer", 
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device, 
        save_dir: str
    ) -> None:
        """
        Trainer 클래스 초기화 메서드입니다.

        Args:
            model (torch.nn.Module): 학습할 PyTorch 모델.
            train_loader (torch.utils.data.DataLoader): 학습 데이터 로더.
            valid_loader (torch.utils.data.DataLoader): 검증 데이터 로더.
            criterion (torch.nn.Module): 손실 함수.
            optimizer (torch.optim.Optimizer): 최적화 알고리즘.
            device (torch.device): 모델과 데이터를 사용할 장치 (CPU 또는 GPU).
            save_dir (str): 학습 중 생성된 모델 및 TensorBoard 기록을 저장할 디렉토리.
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.lowest_loss = float('inf')
        self.writer = SummaryWriter(save_dir)  # TensorBoard SummaryWriter 초기화

    def train(self: "Trainer") -> tuple[float, float]:
        """
        모델을 학습 데이터셋으로 학습하는 메서드입니다.

        Returns:
            tuple[float, float]: 평균 학습 손실과 학습 데이터셋에서의 정확도.
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

    def valid(self: "Trainer") -> tuple[float, float]:
        """
        모델을 검증 데이터셋으로 평가하는 메서드입니다.

        Returns:
            tuple[float, float]: 평균 검증 손실과 검증 데이터셋에서의 정확도.
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

    def test(self: "Trainer", test_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        """
        모델을 테스트 데이터셋으로 평가하는 메서드입니다.

        Args:
            test_loader (torch.utils.data.DataLoader): 테스트 데이터 로더.

        Returns:
            tuple[float, float]: 평균 테스트 손실과 테스트 데이터셋에서의 정확도.
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

    def training(self: "Trainer", num_epochs: int, logger: logging.Logger) -> None:
        """
        모델 학습과 검증 과정을 관리하며 TensorBoard에 기록하는 메서드입니다.

        Args:
            num_epochs (int): 총 학습 에포크 수.
            logger (logging.Logger): 학습 진행 상황을 기록할 로거 객체.
        """
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
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
                logger.info(f"New best model saved with Validation Loss: {valid_loss:.4f}")

        # TensorBoard Writer 닫기
        self.writer.close()
