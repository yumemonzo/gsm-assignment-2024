"""
이 파일은 CIFAR-10 데이터셋을 사용하여 간단한 CNN 모델을 학습, 검증, 테스트하는 작업을 수행합니다.
데이터셋 준비, 모델 구성, 학습/평가 관리를 포함한 전 과정을 메인 함수에서 처리합니다.

구체적인 기능:
1. CIFAR-10 데이터셋 로드 및 변환
2. 학습, 검증, 테스트 데이터셋 분할 및 로더(DataLoader) 생성
3. CNN(SimpleCNN) 모델 구성
4. 모델 학습(Trainer 클래스 사용)
5. 테스트 데이터셋 평가 및 결과 출력

Author: yumemonzo@gmail.com
Date: 2024-11-26
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import get_transform, get_datasets, split_dataset
from model import SimpleCNN
from trainer import Trainer
from utils import ensure_dir_exists


def main():
    """
    모델 학습 및 평가를 관리하는 메인 함수.

    1. 데이터셋을 준비하고, 학습, 검증, 테스트 데이터셋으로 분할합니다.
    2. 모델, 손실 함수, 옵티마이저를 정의합니다.
    3. Trainer 클래스를 사용하여 모델을 학습, 검증, 평가합니다.
    4. 최종적으로 테스트 데이터셋에 대한 손실 및 정확도를 출력합니다.
    """

    # 하이퍼파라미터 설정
    output_dir = "./outputs"  # 출력 결과 저장 경로
    train_ratio = 0.8  # 학습 데이터 비율
    batch_size = 64  # 배치 크기
    lr = 0.0001  # 학습률
    epochs = 30  # 학습 에포크 수

    # 출력 디렉토리 확인 및 생성
    ensure_dir_exists(output_dir)

    # 데이터 변환 정의
    transform = get_transform()

    # CIFAR10 데이터셋 로드 및 분할
    train_dataset, test_dataset = get_datasets(transform=transform)
    train_dataset, valid_dataset = split_dataset(dataset=train_dataset, split_size=train_ratio)
    print(f"train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)} | test_dataset: {len(test_dataset)}\n")

    # 데이터 로더 생성
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 학습 장치 설정 (GPU 또는 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # 모델 초기화 및 장치로 이동
    model = SimpleCNN().to(device)
    print(f"model: {model}\n")

    # 옵티마이저 및 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Trainer 클래스 초기화 및 학습 시작
    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, device, save_dir=output_dir)
    trainer.training(num_epochs=epochs)

    # 테스트 데이터셋 평가
    test_loss, test_acc = trainer.test(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
