"""
이 파일은 간단한 CNN 모델(SimpleCNN)을 정의합니다. 이 모델은 CIFAR-10과 같은 작은 이미지 데이터셋을
분류하는 데 사용됩니다. 모델은 2개의 컨볼루션 층, 1개의 최대 풀링 층, 2개의 완전 연결층(Fully Connected Layer)으로 구성되어 있습니다.

Author: yumemonzo@gmail.com
Date: 2024-11-18
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    간단한 CNN 모델을 정의하는 클래스입니다.

    Attributes:
        conv1 (nn.Conv2d): 첫 번째 컨볼루션 층 (입력 채널: 3, 출력 채널: 32).
        conv2 (nn.Conv2d): 두 번째 컨볼루션 층 (입력 채널: 32, 출력 채널: 64).
        pool (nn.MaxPool2d): 2x2 크기의 최대 풀링 층.
        fc1 (nn.Linear): 첫 번째 완전 연결층 (입력 크기: 64*8*8, 출력 크기: 128).
        fc2 (nn.Linear): 두 번째 완전 연결층 (입력 크기: 128, 출력 크기: 10).
        relu (nn.ReLU): ReLU 활성화 함수.
        dropout (nn.Dropout): 드롭아웃 레이어 (드롭 확률: 0.5).
    """

    def __init__(self: "SimpleCNN") -> None:
        """
        SimpleCNN 클래스의 초기화 메서드입니다.
        
        Conv2d, MaxPool2d, Linear, Dropout 등 필요한 층들을 정의합니다.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self: "SimpleCNN", x: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터 x를 모델의 계층을 통해 순전파(forward) 방식으로 처리합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서 (배치 크기, 채널, 높이, 너비).

        Returns:
            output (torch.Tensor: 분류 결과 텐서 (배치 크기, 클래스 수).
        """
        x = self.pool(self.relu(self.conv1(x)))  # 첫 번째 컨볼루션 + ReLU + 풀링
        x = self.pool(self.relu(self.conv2(x)))  # 두 번째 컨볼루션 + ReLU + 풀링
        x = x.view(-1, 64 * 8 * 8)  # 텐서를 1차원 벡터로 변환
        x = self.relu(self.fc1(x))  # 첫 번째 완전 연결층 + ReLU
        x = self.dropout(x)         # 드롭아웃 적용
        output = self.fc2(x)             # 두 번째 완전 연결층

        return output
