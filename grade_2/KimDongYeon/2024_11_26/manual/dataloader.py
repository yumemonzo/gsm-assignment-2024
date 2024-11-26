"""
이 파일은 CIFAR10 데이터셋을 로드하고 데이터 변환(transform)을 적용하며,
훈련 및 테스트 데이터셋을 생성하고 분할하는 기능을 제공합니다.

Author: yumemonzo@gmail.com
Date: 2024-11-18
"""

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, Dataset
from typing import Tuple


def get_transform() -> v2.Compose:
    """
    CIFAR10 데이터셋에 적용할 변환을 정의합니다. 
    이미지를 PIL 형식으로 변환하고 스케일링합니다.

    Returns:
        v2.Compose: 변환 파이프라인 객체
    """
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    return transform


def get_datasets(transform: v2.Compose) -> Tuple[CIFAR10, CIFAR10]:
    """
    CIFAR10 데이터셋을 로드하여 훈련 및 테스트 데이터셋으로 반환합니다.

    Args:
        transform (v2.Compose): 데이터셋에 적용할 변환 파이프라인

    Returns:
        Tuple[CIFAR10, CIFAR10]: 훈련 데이터셋과 테스트 데이터셋
    """
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def split_dataset(dataset: Dataset, split_size: float) -> Tuple[Dataset, Dataset]:
    """
    데이터셋을 훈련 세트와 검증 세트로 분할합니다.

    Args:
        dataset (Dataset): 분할할 데이터셋
        split_size (float): 훈련 세트의 비율 (0.0 ~ 1.0)

    Returns:
        Tuple[Dataset, Dataset]: 훈련 세트와 검증 세트
    """
    size_A = int(split_size * len(dataset))
    size_B = len(dataset) - size_A
    dataset_A, dataset_B = random_split(dataset, [size_A, size_B])
    
    return dataset_A, dataset_B
