"""
이 파일은 CIFAR10 데이터셋을 로드하고, 데이터 변환(transform)을 적용하며,
훈련 및 테스트 데이터셋을 생성하는 기능을 제공합니다. 또한, 훈련 데이터를
훈련/검증 세트로 분할하는 기능도 포함되어 있습니다.

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
    CIFAR10 데이터셋에 적용할 데이터 변환(transform) 파이프라인을 정의합니다.

    Returns:
        v2.Compose: 이미지 변환 파이프라인 객체.
    """
    transform = v2.Compose([
        v2.ToImage(),  # 이미지를 PIL.Image 형식으로 변환
        v2.ToDtype(torch.float32, scale=False),  # 데이터를 float32 형식으로 변환
    ])

    return transform


def get_datasets(transform: v2.Compose) -> Tuple[CIFAR10, CIFAR10]:
    """
    CIFAR10 훈련 및 테스트 데이터셋을 생성하고 반환합니다.

    Args:
        transform (v2.Compose): 데이터셋에 적용할 변환 파이프라인.

    Returns:
        Tuple[CIFAR10, CIFAR10]: CIFAR10 훈련 데이터셋과 테스트 데이터셋.
            - train_dataset (CIFAR10): 훈련 데이터셋.
            - test_dataset (CIFAR10): 테스트 데이터셋.
    """
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def split_dataset(dataset: Dataset, split_size: float) -> Tuple[Dataset, Dataset]:
    """
    데이터셋을 지정된 비율로 훈련/검증 세트로 분할합니다.

    Args:
        dataset (Dataset): 분할할 데이터셋.
        split_size (float): 훈련 데이터의 비율 (0.0 ~ 1.0).

    Returns:
        Tuple[Dataset, Dataset]: 분할된 두 데이터셋.
            - dataset_A (Dataset): 훈련 데이터셋.
            - dataset_B (Dataset): 검증 데이터셋.
    """
    size_A = int(split_size * len(dataset))  # 훈련 데이터 크기
    size_B = len(dataset) - size_A  # 검증 데이터 크기

    dataset_A, dataset_B = random_split(dataset, [size_A, size_B])

    return dataset_A, dataset_B
