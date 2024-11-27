"""
이 스크립트는 유틸리티 함수들을 모아놓은 스크립트입니다.

Author: yumemonzo@gmail.com
Date: 2024-11-27
"""

import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """
    시드를 고정하여 실험의 재현 가능성을 확보합니다.

    Args:
        seed (int): 고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
