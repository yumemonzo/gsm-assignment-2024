import os
import torch
from .utils import get_device
from .model.model import SimpleCNN


def load_model(model_path: str = None) -> torch.nn.Module:
    """
    모델을 로드하여 반환합니다.
    Args:
        model_path (str): 모델 가중치 파일 경로.

    Returns:
        torch.nn.Module: 로드된 모델.
    """
    if model_path is None:
        # 현재 파일 위치를 기준으로 model_path 설정
        model_path = os.path.join(os.path.dirname(__file__), "model/best_model.pth")

    device = get_device()
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model
