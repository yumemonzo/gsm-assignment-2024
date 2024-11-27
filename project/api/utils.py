import torch
from torchvision.transforms import v2


def get_device() -> torch.device:
    """
    사용 가능한 디바이스를 반환합니다 (CUDA 또는 CPU).

    Returns:
        torch.device: 사용 가능한 디바이스.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image) -> torch.Tensor:
    """
    입력 이미지를 모델에 적합한 텐서로 변환합니다.
    
    Args:
        image (PIL.Image): 입력 이미지.

    Returns:
        torch.Tensor: 모델 입력용 텐서.
    """
    transform = v2.Compose([
        v2.ToImage(),  # 이미지를 PIL.Image 형식으로 변환
        v2.Resize((32, 32)),  # 이미지를 32x32로 리사이즈
        v2.ToDtype(torch.float32, scale=True),  # 데이터를 float32 형식으로 변환 및 스케일링
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
    ])

    tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
    return tensor.to(get_device())


def decode_prediction(outputs: torch.Tensor) -> dict:
    """
    모델 출력 텐서를 해석 가능한 결과로 변환합니다.

    Args:
        outputs (torch.Tensor): 모델 출력.

    Returns:
        dict: 클래스와 신뢰도 점수.
    """
    class_labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_class = probabilities.max(1)
    return {
        "class": class_labels[predicted_class.item()],
        "confidence": confidence.item()
    }
