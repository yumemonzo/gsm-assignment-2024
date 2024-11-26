"""
이 파일은 유틸리티 함수들을 모아놓은 파일입니다.

Author: yumemonzo@gmail.com
Date: 2024-11-26
"""

import os


def ensure_dir_exists(directory: str) -> None:
    """
    주어진 디렉토리가 존재하는지 확인하고, 존재하지 않을 경우 생성합니다.

    Args:
        directory (str): 확인 및 생성할 디렉토리 경로.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")
