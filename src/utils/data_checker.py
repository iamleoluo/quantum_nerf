"""
NeRF 數據集格式檢測工具

這個腳本用於檢查 NeRF 數據集的格式是否正確，
並提供詳細的診斷信息。
"""

import os
from pathlib import Path
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional


class NeRFDataChecker:
    """NeRF 數據集格式檢測器"""
    
    def __init__(self, data_dir: str):
        """
        初始化檢測器
        
        Args:
            data_dir: 數據目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.issues = []
        self.warnings = []
        
    def check_directory_structure(self) -> bool:
        """檢查目錄結構"""
        print("\n📁 檢查目錄結構...")
        
        # 檢查基本目錄
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                self.issues.append(f"缺少必要目錄: {dir_name}")
            elif not any(dir_path.iterdir()):
                self.warnings.append(f"目錄為空: {dir_name}")
        
        return len(self.issues) == 0
    
    def check_transforms_files(self) -> bool:
        """檢查相機參數文件"""
        print("\n📷 檢查相機參數文件...")
        
        required_files = ['transforms_train.json', 'transforms_val.json', 'transforms_test.json']
        for file_name in required_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                self.issues.append(f"缺少相機參數文件: {file_name}")
            else:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if 'frames' not in data:
                        self.issues.append(f"{file_name} 格式錯誤: 缺少 'frames' 字段")
                except json.JSONDecodeError:
                    self.issues.append(f"{file_name} 不是有效的 JSON 文件")
        
        return len(self.issues) == 0
    
    def check_images(self) -> Tuple[Optional[Tuple[int, int]], bool]:
        """檢查圖像文件"""
        print("\n🖼️ 檢查圖像文件...")
        
        image_sizes = set()
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
                
            image_files = list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpg'))
            if not image_files:
                self.warnings.append(f"{split} 目錄中沒有圖像文件")
                continue
                
            for img_file in image_files:
                try:
                    with Image.open(img_file) as img:
                        image_sizes.add(img.size)
                except Exception as e:
                    self.issues.append(f"無法讀取圖像 {img_file}: {str(e)}")
        
        if len(image_sizes) > 1:
            self.warnings.append(f"發現多個圖像尺寸: {image_sizes}")
        
        return (image_sizes.pop() if image_sizes else None), len(self.issues) == 0
    
    def check_camera_parameters(self) -> bool:
        """檢查相機參數格式"""
        print("\n🎥 檢查相機參數格式...")
        
        for file_name in ['transforms_train.json', 'transforms_val.json', 'transforms_test.json']:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'camera_angle_x' not in data:
                    self.warnings.append(f"{file_name} 缺少 'camera_angle_x' 參數")
                
                for frame in data.get('frames', []):
                    if 'transform_matrix' not in frame:
                        self.issues.append(f"{file_name} 中的幀缺少 'transform_matrix'")
                    else:
                        matrix = np.array(frame['transform_matrix'])
                        if matrix.shape != (4, 4):
                            self.issues.append(f"{file_name} 中的變換矩陣形狀錯誤: {matrix.shape}")
            except Exception as e:
                self.issues.append(f"處理 {file_name} 時出錯: {str(e)}")
        
        return len(self.issues) == 0
    
    def run_all_checks(self) -> Dict:
        """運行所有檢查"""
        print(f"\n🔍 開始檢查數據集: {self.data_dir}")
        
        dir_ok = self.check_directory_structure()
        transforms_ok = self.check_transforms_files()
        image_size, images_ok = self.check_images()
        camera_ok = self.check_camera_parameters()
        
        # 生成報告
        report = {
            "數據集路徑": str(self.data_dir),
            "目錄結構檢查": "通過" if dir_ok else "失敗",
            "相機參數文件檢查": "通過" if transforms_ok else "失敗",
            "圖像文件檢查": "通過" if images_ok else "失敗",
            "相機參數格式檢查": "通過" if camera_ok else "失敗",
            "圖像尺寸": str(image_size) if image_size else "未知",
            "問題": self.issues,
            "警告": self.warnings
        }
        
        # 打印報告
        print("\n📊 檢查報告:")
        print("-" * 50)
        for key, value in report.items():
            if key in ["問題", "警告"]:
                print(f"\n{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        print("-" * 50)
        
        return report


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeRF 數據集格式檢測工具")
    parser.add_argument("data_dir", help="數據集目錄路徑")
    args = parser.parse_args()
    
    checker = NeRFDataChecker(args.data_dir)
    checker.run_all_checks()


if __name__ == "__main__":
    main() 