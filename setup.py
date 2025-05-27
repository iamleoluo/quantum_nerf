#!/usr/bin/env python3
"""
Quantum NeRF 專案設置腳本

用於安裝和配置專案環境
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml


def print_banner():
    """打印歡迎橫幅"""
    banner = """
    🌌 Quantum NeRF 專案設置
    ========================
    
    歡迎使用 Quantum NeRF！
    這個腳本將幫助您設置完整的開發環境。
    """
    print(banner)


def check_python_version():
    """檢查 Python 版本"""
    print("🐍 檢查 Python 版本...")
    
    if sys.version_info < (3, 8):
        print("❌ 錯誤：需要 Python 3.8 或更高版本")
        print(f"   當前版本：{sys.version}")
        return False
    
    print(f"✅ Python 版本：{sys.version}")
    return True


def install_dependencies():
    """安裝依賴包"""
    print("\n📦 安裝依賴包...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ 依賴包安裝完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依賴包安裝失敗：{e}")
        return False


def create_sample_data():
    """創建示例數據"""
    print("\n📊 創建示例數據...")
    
    data_dir = Path("data/synthetic/lego")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建簡單的配置文件
    sample_config = {
        "scene_name": "demo_scene",
        "image_height": 64,
        "image_width": 64,
        "focal_length": 30.0,
        "n_images": 20
    }
    
    with open(data_dir / "scene_config.yaml", "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print("✅ 示例數據配置已創建")


def setup_git_hooks():
    """設置 Git hooks（如果使用 Git）"""
    print("\n🔧 設置開發工具...")
    
    if Path(".git").exists():
        print("✅ 檢測到 Git 倉庫")
        # 可以在這裡添加 pre-commit hooks 等
    else:
        print("ℹ️  未檢測到 Git 倉庫")


def run_basic_tests():
    """運行基礎測試"""
    print("\n🧪 運行基礎測試...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pytest", "tests/test_basic.py", "-v"
        ])
        print("✅ 基礎測試通過")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  測試失敗：{e}")
        print("   這可能是正常的，如果這是首次設置")
        return False


def print_next_steps():
    """打印後續步驟"""
    next_steps = """
    🎉 設置完成！

    📋 後續步驟：
    
    1. 🚀 運行演示筆記本：
       jupyter notebook notebooks/01_基礎演示.ipynb
    
    2. 🧪 運行測試：
       python -m pytest tests/ -v
    
    3. 🏃 開始訓練：
       python src/training/train.py --config configs/basic_config.yaml
    
    4. 📚 閱讀文檔：
       查看 README.md 和 docs/ 目錄
    
    5. 🌌 探索量子功能：
       查看 src/quantum/ 模組
    
    ❓ 需要幫助？
    - 查看 README.md
    - 運行 python setup.py --help
    - 創建 GitHub Issue
    
    祝您使用愉快！🚀
    """
    print(next_steps)


def main():
    """主設置流程"""
    print_banner()
    
    # 檢查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 安裝依賴
    if not install_dependencies():
        print("⚠️  依賴安裝失敗，但可以繼續設置")
    
    # 創建示例數據
    create_sample_data()
    
    # 設置開發工具
    setup_git_hooks()
    
    # 運行測試
    run_basic_tests()
    
    # 打印後續步驟
    print_next_steps()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
        Quantum NeRF 設置腳本
        
        用法：
            python setup.py        # 完整設置
            python setup.py --help # 顯示幫助
        
        這個腳本會：
        1. 檢查 Python 版本
        2. 安裝依賴包
        3. 創建示例數據
        4. 設置開發工具
        5. 運行基礎測試
        """)
    else:
        main() 