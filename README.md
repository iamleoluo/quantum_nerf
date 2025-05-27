# Quantum NeRF 專案 🌌

## 專案概述

這是一個從零開始實現的 Neural Radiance Fields (NeRF) 專案，具有量子計算整合的前瞻性架構設計。專案採用模組化設計，便於理解、測試和擴展。

## 專案特色

- 🎯 **教育導向**：清晰的代碼結構和詳細的中文註釋
- 🔬 **模組化設計**：每個組件都可以獨立測試和替換
- 🌌 **量子就緒**：為未來的量子計算整合預留接口
- 📊 **完整流程**：從數據處理到模型訓練到結果渲染
- 🧪 **測試驅動**：完整的測試套件確保代碼品質

## 專案架構

```
quantum_nerf/
├── 📁 src/                     # 核心源代碼
│   ├── 📁 models/              # 模型定義
│   ├── 📁 utils/               # 工具函數
│   ├── 📁 training/            # 訓練相關
│   ├── 📁 rendering/           # 渲染相關
│   └── 📁 quantum/             # 量子計算模組
├── 📁 data/                    # 數據目錄
│   ├── 📁 raw/                 # 原始數據
│   ├── 📁 processed/           # 處理後數據
│   └── 📁 synthetic/           # 合成數據
├── 📁 configs/                 # 配置文件
├── 📁 tests/                   # 測試文件
├── 📁 notebooks/               # Jupyter 筆記本
├── 📁 docs/                    # 文檔
├── 📁 outputs/                 # 輸出結果
│   ├── 📁 models/              # 保存的模型
│   ├── 📁 images/              # 渲染圖像
│   └── 📁 logs/                # 訓練日誌
└── 📁 claude_advice/           # Claude 建議和思考基礎
```

## 快速開始

### 1. 環境設置
```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2. 運行演示
```bash
# 基本演示
jupyter notebook notebooks/01_基礎演示.ipynb

# 完整訓練
python src/training/train.py --config configs/basic_config.yaml
```

### 3. 測試
```bash
# 運行所有測試
pytest tests/

# 運行特定測試
pytest tests/test_models.py -v
```

## 主要組件

### 🧠 模型組件
- **NeRF 網絡**：核心神經輻射場模型
- **位置編碼器**：將 3D 座標編碼為高維特徵
- **體積渲染器**：實現體積渲染方程

### 🔧 工具組件
- **數據加載器**：處理圖像和相機參數
- **可視化工具**：結果展示和分析
- **配置管理**：靈活的參數配置

### 🌌 量子組件（未來）
- **量子位置編碼**：使用量子傅立葉變換
- **量子神經層**：變分量子電路
- **量子採樣**：量子增強的蒙特卡羅方法

## 開發指南

### 代碼風格
- 使用中文註釋說明關鍵概念
- 遵循 PEP 8 編碼規範
- 每個函數都有詳細的 docstring

### 測試要求
- 每個新功能都需要對應的測試
- 測試覆蓋率應保持在 80% 以上
- 使用 pytest 框架

### 文檔要求
- 重要概念需要有中文說明
- 代碼變更需要更新相應文檔
- 使用 Jupyter notebook 進行演示

## 貢獻指南

1. Fork 專案
2. 創建功能分支
3. 提交變更
4. 創建 Pull Request

## 授權

MIT License

## 聯繫方式

如有問題或建議，請創建 Issue 或聯繫專案維護者。 