# NeRF 數據流分析報告 📊

## 報告概述

本報告詳細分析了 Neural Radiance Fields (NeRF) 從原始輸入數據到最終渲染輸出的完整數據流程。我們將以 100 張不同視角的圖像作為輸入範例，說明數據在整個系統中的流動、處理和轉換過程。

## 📸 輸入數據結構

### 1. 原始輸入數據
```
輸入數據集/
├── 📁 images/                  # 圖像文件夾
│   ├── 📷 img_001.jpg         # 視角 1 的圖像 (800x600)
│   ├── 📷 img_002.jpg         # 視角 2 的圖像 (800x600)
│   ├── ...                    # ...
│   └── 📷 img_100.jpg         # 視角 100 的圖像 (800x600)
├── 📄 camera_poses.json       # 相機位置和方向
├── 📄 camera_intrinsics.json  # 相機內參
└── 📄 scene_metadata.json     # 場景元數據
```

### 2. 數據詳細說明

#### 圖像數據 (Images)
- **數量**: 100 張圖像
- **解析度**: 800×600 像素 (可調整)
- **格式**: RGB 彩色圖像
- **內容**: 同一個 3D 場景的不同視角
- **命名**: 按順序編號 (img_001.jpg 到 img_100.jpg)

#### 相機參數 (Camera Parameters)
```json
{
  "img_001.jpg": {
    "pose": [
      [0.9999, 0.0000, 0.0000, 2.5],    # 旋轉矩陣 + 平移向量
      [0.0000, 0.8660, -0.5000, 1.5],   # 4x4 變換矩陣
      [0.0000, 0.5000, 0.8660, 3.0],
      [0.0000, 0.0000, 0.0000, 1.0]
    ],
    "intrinsics": {
      "focal_length": 525.0,             # 焦距
      "cx": 400.0,                       # 主點 x 座標
      "cy": 300.0                        # 主點 y 座標
    }
  }
}
```

## 🔄 數據流程詳細分析

### 階段 1: 數據預處理 (Data Preprocessing)

#### 1.1 圖像載入與標準化
```python
# 輸入: 原始圖像文件
raw_images = load_images("images/")  # Shape: (100, 800, 600, 3)

# 處理: 標準化到 [0, 1] 範圍
normalized_images = raw_images / 255.0

# 輸出: 標準化圖像張量
# Shape: (100, 800, 600, 3), dtype: float32
```

#### 1.2 相機參數解析
```python
# 輸入: 相機參數 JSON 文件
camera_data = load_camera_parameters()

# 處理: 提取位姿矩陣和內參
poses = extract_poses(camera_data)      # Shape: (100, 4, 4)
intrinsics = extract_intrinsics(camera_data)  # Shape: (100, 3, 3)

# 輸出: 相機位姿和內參矩陣
```

#### 1.3 數據分割
```python
# 輸入: 100 張圖像和對應的相機參數
total_images = 100

# 處理: 按比例分割數據
train_indices = range(0, 80)      # 80 張訓練圖像
val_indices = range(80, 90)       # 10 張驗證圖像  
test_indices = range(90, 100)     # 10 張測試圖像

# 輸出: 分割後的數據集
train_data = {
    'images': images[train_indices],    # Shape: (80, 800, 600, 3)
    'poses': poses[train_indices],      # Shape: (80, 4, 4)
    'intrinsics': intrinsics[train_indices]  # Shape: (80, 3, 3)
}
```

### 階段 2: 射線生成 (Ray Generation)

#### 2.1 像素座標生成
```python
# 輸入: 圖像尺寸
height, width = 600, 800

# 處理: 生成像素網格
i, j = torch.meshgrid(
    torch.linspace(0, width-1, width),   # x 座標
    torch.linspace(0, height-1, height)  # y 座標
)

# 輸出: 像素座標網格
# i.shape: (600, 800), j.shape: (600, 800)
```

#### 2.2 射線方向計算
```python
# 輸入: 像素座標和相機內參
focal_length = 525.0
cx, cy = 400.0, 300.0

# 處理: 轉換為相機座標系
dirs = torch.stack([
    (i - cx) / focal_length,      # x 方向
    -(j - cy) / focal_length,     # y 方向 (注意負號)
    -torch.ones_like(i)           # z 方向 (朝向場景)
], dim=-1)

# 輸出: 射線方向向量
# dirs.shape: (600, 800, 3)
```

#### 2.3 世界座標系轉換
```python
# 輸入: 相機座標系射線方向和相機位姿
camera_pose = poses[0]  # 第一張圖像的位姿

# 處理: 旋轉到世界座標系
rays_d = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)

# 處理: 射線起點 (相機位置)
rays_o = camera_pose[:3, -1].expand(rays_d.shape)

# 輸出: 世界座標系中的射線
# rays_o.shape: (600, 800, 3) - 射線起點
# rays_d.shape: (600, 800, 3) - 射線方向
```

### 階段 3: 訓練數據採樣 (Training Data Sampling)

#### 3.1 隨機射線採樣
```python
# 輸入: 一張訓練圖像的所有射線
total_rays = height * width  # 600 * 800 = 480,000 條射線
batch_size = 1024

# 處理: 隨機選擇射線
ray_indices = torch.randperm(total_rays)[:batch_size]

# 輸出: 批次射線數據
batch_rays_o = rays_o.reshape(-1, 3)[ray_indices]  # Shape: (1024, 3)
batch_rays_d = rays_d.reshape(-1, 3)[ray_indices]  # Shape: (1024, 3)
batch_target_rgb = target_image.reshape(-1, 3)[ray_indices]  # Shape: (1024, 3)
```

### 階段 4: 3D 點採樣 (3D Point Sampling)

#### 4.1 沿射線採樣點
```python
# 輸入: 射線參數
near, far = 2.0, 6.0  # 近平面和遠平面
n_samples = 64        # 每條射線採樣 64 個點

# 處理: 分層採樣
t_vals = torch.linspace(0., 1., steps=n_samples)
z_vals = near * (1. - t_vals) + far * t_vals  # 深度值

# 處理: 添加隨機擾動 (訓練時)
if training:
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand

# 輸出: 採樣深度
# z_vals.shape: (1024, 64)
```

#### 4.2 計算 3D 點座標
```python
# 輸入: 射線起點、方向和深度值
# 處理: 計算 3D 點位置
pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

# 輸出: 3D 點座標
# pts.shape: (1024, 64, 3)
```

### 階段 5: 位置編碼 (Positional Encoding)

#### 5.1 正弦餘弦編碼
```python
# 輸入: 3D 點座標
input_pts = pts.reshape(-1, 3)  # Shape: (65536, 3)

# 處理: 多頻率編碼
L = 10  # 頻率數量
encoded_pts = [input_pts]  # 包含原始座標

for i in range(L):
    freq = 2.**i
    encoded_pts.append(torch.sin(freq * input_pts))
    encoded_pts.append(torch.cos(freq * input_pts))

# 輸出: 編碼後的位置特徵
encoded_pts = torch.cat(encoded_pts, dim=-1)
# encoded_pts.shape: (65536, 63)  # 3 + 3*2*10 = 63
```

#### 5.2 觀看方向編碼
```python
# 輸入: 射線方向
viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
viewdirs = viewdirs[:, None].expand(pts.shape)  # 擴展到所有採樣點

# 處理: 方向編碼 (較少頻率)
L_dir = 4
encoded_dirs = [viewdirs.reshape(-1, 3)]

for i in range(L_dir):
    freq = 2.**i
    encoded_dirs.append(torch.sin(freq * viewdirs.reshape(-1, 3)))
    encoded_dirs.append(torch.cos(freq * viewdirs.reshape(-1, 3)))

# 輸出: 編碼後的方向特徵
encoded_dirs = torch.cat(encoded_dirs, dim=-1)
# encoded_dirs.shape: (65536, 27)  # 3 + 3*2*4 = 27
```

### 階段 6: NeRF 網絡推理 (NeRF Network Inference)

#### 6.1 網絡前向傳播
```python
# 輸入: 編碼後的位置和方向特徵
# encoded_pts.shape: (65536, 63)
# encoded_dirs.shape: (65536, 27)

# 處理: 通過 NeRF 網絡
rgb, density = nerf_network(encoded_pts, encoded_dirs)

# 輸出: 顏色和密度預測
# rgb.shape: (65536, 3)      # RGB 顏色值 [0, 1]
# density.shape: (65536, 1)  # 體積密度值 [0, +∞]
```

#### 6.2 重塑為射線格式
```python
# 輸入: 平坦化的預測結果
# 處理: 重塑回射線×採樣點格式
rgb = rgb.reshape(1024, 64, 3)      # (batch_rays, n_samples, 3)
density = density.reshape(1024, 64, 1)  # (batch_rays, n_samples, 1)

# 輸出: 按射線組織的預測結果
```

### 階段 7: 體積渲染 (Volume Rendering)

#### 7.1 計算透射率和權重
```python
# 輸入: 密度值和深度間隔
# 處理: 計算相鄰點間距離
dists = z_vals[..., 1:] - z_vals[..., :-1]
dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

# 處理: 計算 alpha 值 (射線終止機率)
alpha = 1. - torch.exp(-density[..., 0] * dists)

# 處理: 計算透射率 (光線到達該點的機率)
transmittance = torch.cumprod(
    torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1
)[:, :-1]

# 處理: 計算重要性權重
weights = alpha * transmittance

# 輸出: 權重矩陣
# weights.shape: (1024, 64)
```

#### 7.2 顏色積分
```python
# 輸入: RGB 值和權重
# 處理: 加權平均得到最終像素顏色
rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

# 處理: 計算深度圖
depth_map = torch.sum(weights * z_vals, dim=-1)

# 輸出: 渲染結果
# rgb_map.shape: (1024, 3)  # 最終像素顏色
# depth_map.shape: (1024,)  # 深度值
```

### 階段 8: 損失計算與反向傳播 (Loss Computation & Backpropagation)

#### 8.1 損失計算
```python
# 輸入: 預測顏色和真實顏色
predicted_rgb = rgb_map      # Shape: (1024, 3)
target_rgb = batch_target_rgb  # Shape: (1024, 3)

# 處理: 計算 MSE 損失
mse_loss = torch.mean((predicted_rgb - target_rgb) ** 2)

# 處理: 計算 PSNR 指標
psnr = -10. * torch.log10(mse_loss)

# 輸出: 損失值和指標
# mse_loss: 標量
# psnr: 標量 (dB)
```

#### 8.2 梯度更新
```python
# 輸入: 損失值
# 處理: 反向傳播
optimizer.zero_grad()
mse_loss.backward()
optimizer.step()

# 輸出: 更新後的網絡參數
```

## 📊 數據集分割策略

### 訓練集 (Training Set) - 80 張圖像
- **用途**: 訓練 NeRF 網絡參數
- **特點**: 
  - 隨機射線採樣
  - 每個 epoch 使用不同的射線組合
  - 包含數據增強 (射線擾動)

### 驗證集 (Validation Set) - 10 張圖像  
- **用途**: 監控訓練過程，防止過擬合
- **特點**:
  - 固定的評估流程
  - 不參與梯度更新
  - 用於早停和超參數調整

### 測試集 (Test Set) - 10 張圖像
- **用途**: 最終性能評估
- **特點**:
  - 完全未見過的視角
  - 用於報告最終結果
  - 評估泛化能力

## 🔄 完整數據流程圖

```
原始數據 (100 張圖像)
         ↓
    數據預處理
    ├── 圖像標準化
    ├── 相機參數解析  
    └── 數據集分割
         ↓
    射線生成 (每張圖像 480,000 條射線)
    ├── 像素座標 → 相機座標
    ├── 相機座標 → 世界座標
    └── 射線起點 + 方向
         ↓
    訓練採樣 (每批次 1,024 條射線)
         ↓
    3D 點採樣 (每條射線 64 個點)
         ↓
    位置編碼
    ├── 3D 座標 → 63 維特徵
    └── 觀看方向 → 27 維特徵
         ↓
    NeRF 網絡推理
    ├── 輸入: (65,536, 63+27)
    └── 輸出: RGB (65,536, 3) + 密度 (65,536, 1)
         ↓
    體積渲染
    ├── 透射率計算
    ├── 權重計算
    └── 顏色積分
         ↓
    損失計算 & 反向傳播
    ├── MSE 損失
    ├── PSNR 指標
    └── 梯度更新
         ↓
    渲染輸出 (新視角圖像)
```

## 📈 數據量級分析

### 記憶體使用估算
```
單張圖像數據:
- 原始圖像: 800×600×3×4 bytes = 7.2 MB
- 射線數據: 480,000×6×4 bytes = 11.5 MB (起點+方向)
- 採樣點: 480,000×64×3×4 bytes = 368 MB

批次訓練數據:
- 射線批次: 1,024×6×4 bytes = 24.6 KB
- 採樣點: 1,024×64×3×4 bytes = 786 KB  
- 編碼特徵: 65,536×90×4 bytes = 23.6 MB
- 網絡輸出: 65,536×4×4 bytes = 1.0 MB
```

### 計算複雜度
```
每個訓練步驟:
- 射線採樣: O(1) - 隨機索引
- 3D 點採樣: O(batch_size × n_samples)
- 位置編碼: O(batch_size × n_samples × encoding_dim)
- 網絡推理: O(batch_size × n_samples × network_params)
- 體積渲染: O(batch_size × n_samples)

總複雜度: O(batch_size × n_samples × max(encoding_dim, network_params))
```

## 🎯 關鍵數據流特點

### 1. 多尺度處理
- **圖像級**: 整張圖像的相機參數
- **射線級**: 每個像素對應一條射線  
- **點級**: 每條射線上的多個採樣點

### 2. 隨機性引入
- **射線採樣**: 每批次隨機選擇射線
- **點採樣**: 分層採樣中的隨機擾動
- **數據增強**: 可選的射線抖動

### 3. 維度變換
- **2D → 3D**: 像素座標 → 世界座標
- **3D → 高維**: 位置編碼擴展特徵維度
- **高維 → 3D**: 網絡輸出 RGB + 密度
- **3D → 2D**: 體積渲染回到像素顏色

### 4. 並行處理
- **批次並行**: 同時處理多條射線
- **點並行**: 同時處理射線上的多個點
- **GPU 加速**: 所有計算都可以向量化

## 📝 總結

NeRF 的數據流程是一個複雜但優雅的管道，它將 2D 圖像觀測轉換為 3D 場景的隱式表示。關鍵在於：

1. **數據組織**: 從圖像到射線到點的層次化結構
2. **特徵編碼**: 位置編碼提供高頻細節表示能力  
3. **體積渲染**: 物理上合理的 3D 到 2D 投影過程
4. **端到端訓練**: 直接從 2D 監督信號學習 3D 表示

這個數據流設計使得 NeRF 能夠從有限的 2D 觀測中學習到豐富的 3D 場景表示，並能夠渲染出高質量的新視角圖像。 