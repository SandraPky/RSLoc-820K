# RSLoc-82K: A Large-Scale Benchmark for Remote Sensing Image Geo-Localization
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**RSLoc-82K** 是首个面向大规模遥感图像地理定位任务的开源基准数据集，旨在推动复杂场景下的高精度地理空间感知研究。本数据集包含 **82,000+** 高分辨率遥感影像，覆盖 **100万平方公里** 的多样化地形，支持跨时相、跨分辨率及跨传感器场景下的算法评测。

🔗 **数据访问** | 📄 [论文链接（待发布）]() | 💻 [代码仓库（待发布）]()

---

## 🌍 数据集亮点
### 📊 关键特性
- **规模与多样性**  
  包含 **82,000+** 地理参考图像，覆盖多种地形，分辨率范围 **0.3m–2m**。
- **真实场景挑战**  
  模拟跨时相（季节变化）、跨分辨率（多卫星源）及光照变化（晨昏/云层干扰）等实际定位难题。
- **结构化标注**  
  每张图像提供精确地理坐标（WGS84）、拍摄时间、传感器类型及地形类别标签，支持端到端空间建模。

### 🚀 设计目标
- 填补现有数据集（如University-1652、SUES-200）在规模与场景覆盖上的不足。
- 为评估大规模地理定位算法的鲁棒性、泛化性及计算效率提供标准化基准。

---
## 🗂️ 数据集结构
RSLoc-82K/
├── queries/ # 测试集（500张）
│ ├── test/ # 用于参数测试
│ ├── queries/ # 用于验证
│ └── metadata_images/ # 元数据图像文件（tiff格式）
│
└── references/ # 图库集（参考数据库，连续覆盖，超过820,000张）
│
└── evaluation_scripts/ # 评估工具包
└── topk_accuracy.py # Top-K准确率计算
└── geospatial_utils/ # 地理空间分析工具

---

## 📥 下载与使用
### 步骤1：下载数据集
```bash
# 通过Git LFS下载（推荐）
git clone https://github.com/yourusername/RSLoc-82K-dataset.git
cd RSLoc-82K-dataset
git lfs pull
### 步骤2：快速验证
