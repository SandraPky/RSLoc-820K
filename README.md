# RSLoc-82K: A Large-Scale Benchmark for Remote Sensing Image Geo-Localization

#### 这是论文 “Large-Scale Geo-Localization of Remote Sensing Images: A Three-Stage Framework Leveraging Maximal Clique Theory” 的官方数据集。

#### 测绘遥感信息工程全国重点实验室（武汉大学）

---

## 💬 简介

**RSLoc-82K** 是首个面向大规模遥感图像地理定位任务的开源基准数据集，旨在推动复杂场景下的高精度地理空间感知研究。

本数据集包含 **82,000+** 高分辨率遥感影像，覆盖 **100万平方公里** 的多样化地形，支持基于地理空间建模定位算法的评测。

🔗 **数据访问** | 📄 [论文链接（待发布）]() | 📦 [数据集]() | 💻 [代码仓库](https://github.com/SandraPky/RSLoc-82K)


## 🌍 数据集亮点
### 📊 关键特性
- **参考图库集：大规模、连续覆盖**  

  包含**多层级**、**多分辨率**、**多时相**卫星遥感影像，覆盖多种地形
  
| Zoom level | Tile Count | Time Range                         | Resolution (m/px) | Tile Spacing |
|-------------|------------|-------------------------------------|-------------------|--------------|
| 13          | 66,144     | 2020/12/31                          | 19.109            | 4891.970m    |
| 14          | 240,400    | 1985-12-31 to 2023-8-16             | 9.554             | 2445.985m    |
| 15     | 824,796| 1991-12-31 to 2023-8-16     | 4.777             | 1222.991m    |
| 16          | 3,216,120  | 1991-12-31 to 2023-8-16             | 2.389             | 611.494m     |
| Total       | 4,347,460  | 1985-12-31 to 2020-12-31            | —                 | —            |

  其中，包含 15层级具有 **820,000+** 张地理参考图像，分辨率约 **4.777m**，是论文使用的主要参考图像。
  
  ![Example Image](paper../paper/dataset.jpg)

- **真实场景挑战**  
  待定位图像400张，模拟跨时相（季节变化）、跨源（多卫星源）、跨尺度（多层级与分辨率）及等实际定位难题。

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
