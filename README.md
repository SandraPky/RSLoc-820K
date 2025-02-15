# RSLoc-82K: A Large-Scale Benchmark for Remote Sensing Image Geo-Localization

#### 这是论文 “Large-Scale Geo-Localization of Remote Sensing Images: A Three-Stage Framework Leveraging Maximal Clique Theory” 的官方数据集。

#### 测绘遥感信息工程全国重点实验室（武汉大学）

---

## 💬 简介

**RSLoc-82K** 是首个面向大规模遥感图像地理定位任务的开源基准数据集，旨在推动复杂场景下的高精度地理空间感知研究。

本数据集包含 **82,000+** 高分辨率遥感影像，覆盖 **100万平方公里** 的多样化地形，支持基于地理空间建模定位算法的评测。

🔗 **数据访问** | 📄 [论文链接（待发布）]() | 📦 [数据集]() | 💻 [代码仓库](https://github.com/SandraPky/RSLoc-82K)


## 🌍 数据集亮点

![dataset Example Image(15 zoom level)](paper/dataset.png)
RSLoc-82K数据集 数据示例 待定位图像/对应参考图像（部分）

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

其中，**15层级**图像数据是论文使用的主要参考图像，包含**820,000+** 张地理参考图像 ，分辨率约 **4.777m**。

![dataset Example Image(15 zoom level)](paper/gallery_area.PNG)
参考图库集连续覆盖的地理范围

- **真实场景挑战**  

  待定位图像**400**张，尺度较大可进行切割，主要分为**城市(48张)**和**非城市(352张)**遥感场景。

  ![city Example Image(15 zoom level)](paper/test_imgs_city.PNG)

  待定位图像：**城市**场景（F*文件）

  ![notcity Example Image(15 zoom level)](paper/test_imgs_notcity.PNG)

  待定位图像：**非城市**场景（a*/b*文件）

- **地理信息完整**  

  图像来自arcgis（a**.tiff/F**.tiff）和mapbox(a**.tiff), 数据集中提供tiff格式的元数据图像，可获得图像具体像素坐标/覆盖范围（WGS84），支持定位方法的准确性评估。

  
### 🚀 设计目标
- 填补现有数据集（如University-1652、SUES-200）在规模与场景覆盖上的不足。
- 为评估大规模地理定位算法的鲁棒性、泛化性及计算效率提供标准化基准。

---

## 🗂️ 数据集结构
```bash
RSLoc-82K/
├── queries/ # 测试集（500张）
│   ├── test100/ # 用于参数测试
│   ├── test400/ # 用于验证
│   │   ├── a*_L15_arcgis.tiff/m*_L15_mapbox.tiff # 非城市
│   │   └── F*_L15_arcgis.tiff # 城市
│   │
│   └── test100.csv/ # 图像对应地理信息（中心坐标、层级、尺寸、分辨率、卫星、地理坐标范围）
│   └── test400.csv/ # 
│
└── references/ # 图库集（参考数据库，连续覆盖，超过820,000张）
└── demo/ # 数据处理工具 

---

## 📥 下载与使用
### 步骤1：下载数据集
```bash
# 通过Git LFS下载（推荐）
git clone https://github.com/yourusername/RSLoc-82K-dataset.git
cd RSLoc-82K-dataset
git lfs pull
### 步骤2：快速验证
