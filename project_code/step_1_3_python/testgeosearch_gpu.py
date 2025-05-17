# MAC-based Geo-Localization step1 & step2
import torch
import time
from torchvision.io import decode_image,ImageReadMode
from torch.multiprocessing import Process, Queue, Manager
from transformers import CLIPModel
import torchvision.transforms as transforms
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import json
import csv
import glob
import shutil
import cv2
import sqlite3
import numpy as np
from PIL import Image
from osgeo import gdal
from pyproj import CRS, Transformer
import datetime
import logging
from logging.handlers import QueueHandler, QueueListener
logging.getLogger("QdrantClient").setLevel(logging.WARNING)
import networkx as nx

# STAGE 0：直接V-sift，不做hits找团
# 1:step1 Semantic-Guided Patch-Level Candidate Retrieval
# 2:step3 MAC-Guided Geometric Consistency for Fine Localization
STAGE = 1


class HTTPRequestFilter(logging.Filter):
    def filter(self, record):
        if "HTTP Request" in record.msg:
            return False
        return True
http_request_filter = HTTPRequestFilter()

logger = logging.getLogger()  # 获取根日志记录器
logger.setLevel(logging.INFO)  # 设置根日志记录器的日志级别为INFO
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 定义日志格式
# 创建文件处理器FileHandle 一小时一份
t1 = time.strftime("%Y%m%d", time.localtime(time.time()))  # %Y-%m-%d %H:%M:%S
file_handler = logging.FileHandler(f'CLIP_search_gpu{t1}.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # 将文件处理器添加到 根记录器logger
file_handler.addFilter(http_request_filter)
# 创建一个控制台处理器StreamHandler，并将其添加到根日志记录
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)  # 将控制台处理器添加到 根记录器logger
console_handler.addFilter(http_request_filter)

class Point:
    def __init__(self, x1, y1,res1, x2, y2,res2):
        self.x1 = x1
        self.y1 = y1
        self.res1 = res1
        self.x2 = x2
        self.y2 = y2
        self.res2 = res2
    def __str__(self):
        return f"({self.x1}, {self.y1}),({self.x2}, {self.y2})"

    def distance_th(self, other_point):
        dis1 = ((self.x1 - other_point.x1) ** 2 + (self.y1 - other_point.y1) ** 2) ** 0.5
        dis2 = ((self.x2 - other_point.x2) ** 2 + (self.y2 - other_point.y2) ** 2) ** 0.5
        th = abs(dis1* self.res1 - dis2* self.res2)
        return th

# ******基础计算函数******
# 行列号-瓦片中心经纬度
def xyztolonlat(level, col, row):
    min_x = -180
    min_y = -180
    max_x = 180
    max_y = 180
    level1_xn = 2
    level1_yn = 2
    left_top = True

    xTiles = level1_xn * pow(2, level - 1)
    yTiles = level1_yn * pow(2, level - 1)
    xTileWidth = (max_x - min_x) / xTiles
    yTileHeight = (max_y - min_y) / yTiles
    t_col = col
    t_row = row
    if not left_top:
        t_row = xTiles - row - 1

    w_half = (max_x - min_x) / 2;
    h_half = (max_y - min_y) / 2;
    x0 = col * xTileWidth - w_half;
    y0 = h_half - (t_row + 1) * yTileHeight;
    x1 = (col + 1) * xTileWidth - w_half;
    y1 = h_half - t_row * yTileHeight;

    x = (x0+x1)/2
    y = (y0 + y1)/2
    return x, y
#行列号-瓦片最大最小经纬度
def xyztolonlatmm(level, col, row):
    min_x = -180
    min_y = -180
    max_x = 180
    max_y = 180
    level1_xn = 2
    level1_yn = 2
    left_top = True

    xTiles = level1_xn * pow(2, level - 1)
    yTiles = level1_yn * pow(2, level - 1)
    xTileWidth = (max_x - min_x) / xTiles
    yTileHeight = (max_y - min_y) / yTiles
    t_col = col
    t_row = row
    if not left_top:
        t_row = xTiles - row - 1

    w_half = (max_x - min_x) / 2;
    h_half = (max_y - min_y) / 2;
    # 一张瓦片的范围
    x0 = col * xTileWidth - w_half;
    y0 = h_half - (t_row + 1) * yTileHeight;
    x1 = (col + 1) * xTileWidth - w_half;
    y1 = h_half - t_row * yTileHeight;
    dx = x1-x0
    dy = y1-y0
    #
    x0 = x0-dx
    y0 = y0-dy
    x1 = x1 + dx # 3*3的大小
    y1 = y1 + dy
    return x0,y0,x1,y1
# 经纬度-瓦片行列号索引
def geo_to_tile(level, lon, lat):
    min_x = -180.0
    min_y = -180.0
    max_x = 180.0
    max_y = 180.0
    level1_xn = 2.0
    level1_yn = 2.0

    xTiles = level1_xn * pow(2, level - 1)
    yTiles = level1_yn * pow(2, level - 1)

    xTileWidth = (max_x - min_x) / xTiles
    yTileHeight = (max_y - min_y) / yTiles

    w_half = (max_x - min_x) / 2
    h_half = (max_y - min_y) / 2

    tile_x = int((lon + w_half) / xTileWidth)
    tile_y = int((h_half - lat) / yTileHeight)

    return tile_x, tile_y
# 计算层级瓦片的空间分辨率
def compute_res(level):
    meters_per_pixel_at_level_0 = 20037508.3427892
    tile_size = 256
    resolution = meters_per_pixel_at_level_0 * 2 / tile_size / (2 ** level)
    return resolution
# 地理图像中心经纬度
def get_gdal_lonlat(dataset):
    geotransform = dataset.GetGeoTransform()
    projection_info = dataset.GetProjection()
    utm = CRS.from_wkt(projection_info)  # 获取原始影像的投影信息
    wgs84 = CRS.from_epsg(4326)  # WGS84 坐标系
    transformer = Transformer.from_crs(utm, wgs84, always_xy=True)  # 确保 x, y -> lon, lat
    xmin = geotransform[0]  # 左上角 X 坐标
    ymax = geotransform[3]  # 左上角 Y 坐标
    xres = abs(geotransform[1])  # X 方向分辨率
    yres = abs(geotransform[5])  # Y 方向分辨率
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    xmax = xmin + width * xres
    ymin = ymax - height * yres  # 右下角 Y 坐标
    # 计算中心点的像素坐标
    center_pixel_x = width // 2
    center_pixel_y = height // 2
    # 计算中心点的投影坐标
    center_x = xmin + center_pixel_x * xres
    center_y = ymax - center_pixel_y * yres
    # 转换为 WGS84 (lon, lat)
    lon, lat = transformer.transform(center_x, center_y)

    return lon, lat
# 地理图像最大最小经纬度
def get_gdal_extent(dataset):
    geotransform = dataset.GetGeoTransform()
    projection_info = dataset.GetProjection()
    utm = CRS.from_wkt(projection_info)
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(utm, wgs84, always_xy=True)  # 确保x,y顺序
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + dataset.RasterXSize * geotransform[1]
    ymin = ymax + dataset.RasterYSize * geotransform[5]

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    return lon_min, lat_min, lon_max, lat_max
# 计算两个向量之间的余弦相似度
def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
# 转换器序列（调大小平滑缩放、浮点归一、标准化）
transform2 = transforms.Compose([
        transforms.Resize([224,224], antialias=True),
        transforms.Lambda(lambda x: x.float().div(255)),
        transforms.Normalize((0.48145466,  0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
# 去除图像黑边
def cutBlack(pic):
    rows, cols = np.where(pic[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    return pic[min_row:max_row, min_col:max_col, :]
# 重新创建临时文件夹
def recreate_dir(dir):
    try:
        shutil.rmtree(dir)
    except Exception as e:
        pass
    try:
        os.makedirs(dir)
    except Exception as e:
        pass
# 判断这两个区域是否重叠
def do_regions_overlap(lon_min_A, lat_min_A, lon_max_A, lat_max_A, lon_min_B, lat_min_B, lon_max_B, lat_max_B):
    # 计算重叠区域的经纬度边界
    lon_overlap = max(lon_min_A, lon_min_B)
    lat_overlap = max(lat_min_A, lat_min_B)
    lon_overlap_end = min(lon_max_A, lon_max_B)
    lat_overlap_end = min(lat_max_A, lat_max_B)
    # 检查是否存在重叠
    if lon_overlap < lon_overlap_end and lat_overlap < lat_overlap_end:
        # 计算重叠区域的宽度和高度（假设地球是一个平面）
        width = lon_overlap_end - lon_overlap
        height = lat_overlap_end - lat_overlap
        if width >= 0.0002 and height >= 0.00015:
            return 1
        else:
            return 0
    else:
        # 没有重叠
        return 0

# 读取S1找到的点
def read_hits_CSV(hits_file_path):
    hits_pts = {}
    with open(hits_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader) # 跳过标题行
        for line in csv_reader:
            id = int(line[0])
            level = int(line[1])
            col =int(line[2])
            row = int(line[3])
            lon  = float(line[4])
            lat = float(line[5])
            time = int(line[6])
            rank = int(line[7])
            score =float(line[8])
            cos_sim = float(line[9])
            imgname=line[10]
            IMG=line[20]
            isplace = int(line[21])
            hits_pts.setdefault(imgname, [])  # 检查键是否存在，不存在则初始化为空列表
            point = {'id':id,'level':level,'col':col,'row':row,'lon':lon,'lat':lat,'time':time,
                     'rank':rank,'score':score,'cos_sim':cos_sim,'IMG':IMG,'isplace':isplace}
            hits_pts[imgname].append(point)
    return hits_pts
# 读取找团后 筛选出来的点
def read_cilque(cilque_file_path):
    points=[]
    clique_pts = {}
    with open(cilque_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader) # 跳过标题行
        for line in csv_reader:
            points.append(line)
            id = int(line[0])
            level = int(line[1])
            col =int(line[2])
            row = int(line[3])
            imgname=line[9]
            clique_pts.setdefault(imgname, [])  # 检查键是否存在，不存在则初始化为空列表
            clique_pts[imgname].append((id,level,col,row))
    return clique_pts

def read_into_buffer(filename):
    buf = bytearray(os.path.getsize(filename))
    with open(filename, 'rb') as f:
        f.readinto(buf)
    return buf

def read_tiff(img_path, img_dir, want_level, img_resolution,cut_th, gpu):
    IMG_name = os.path.basename(img_path)
    suffix = IMG_name.split(".")[-1]  # 图片后缀
    logger.info(f'{gpu}:正在预处理 {IMG_name} ')
    ds = gdal.Open(img_path)
    width, height = ds.RasterXSize, ds.RasterYSize
    img_lon, img_lat = get_gdal_lonlat(ds)
    # 计算裁剪策略
    tile_resolution = compute_res(want_level)
    small_size = int(3 * 256 * tile_resolution * cut_th / img_resolution)  # 切割窗口的大小#实验得出******1.2倍***************
    g_w_cut, g_h_cut = (width // small_size, height // small_size)
    # 计算裁剪影像的偏差 % 取模
    g_w_cut += width % small_size > 50
    g_h_cut += height % small_size > 50
    # 直接读取已处理数据
    image_info = {}
    json_file_path = os.path.join(img_dir, f"{IMG_name}.json")
    if os.path.exists(json_file_path):
        logger.info(f'{gpu}:{IMG_name} 存在历史处理信息')
        with open(json_file_path, 'r') as json_file:
            image_info = json.load(json_file)
    else:
        save_image(img_dir,  f"{IMG_name}_{img_lat}_{img_lon}.png", ds)  # 保存完整原始影像
        for i in range(g_h_cut):
            for j in range(g_w_cut):
                x = min(j * small_size, width - small_size)
                y = min(i * small_size, height - small_size)
                center_x, center_y = (x+small_size)/2, (y+small_size)/2  # 像素坐标
                # 会直接保存为tiff
                small_ds = gdal.Translate(os.path.join(img_dir, f'{IMG_name}_{i + 1}_{j + 1}_org.{suffix}'),
                                          ds,
                                          srcWin=[x, y, small_size, small_size],
                                          noData=0)  # 执行裁剪等操作，但不会将结果保存到文件中srcWin[xoff, yoff, xsize, ysize]
                # 空图像 跳过保存图像的步骤
                valid_pixel_count = (small_ds.GetRasterBand(1).ReadAsArray() != 0).sum()
                if valid_pixel_count < 50:
                    continue

                lon, lat = get_gdal_lonlat(small_ds)  # 中心经纬度
                col, row = geo_to_tile(want_level, lon, lat)  # 中心对应的瓦片行列号
                lon_min, lat_min, lon_max, lat_max = get_gdal_extent(small_ds)
                output_name = f"{IMG_name}_{i + 1}_{j + 1}_{col}_{row}.png"
                save_image(img_dir, output_name, small_ds)

                image_info[output_name] = (want_level,col,row,lon,lat,
                                           center_x,center_y,small_size,i + 1,j + 1,
                                           IMG_name,lon_min, lat_min, lon_max, lat_max)
                logger.info(f'{gpu}:图像块{output_name}, Lon_Lat({lon},{lat}), Col_Row({col},{row})')

        with open(json_file_path, 'w') as json_file:
            json.dump(image_info, json_file)

    return image_info, g_w_cut, g_h_cut

# *****输出文件函数*****
def save_image(image_path, file_name, image_data):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    file_path = os.path.join(image_path, file_name)
    # 如果文件已存在，跳过保存
    if os.path.exists(file_path):
        print(f"文件已存在，跳过保存：{file_name}")
        return
    # 保存图像数据到指定路径
    if isinstance(image_data, np.ndarray):  # 如果是 NumPy 数组
        if len(image_data.shape) == 3:
            pil_image = Image.fromarray(image_data.astype('uint8'))
            pil_image.save(file_path)
        else:
            print("非法NumPy图像数据格式")
    elif isinstance(image_data, Image.Image):  # 如果是 PIL 图像对象
        image_data.save(file_path)
    elif isinstance(image_data, np.uint8):  # 如果是 OpenCV 图像对象
        cv2.imwrite(file_path, image_data)
    elif isinstance(image_data, gdal.Dataset):  # 如果是 Dataset 图像对象
        rgb_image = image_data.ReadAsArray()  # 读取多个通道数组
        rgb_image = np.stack((rgb_image[0], rgb_image[1], rgb_image[2]), axis=-1)  # 合并三个通道成为真彩色图像
        pil_image = Image.fromarray(rgb_image.astype('uint8'))  # 数值转化为pil对象
        pil_image.save(file_path)
    else:
        # 如果是其他类型的二进制数据，直接写入文件
        with open(file_path, 'wb') as f:
            f.write(image_data)

    print(f"图像已保存为：{file_path}")
# 将搜索结果写入到 CSV 文件中
def write_hits(hits,csv_path,img_name,info):
    t_level = info[0]
    t_col = info[1]
    t_row = info[2]
    t_lon = info[3]
    t_lat = info[4]
    center_x = info[5]
    center_y = info[6]
    length = info[7]
    it_i = info[8]
    it_j = info[9]
    IMG_name = info[10]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w") as csvfile:
        if not file_exists:
            csvfile.write("id,level,col,row,lon,lat,time,rank,score,cos_sim,img_name,center_x,center_y,t_col,t_row,t_lon,t_lat,length,it_i,it_j,IMG,isplace\n")  # 表头，每一列的名称
        if len(hits) > 0:
            # 提取每个结果的属性
            for hit in hits:
                id = hit.id
                score = hit.score
                payload = hit.payload
                level = payload["level"]
                col = payload["col"]
                row = payload["row"]
                lon = payload["lon"]
                lat = payload["lat"]
                time = payload["time"]
                rank = payload["rank"]
                cos_sim = hit.payload.get("cos_sim",0)
                isplace = hit.payload.get('isplace',0)
                csvfile.write(f"{id},{level},{col},{row},{lon},{lat},{time},{rank},{score},{cos_sim},{img_name},{center_x},{center_y},{t_col},{t_row},{t_lon},{t_lat},{length},{it_i},{it_j},{IMG_name},{isplace}\n")  # 以csv写入到文件中
# 计算验证结果
def eval_result(img_files, eval_dir0,topK,d_time):
    if STAGE == 1:
        return 0
    eval_file = f'{eval_dir0}/all.txt'
    with open(eval_file, "w") as f:
        pass
    IMG, img_N, V_hits, hit_NUM, sum_p,IMG_acc= 0, 0, 0, 0, 0,0.0
    # 大图像  小图像 第一步召回 第二步成功 大影像成功 小图像准确率
    m = 0
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        eval = next(glob.iglob(f'{eval_dir0}/{img_name}*.json', recursive=True), None)
        if eval is None:
            continue
        with open(eval, 'r') as file:
            lines = file.readlines()
        if not lines:  # 如果lines为空
            print(f'json of {img_name} is NONE!')
            continue

        IMG +=1
        all_hits, hit_true, img_n, H_success = [], 0, 0,0
        # 候选(topk) 正确， 图像块数量

        for line in lines:
            img_n += 1
            json_data = json.loads(line.strip())
            v_hits = json_data['accuracy'][1]
            hits = json_data['hits']
            for hit in hits[:topK]:
            # for hit in hits:
                if hit['isplace'] == 1:
                    hit_true += 1
                    break
            all_hits.extend(hits[:topK])
        # 小图像的top1正确，则判定
        if any(hit['isplace'] == 1 for hit in all_hits):
            m += 1
        # 真正的top1，大图像top1
        top_hits = sorted(all_hits, key=lambda x: x['npts'], reverse=True)[:topK]  # 根据 npts 值排序并获取 topK
        if any(hit['isplace'] == 1 for hit in top_hits):
            sum_p += 1
        img_acc = hit_true / img_n if img_n != 0 else 0.0
        # if img_acc != 0.0:
        #     m += 1
        hit_NUM += hit_true  # 小图像成功 总数
        V_hits += v_hits
        img_N += img_n  # 小图像 总数
        IMG_acc += img_acc

        print(f"[{img_name}]: hits {hit_true}/{v_hits}/{img_n} [img_acc]:{img_acc} ")
        with open(eval_file, "a") as f:
            f.write(f"{img_name}: hits {hit_true}/{v_hits}/{img_n} [img_acc]:{img_acc} \n")

    # 计算总体结果
    Acc = sum_p / IMG  # 数据集成功率
    Acc_easy = m / IMG
    IMG_acc = IMG_acc / IMG  # 平均准确度
    with open(eval_file, "a") as f:
        f.write(f"""IMG_EVAL:
                    [hits]:{hit_NUM}/{V_hits}/{img_N} 
                    [Acc]:{sum_p}/{IMG}={Acc}
                    [Acc_easy]:{m}/{IMG}={Acc_easy}
                    [IMG_acc]:{IMG_acc}
                    Execution time: {d_time} s \n""")
    print(f"m = {m}/350")
    print(f"IMG_EVAL: [hits]:{hit_NUM}/{V_hits}/{img_N} [Acc]:{sum_p}/{IMG}={Acc} [Acc_easy]:{m}/{IMG}={Acc_easy} [IMG_acc]:{IMG_acc} Execution time: {d_time} s \n")


# ****数据库相关函数**********
# 扩展瓦片至3*3图像
db_path = './Gallery/db1.db'
connection = sqlite3.connect(db_path)
def extract_db_image_expand(connection,level,col,row, size):
    trans_image = np.ones((256*size, 256*size, 3), dtype=np.uint8) * 255  # 初始化图像矩阵
    s = int(size / 2)  #
    s_ = -s
    for i in range(s_,s+1,1):
        for j in range(s_,s+1,1):
            cursor = connection.cursor()  # 游标
            clevel = level
            ccol = col + j
            crow = row + i
            sql = f"select rowid,tile_data  from ge_tiles where zoom_level = {clevel} and tile_column = {ccol} and tile_row = {crow}"
            cursor.execute(sql)
            rows = cursor.fetchmany(1)
            if len(rows) == 0:
                continue
            # 获取图像数据并进行解码、拼接
            rowid, tile_data = rows[0]
            image = np.asarray(bytearray(tile_data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            trans_image[(i+s)*256:(i+s+1)*256,(j+s)*256:(j+s+1)*256,:] = image[:, :, :]
    return trans_image
# 在数据库中搜索具有相似特征向量的数据
def search_vector_db(client,collection_name,query_vector,limit,imginfo,level,gpu):
    total_hits = []
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        # with_vectors=True,
        score_threshold=0.80,
        offset=0,
        search_params=models.SearchParams(hnsw_ef=512,exact=True),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="level",match=models.MatchAny(any=[level,0]),),]
        ),
        limit=limit,
    )

    lon_min = imginfo[11]
    lat_min = imginfo[12]
    lon_max = imginfo[13]
    lat_max = imginfo[14]

    for i, hit in enumerate(hits):
        col = hit.payload['col']
        row = hit.payload['row']
        level0 = hit.payload["level"]
        hit.payload["rank"] = i + 1
        hit.payload["isplace"] = 0
        lon0, lat0, lon1, lat1 = xyztolonlatmm(level0, col, row)
        if do_regions_overlap(lon_min, lat_min, lon_max, lat_max, lon0, lat0, lon1, lat1) == 1:
            hit.payload["isplace"] = 1
        lon, lat = xyztolonlat(level, col, row)
        hit.payload["lon"] = lon
        hit.payload["lat"] = lat

        total_hits.append(hit)

    logger.info(f'{gpu}:=====return:V 0-{limit}--[{len(total_hits)}]=====')
    return total_hits

# 特征计算函数******************
# 图像中提取特征,转换为np数组返回
def extract_file_feature(model,device,image_path):
    try:
        with torch.no_grad():
            tile_data = read_into_buffer(image_path)  # 读取图像数据
            nparr = torch.frombuffer(tile_data, dtype=torch.uint8)  # 数据转换为张量
            org_img = decode_image(nparr, ImageReadMode.RGB)  # RGB 模式
            img = org_img.to(device)  # 转移设备
            img = transform2(img).unsqueeze(0)  # 图像预处理，增加维度以便作为模型输入
            image = img
            image_features_g = model.get_image_features(image)  # 利用模型提取图像的特征表示   不归一化
            image_features = torch.nn.functional.normalize(image_features_g, p=2, dim=1)  # 归一化处理，具有单位范数
            return image_features.view(-1).cpu().numpy()  # 张量展平，转换为np数组
    except (torch.nn.modules.module.ModuleAttributeError, IOError, RuntimeError) as e:
        logger.exception(f'图片提取特征向量出错: {image_path}')
        return None

# sift匹配点，筛选团
def find_sift_clique(src_pts,dst_pts,good,maxcli=10):
    g_img_resolution = compute_res(15)  # MapBox/Arcgis【14:9.554 15:4.777 16:2.388】 ***
    g_tile_resolution = compute_res(15)
    res_ave = (g_img_resolution+g_tile_resolution)/2
    max_th = res_ave * 64
    # 提取源图像中关键点
    points = []
    for i in range(len(good)):
        x1, y1 = src_pts[i][0]
        x2, y2 = dst_pts[i][0]
        res1 = g_img_resolution # 你需要提供res1的值
        res2 = g_tile_resolution # 你需要提供res2的值
        point = Point(x1, y1, res1, x2, y2, res2)
        points.append(point)
    G = nx.Graph()
    edges = []
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            th = points[i].distance_th(points[j])
            if th < max_th:
                edges.append((i,j))
    G.add_edges_from(edges)
    cliques = list(nx.find_cliques(G))
    cliques.sort(key=len, reverse=True)
    top_cliques = cliques[:maxcli]
    points_cli = set()
    for clique in top_cliques:
        points_cli.update(clique)
    new_src_pts = np.array([src_pts[i] for i in points_cli])
    new_dst_pts = np.array([dst_pts[i] for i in points_cli])
    filtered_good = [good[i] for i in points_cli]
    return new_src_pts,new_dst_pts,filtered_good

# 从给定图像/NumPy数组中提取局部特征信息
sift = cv2.SIFT_create()
def extract_img_local_feature(image_input):
    # 初始化变量
    img = None
    gray = None
    kp = None
    des = None
    # 检查输入类型
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"The image at {image_input} could not be read.")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError("Input must be a string (file path) or a NumPy array.")
    # 将图像数据转换为灰度图像
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    # 使用 SIFT 特征提取器对灰度图像进行特征检测和计算,返回关键点 kp1 和描述子 des1
    kp, des = sift.detectAndCompute(gray, None)  # des是描述子
    return img,gray,kp,des
# 计算图像之间的局部特征相似度
def compute_loca_sim(model,device,image_path1,image_path2,tmp_dir):
    match1 = 0.85  # 原始图像 用于筛选匹配点的比率阈值。
    match2 = 0.85  # 变换后 比率阈值
    secod_sim_T = 0.85 # 0.85   # 用于确定图像是否足够相似的相似度阈值。
    try:
        img1, gray1, kp1, des1 = extract_img_local_feature(image_path1)
        img2, gray2, kp2, des2 = extract_img_local_feature(image_path2)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        matches_with_ratio = []
        for m, n in matches:
            ratio = m.distance / n.distance
            matches_with_ratio.append((m, n, ratio))
        matches_with_ratio = sorted(matches_with_ratio, key=lambda x: x[2], reverse=False)
        for m, _, ratio in matches_with_ratio:
            if ratio < 0.85 and len(good)<500:
                good.append(m)
        if len(good) < 4:
            return 0,0
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # 原图像匹配点坐标
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # 图像2匹配点坐标
        src_pts, dst_pts, filtered_good = find_sift_clique(src_pts, dst_pts, good, 3) #*******

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)  # 计算单应性矩阵
        match_num1 = np.sum(mask, axis=0)[0]  # 计算匹配点数量
        if M is None:
            return 0, 0
        result = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) # 透视变换第一张图像以与第二张图像对齐
        result = cutBlack(result)
        tmp_img = os.path.join(tmp_dir, 'localsim_temp.jpg')
        try:
            os.remove(tmp_img)
        except Exception as e:
            pass
        cv2.imwrite(tmp_img, result)
        # 从变换后的图像和第二张图像中提取特征
        f1 = extract_file_feature(model,device,tmp_img)
        f2 = extract_file_feature(model,device,image_path2)
        sim = get_cos_similar(f1,f2)  # 计算相似度分数
        # 如果相似度低于阈值，则返回0
        if sim < secod_sim_T:
            return 0, sim
        # 如果相似度高于阈值，对变换后图像和第二张图像的局部特征进行匹配
        kp1, des1 = sift.detectAndCompute(result, None)  # des是描述子
        matches2 = bf.knnMatch(des1, des2, k=2)
        good2 = []
        # ******************
        for m, n in matches2:
            if m.distance < match2 * n.distance:
                good2.append(m)

        if len(good2) < 4:
            return 0, 0
        src_pts2 = np.float32([kp1[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
        dst_pts2 = np.float32([kp2[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)
        M2, mask2 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 4.0)
        if M2 is None:
            return 0, 0
        match_num = np.sum(mask2, axis=0)[0]  # 计算匹配点数量
        return int(match_num), sim   # 返回匹配点数量和相似度

    except Exception as e:
        print(f"local SIFT-sim 发生错误：{e}")
        return 0, 0

# 从hits中找出目标点（SIFT+找团）
def hits_local_sift_sim(q_img_name,hits,search_dic,g_connect,g_local_img_dir2, g_model, g_device,gpu):
    if not isinstance(hits[0], dict):  # hit是原始的查询结果格式
        hits = [{'id': hit.id, 'score': hit.score, 'level': hit.payload['level'], 'col': hit.payload['col'],
                 'row': hit.payload['row'], 'lat': hit.payload['lat'], 'lon': hit.payload['lon'],
                 'time': hit.payload['time'], 'rank': hit.payload['rank'],
                 'isplace': hit.payload['isplace']} for hit in hits]
    image_path = os.path.join(g_local_img_dir2, q_img_name)
    find = False
    for index, hit in enumerate(hits):
        id = hit['id']
        score = hit['score']
        level = hit['level']
        col = hit["col"]
        row = hit["row"]
        time = hit["time"]
        rank = hit["rank"]
        lat = hit["lat"]
        lon = hit["lon"]
        isplace = hit["isplace"]
        time_str = datetime.datetime.utcfromtimestamp(time).strftime('%Y%m%d')
        trans_image = extract_db_image_expand(g_connect, level, col, row, 3)  # 扩展图像并保存
        expand_tile_path = f'{g_local_img_dir2}/{q_img_name}_expandtemp.png'
        cv2.imwrite(expand_tile_path, trans_image)

        # 局部匹配
        npts, sim = compute_loca_sim(g_model, g_device, image_path, expand_tile_path, g_local_img_dir2)
        f = {'id': id, 'level': level, 'col': col, 'row': row, 'lon': lon, 'lat': lat, 'time': time,
             'time_str': time_str, 'rank': rank, 'score': score, 'npts': npts, 'loca_sim': sim,
             'isplace': isplace}
        if npts >= 25 and sim > 0.92:  # 20 0.85
            search_dic['hits'].append(f)
            logger.info(f'{gpu}:{q_img_name} local_hit[{col}_{row}]：{index}/{len(hits)},npts{npts},sim{sim},answer:{isplace}')
            if isplace == 1:
                find = True  # 这里单指最后一步正确
            if len(search_dic.get('hits', [])) >= g_topK.value:
                break
        else:
            if isplace == 1:
                logger.info(f'{gpu}:{q_img_name} local_hit[{col}_{row}]：{index}/{len(hits)},npts{npts},sim{sim},answer:{isplace} ***fail***')

    return find,search_dic


def search_image(q, gpu):
    # 在进程中将日志记录器切换为QueueHandler
    logger = logging.getLogger()
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
    logger.addHandler(queue_handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # 指定使用的GPU设备
    logger.info(f'{gpu}:Process ({os.getpid()}) is reading...')
    g_device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  # 根据系统中是否可用CUDA，选择将模型加载到GPU还是CPU设备上
    g_model = CLIPModel.from_pretrained(mode_path).to(g_device)  # 将模型加载到相应设备**************先关掉
    g_client = QdrantClient(host='192.168.210.204', port=int(6333), timeout=20)
    g_connect = sqlite3.connect('/data/gw/france/db1.db')
    glevel = g_level.value
    IMG_dir = g_img_dir.value
    limit = 2000  # 1000  返回一次的数量

    while True:
        IMG_file = q.get()  # 等待10秒 获取队列中的数据
        if IMG_file is None:
            break  # 如果收到终止标志，退出循环
        IMG_name = os.path.basename(IMG_file)
        logger.info(f'{gpu}:=====影像 {IMG_name}=====')
        g_local_img_dir2 = os.path.join(g_local_img_dir.value, os.path.splitext(IMG_name)[0])  # 切割后图像 f'{g_local_img_dir}/{IMG_name}'
        os.makedirs(g_local_img_dir2, exist_ok=True)
        image_info, w_cut, h_cut = read_tiff(IMG_file, g_local_img_dir2, glevel, g_img_resolution.value, g_cut_th.value, gpu)
        # continue
        hit_file_path = g_hits_dir.value + f"/hits_{IMG_name}_{w_cut}x{h_cut}.csv" # 候选点
        cilque_file_path = g_hits_dir.value + f'/clique_{IMG_name}_{w_cut}x{h_cut}.csv'
        g_eval = f'{g_eval_dir.value}/{IMG_name}_{w_cut}x{h_cut}_l{glevel}.json'
        if os.path.exists(hit_file_path):
            hits_pts = read_hits_CSV(hit_file_path)

        simg_num = len(image_info)  # 小图像数量

        # ————————————   Stage 1  ————————————————————————
        if STAGE == 1:   # 有约束
            if os.path.exists(hit_file_path):
                continue
            for img_name, info in image_info.items():
                img_name = img_name
                logger.info(f'{gpu}: 生成 hits [{img_name}]')
                image_path = os.path.join(g_local_img_dir2, img_name)
                query_vector = extract_file_feature(g_model, g_device, image_path)  # 提取查询特征向量
                hits = search_vector_db(g_client, g_collection.value, query_vector, limit, info, glevel, gpu)
                write_hits(hits, hit_file_path, img_name, info)
            continue  # 不进行后续

        if STAGE == 2:
            if not os.path.exists(cilque_file_path):
                print(f'not exists {cilque_file_path}')
                continue
            clique_pts = read_cilque(cilque_file_path)

        # ———————————— Stage 2 小影像 局部 sim————————————————————————
        flag_1 = 0
        flag_2 = 0
        # ***读取之前处理的信息
        img_name_exist = []  # 未完成的验证文件
        if os.path.exists(g_eval):
            with open(g_eval, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]  # 读取并去除空白行
                logger.info(f'{gpu}:{IMG_name}验证 [{len(lines)}/{simg_num}]')
            for line in lines:
                json_data = json.loads(line)
                img = json_data['q_img_name']
                flag_2, flag_1 = json_data['accuracy'][:2]
                img_name_exist.append(img)
                logger.info(f'{gpu}:[{img}]{flag_2}/{flag_1}/{simg_num}')  # 目前已完成的小图像结果
        else:
            with open(g_eval, "w"):
                logger.info(f'{gpu}:{IMG_name}验证 [0/{simg_num}]')

        for img_name, info in image_info.items():
            if img_name in img_name_exist:  # 查询是否已存在验证结果，跳过循环
                continue
            logger.info(f'{gpu}:=== {img_name}===')
            image_path = os.path.join(g_local_img_dir2, img_name)
            q_col,q_row,q_lon,q_lat = info[1],info[2],info[3],info[4]
            search_dic = {'q_img_name': img_name, 'q_col_row_lat_lon': (q_col,q_row,q_lat,q_lon),
                          'accuracy': (flag_2, flag_1, simg_num), 'hits': []}
            v_find = False  # 向量查询
            find = False

            if STAGE == 2:
                hits = hits_pts.get(img_name, []) # 原始的hits

                pts = clique_pts.get(img_name, [])  # clique点过滤

                # hits2 = [hit for pt in pts for hit in hits if hit['id'] == pt[0]] # cli顺序
                hits2 = [hit for hit in hits if any(hit['id'] == pt[0] for pt in pts)]
                v_find = any(hit['isplace'] == 1 for hit in hits2)
                if v_find:
                    flag_1 += 1

                logger.info(f'{gpu}:[{img_name}] clique/org_hits = {len(hits2)}/{len(hits)}')

                hits = hits2

                if hits:
                    find,search_dic= hits_local_sift_sim(img_name,hits,search_dic,g_connect,g_local_img_dir2, g_model, g_device,gpu)
                if find is True:
                    flag_2 += 1
                    logger.info(f'{gpu}:{img_name} 局部特征匹配 成功, eval[{flag_2}/{flag_1}/{simg_num}]')
                else:
                    logger.info(f'{gpu}:{img_name} 局部特征匹配 *失败*, eval[{flag_2}/{flag_1}/{simg_num}]')
                search_dic['accuracy'] = (flag_2,flag_1,simg_num)
                with open(g_eval, "a") as j_:
                    j_.write(json.dumps(search_dic, ensure_ascii=False) + "\n")
                continue


# 初始化数据路径
g_mode_path = None
g_collection = None
g_img_dir = None
g_local_img_dir = None
g_eval_dir = None
g_level = None
g_cut_th = None
g_img_resolution = None
g_tile_resolution = None
g_hits_dir = None
def init_filepath(mode_path, collection_name, img_dir, output_dir, want_level=15, cut_th=1.0,tile_resolution=1.0, img_resolution=1.0):
    global g_level, g_cut_th,  g_img_resolution, g_tile_resolution,g_hits_dir
    g_level = want_level
    g_cut_th = cut_th
    g_img_resolution = img_resolution
    g_tile_resolution = tile_resolution

    global g_mode_path,g_collection, g_img_dir, g_eval_result, g_local_img_dir, g_eval_dir
    g_mode_path = mode_path
    g_collection = collection_name
    g_img_dir = img_dir
    g_local_img_dir = output_dir + f"/local_img_L{g_level}"
    g_eval_dir = output_dir + f"/eval_L{g_level}"  # 0/1
    g_hits_dir = output_dir + f"/eval_L{g_level}"
    if STAGE==2:
        g_eval_dir = output_dir + f"/eval_L{g_level}/S2"

    os.makedirs(g_local_img_dir, exist_ok=True)
    os.makedirs(g_eval_dir, exist_ok=True)


if __name__ == '__main__':
    mode_path = '/.project_code/model/clip-vit-large-patch14'
    qdrant_ip = '192.168.210.204'
    qdrant_port = int(6333)
    db_path = '/data/gw/france/db1.db'
    collection_name = 'test_geosearch_clip_v2'
    img_resolution = None
    topK = 1
    cut_th = 1  # 1.2
    cpu_num =6

    # *****【MapBox/Arcgis瓦片】************
    img_level = 15
    want_level = 15
    img_resolution = compute_res(img_level)  # MapBox/Arcgis【14:9.554 15:4.777 16:2.388】 ***
    tile_resolution = compute_res(want_level)
    IMG_dir = './RSimages/test400'
    output_dir = f'./output/clip-vit-L14'

    init_filepath(mode_path,collection_name, IMG_dir, output_dir, want_level, cut_th, tile_resolution, img_resolution)
    eval_dir = g_eval_dir

    img_files1 = glob.glob(f'{g_img_dir}/**/*.tiff', recursive=True)
    img_files2 = glob.glob(f'{g_img_dir}/**/*.jp2', recursive=True)
    img_files = img_files1 + img_files2
    with Manager() as manager:
        g_collection = manager.Value(str,g_collection)
        g_level = manager.Value(int, g_level)
        g_cut_th = manager.Value(int, g_cut_th)
        g_img_resolution = manager.Value(float, g_img_resolution)
        g_tile_resolution = manager.Value(float, g_tile_resolution)
        g_mode_path = manager.Value(str, g_mode_path)
        g_img_dir = manager.Value(str, g_img_dir)
        g_local_img_dir = manager.Value(str, g_local_img_dir)
        g_eval_dir = manager.Value(str, g_eval_dir)
        g_img_files = manager.list(img_files)
        g_topK = manager.Value(int, topK)
        g_hits_dir = manager.Value(str, g_hits_dir)

        log_queue = manager.Queue()
        queue_handler = QueueHandler(log_queue)  # 创建QueueHandler，并将其添加到根日志记录
        logger.addHandler(queue_handler)
        queue_listener = QueueListener(log_queue, file_handler, console_handler)  # 创建QueueListener，并将其绑定到根日志记录
        queue_listener.start()
        start_time = time.time() # ***

        q = Queue(1000)
        processes = []
        for i in range(cpu_num):
            p = Process(target=search_image, args=(q, str(i)))
            p.start()
            processes.append(p)

        for finish_row, img_file in enumerate(g_img_files, start=1):
            q.put(img_file)
            print(f"--finish {finish_row}/{len(g_img_files)}--")
        for i in range(cpu_num):
            q.put(None)
        for processe in processes:
            processe.join()

        queue_listener.stop()
        end_time = time.time()
        d_time = end_time - start_time
        print("Execution time:", d_time, "seconds")

    eval_result(img_files, eval_dir,topK,d_time)
    print("Execution time:", d_time, "seconds")