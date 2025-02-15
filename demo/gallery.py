import sqlite3
import numpy as np
import cv2


# 行列号->中心经纬度
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

#行列号->经纬度范围
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

# 经纬度->行列号索引
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

def extract_db_image_expand_np(connection,level,col,row,size=3):
    cursor = connection.cursor()  # 游标
    trans_image = np.ones((256*size, 256*size, 3), dtype=np.uint8) * 255  # 初始化图像矩阵
    s = int(size / 2)  #
    s_ = -s
    for i in range(s_,s+1,1):
        for j in range(s_,s+1,1):
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


gallery_dir = './RSLoc-82K/galleryL15.db'
connection = sqlite3.connect(gallery_dir)
cursor = connection.cursor()
sql = "select rowid,zoom_level,tile_column,tile_row,time_ from ge_tiles"
cursor.execute(sql)
rows = cursor.fetchall()
for row in rows:
    rowid,zoom_level, tile_column, tile_row, time,tile_data = row
    image = np.asarray(bytearray(tile_data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # ...
    cv2.imwrite('./img_dir/imgname.jpg', image)