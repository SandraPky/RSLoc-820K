from osgeo import gdal
from pyproj import CRS, transform, Transformer

# 地理图像中心经纬度
def get_gdal_lonlat(dataset):
    geotransform = dataset.GetGeoTransform()
    projection_info = dataset.GetProjection()
    utm = CRS.from_wkt(projection_info)
    #utm = CRS.from_epsg(32631)  # 原始图像数据的坐标参考系统
    wgs84 = CRS.from_epsg(4326)  # 目标坐标参考系统
    xmin = geotransform[0]
    ymin = geotransform[3] + (dataset.RasterYSize * geotransform[5])
    xmax = xmin + (dataset.RasterXSize * geotransform[1])
    ymax = ymin - (dataset.RasterYSize * geotransform[5])
    res_x = abs(geotransform[1])
    res_y = abs(geotransform[5])
    center_pixel_x = int((dataset.RasterXSize / 2))
    center_pixel_y = int((dataset.RasterYSize / 2))
    center_point = [xmin + center_pixel_x * res_x, ymax - center_pixel_y * res_y]
    latlon = transform(utm, wgs84, center_point[0], center_point[1])
    latitude = latlon[0]
    longitude = latlon[1]
    return longitude, latitude

# 地理图像最大最小经纬度
def get_gdal_extent(dataset):
    geotransform = dataset.GetGeoTransform()
    projection_info = dataset.GetProjection()
    utm = CRS.from_wkt(projection_info)
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(utm, wgs84)

    xmin = geotransform[0]
    ymax = geotransform[3]
    xres = geotransform[1]
    yres = geotransform[5]
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    xmax = xmin + width * xres
    ymin = ymax + height * yres

    lat_min,lon_min = transformer.transform(xmin, ymin)
    lat_max,lon_max = transformer.transform(xmax, ymax)

    return lon_min, lat_min, lon_max, lat_max

# 计算层级瓦片的空间分辨率
def compute_res(level):
    meters_per_pixel_at_level_0 = 20037508.3427892
    tile_size = 256
    resolution = meters_per_pixel_at_level_0 * 2 / tile_size / (2 ** level)
    return resolution


IMG_path ='/RSLoc-82K/RSimages/test400/a1_L15_arcgis.tiff'
ds = gdal.Open(IMG_path)
width, height = ds.RasterXSize, ds.RasterYSize
lon, lat = get_gdal_lonlat(ds)
lon_min, lat_min, lon_max, lat_max = get_gdal_extent(ds)