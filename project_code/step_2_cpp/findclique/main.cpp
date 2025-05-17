#include <cstdio>
#include <string>
#include <vector>
#include "math.h"
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include "shapefil.h"
#include <igraph/igraph.h>
#include<functional>
#include <algorithm>
#include <queue>
#include <hiredis.h>
#include "nlohmann/json.hpp"
#include "sqlite3.h"
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <fstream>
#include <chrono>
#include <fstream>
#include <unordered_set>
using namespace nlohmann;
using namespace std;
using SimilarityRecord = std::tuple<std::string, std::string, double>; //相似度矩阵数据，用于存储每一对图像及其相似度

#define pi 3.1415926535897932384626433832795
#define EARTH_RADIUS 6378.137 //地球半径 KM


// 多准则评分结构体
struct CliqueScore {
    int idx;            // 团索引
    double size_val;  // 规模
    double sim_val;   // 相似度
    double geo_val;   // 地理紧密
    double size_score;  // 规模得分
    double sim_score;   // 相似度得分
    double geo_score;   // 地理紧密得分
    double total_score; // 综合得分
    double distance; //与准确坐标的距离
};
// 点 对象
class pt{
public:
    std::string patch_id;
    int level;
    std::string panid;
    int col;
    int row;
    int time;
    float score;
    float x;  //经度
    float y;  //纬度
    std::string imgname;
    float center_x;
    float center_y;
    int length;
    std::string IMG;
    int isplace;
    int rank;
    float cli_score;
};
// 图形 对象
class Graph{
public:
    std::vector<pt> pts; 
    std::vector<std::pair<int, int>> edges; //边：两个点的索引
    igraph_t graph; //图形结构，包含图形信息
    vector<vector<int>> cliques; // 团：多个点索引 向量
    std::vector<CliqueScore> clique_scores;  // 评分存储
public:
    // 构造函数
    Graph() {
        igraph_empty(&graph, 0, IGRAPH_UNDIRECTED); // 初始化一个空图
    }
    // 新增深拷贝构造函数
    Graph(const Graph& other) {
        // 复制基础数据
        pts = other.pts;
        edges = other.edges;
        cliques = other.cliques;
        clique_scores = other.clique_scores;
        // 深拷贝igraph图
        igraph_copy(&this->graph, &other.graph);
    }
    // 禁用默认赋值运算符
    Graph& operator=(const Graph&) = delete;
    // 新增图克隆方法
    Graph clone() const {
        return *this; // 利用拷贝构造函数
    }

    // 析构函数，销毁图形对象
    ~Graph() {
        igraph_destroy(&graph); // 销毁图形对象
    }
};
// 边 对象
class edge_info
{
public:
    double score;
    int id1;
    int id2;
};

//原图像信息
struct ImageInfo {
    std::string imgname;
    double lat;
    double lon;
    int level;
    int width;
    int height;
    double resolution;
    std::string type;
    float grid_w_x;
    float grid_w_y;
    double lon_min;
    double lat_min;
    double lon_max;
    double lat_max;
};
//局部图像信息
struct locImageInfo {
    std::string imgname;
    float center_x;
    float center_y;
    int length;
    int it_i;
    int it_j;
};
//特征点 类
class fpoint {
public:
    int level;
    int col;
    int row;
    int time;
    float score;
    std::string id;  // 特征点的唯一标识符
};
// 团的信息 类
class clique_info
{
public:
    int idx;
    int size;
    double score;
};
// 地理坐标点结构
struct GeoPoint {
    double x;
    double y;
    GeoPoint(double _x, double _y) : x(_x), y(_y) {}
};


//*********文件名、路径***************
// 文件是否存在
bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}
// 分割字符串（待分割字符串，分隔符）
std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();  //分割的起始位置0、终止位置、分隔符的长度
    std::string token; // 单个子字符串 
    std::vector<std::string> res;  //所有子字符串 向量

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token); //添加到结果向量
    }

    res.push_back(s.substr(pos_start));  //剩余的字符串 添加
    return res;
}
// 读取路径中的所有文件名
std::vector<std::string> get_csv_files(const char* directory) {
    std::vector<std::string> files;
    DIR* dir = opendir(directory);
    struct dirent* ent;
    if (dir) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string filename = ent->d_name;
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".csv") {
                files.emplace_back(std::string(directory) + "/" + filename);
            }
        }
        closedir(dir);
    }
    else {
        std::cerr << "Error opening directory" << std::endl;
    }
    return files;
}
// 生成cli文件名
std::string generate_output_csv(const char* input_csv, const char* output_path) {
    std::string filename = std::string(input_csv).substr(std::string(input_csv).find_last_of("/") + 1);
    size_t pos = filename.find("hits_");
    if (pos != std::string::npos) {
        filename.replace(pos, 5, "clique_");
    }
    return std::string(output_path) + "/" + filename;
}

//*********读取数据***************
// 读取hits文件 点数据 过滤
void read_file_to_pts(const char* geo_point_file, std::vector<pt>& pts0, std::vector<locImageInfo>& locimginfos0, int sv = 0) {
    //读取点数据 点结构 点向量集合
    std::ifstream read_file(geo_point_file, std::ios::binary);
    std::string line;
    
    std::vector<pt> pts;
    std::vector<locImageInfo> locimginfos;


    getline(read_file, line);  //  跳过标题行
    while (getline(read_file, line)) {
        //std::vector<std::string> strs = split(line, ",");// 将每一行用逗号分割成子字符串
        auto strs = split(line, ",");  // 分割行数据
        pt p;
        p.patch_id = strs[0];

        if (sv == 1) {p.panid = strs[1];}  // 读取 panid 街景就是panid字段\或者 无人机参考卫星图像名称
        else {p.level = atoi(strs[1].c_str());}  //  RS就是level字段
        
        p.col = std::stoi(strs[2]);
        p.row = std::stoi(strs[3]);
        p.x = std::stod(strs[4]);  // 经度
        p.y = std::stod(strs[5]);  // 纬度
        p.time = std::stoi(strs[6]);
        p.rank = std::stoi(strs[7]);
        p.score = std::stod(strs[8]);  // 向量相似度
        p.imgname = strs[10];
        p.center_x = std::stod(strs[11]);  // 中心点x坐标
        p.center_y = std::stod(strs[12]);  // 中心点y坐标
        p.length = std::stoi(strs[17]);  // 图像边长
        p.IMG = strs[20];
        p.isplace = std::stoi(strs[21]);

        pts.push_back(p);  // 将解析得到的点对象添加到点的向量中

        // 检查 locimginfos 中是否存在相同的 imgname
        if (std::none_of(locimginfos.begin(), locimginfos.end(),
            [&p](const locImageInfo& info) { return info.imgname == p.imgname; })) {

            locImageInfo locimginfo{
                strs[10],                     // imgname
                atoi(strs[11].c_str()),       // center_x
                atoi(strs[12].c_str()),       // center_y
                atoi(strs[17].c_str()),       // length
                atoi(strs[18].c_str()),       // it_i
                atoi(strs[19].c_str())        // it_j
            };
            locimginfos.push_back(locimginfo);
        }
    }
    read_file.close();// 关闭文件
    locimginfos0 = locimginfos;

    //过滤完全重复点
    std::set<std::string> pts_infos;
    std::vector<pt> new_ptts;
    for (const auto& p : pts) {
        std::string uuid = p.imgname + "_" + p.patch_id;
        if (pts_infos.insert(uuid).second) {  // 判断是否插入成功（已存在就不成功）
            new_ptts.push_back(p);
        }
    }
    pts0 = new_ptts;

    std::cout << "local img number : " << locimginfos0.size() << " , pts point number：" << pts0.size() << endl;

    //{   //1. 将重复的同位点进行合并分类
    //    map<string, vector<pt>> patch_match_pts;
    //    std::vector<pt> new_pts;
    //    for (int i = 0; i < pts.size(); i++) {
    //        stringstream uuid;
    //        pt& p = pts[i];
    //        uuid << p.patch_id << "_" << p.level << "_" << p.col << "_" << p.row;
    //        map<string, vector<pt>>::iterator iter = patch_match_pts.find(uuid.str());
    //        if (iter == patch_match_pts.end()) {
    //            vector<pt> pts;
    //            pts.push_back(p);
    //            patch_match_pts[uuid.str()] = pts;
    //        }
    //        else {
    //            patch_match_pts[uuid.str()].push_back(p);
    //        }
    //    }
    //    //2. 若重复点太多则通过 原始匹配相似度分数进行筛选
    //    map<string, vector<pt>>::iterator iter;
    //    for (iter = patch_match_pts.begin(); iter != patch_match_pts.end(); iter++) {
    //        vector<pt>& pts = iter->second;
    //        int top_k = 20;
    //        if (pts.size() > top_k) {
    //            std::sort(pts.begin(), pts.end(), [](const pt p1, const pt p2) {
    //                return p1.score > p2.score;
    //                });
    //        }
    //        int want_n = std::min(int(pts.size()), top_k);
    //        for (int i = 0; i < want_n; i++) {
    //            new_pts.push_back(pts[i]);
    //        }
    //    }
    //    pts = new_pts;
    //}
    //{ // 直接删除重复的点
    //    set<string> pts_map;
    //    std::vector<pt> new_pts;
    //    for (int i = 0; i < pts.size(); i++) {
    //        stringstream uuid;
    //        pt& p = pts[i];
    //        uuid << p.patch_id << "_" << p.level << "_" << p.col << "_" << p.row;
    //        if (pts_map.find(uuid.str()) == pts_map.end()) {
    //            pts_map.insert(uuid.str());
    //            new_pts.push_back(p);
    //        }
    //    }
    //    pts = new_pts;
    //}
    //cout << "new_pts point number：" << pts.size() << endl;   //基本无效
    //pts0 = pts;
}

//*********  小算法 ***************

// 角度值转换为弧度值
double rad(double d){
    return d * pi / 180.0;
}
//计算地球上两点之间的实际距离
double RealDistance(double lat1, double lng1, double lat2, double lng2)//lat1第一个点纬度,lng1第一个点经度,lat2第二个点纬度,lng2第二个点经度
{
    double a;
    double b;
    double radLat1 = rad(lat1);
    double radLat2 = rad(lat2);
    a = radLat1 - radLat2;
    b = rad(lng1) - rad(lng2);
    double s = 2 * asin(sqrt(pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2), 2))); //Haversine 公式计算两点之间的球面距离
    s = s * EARTH_RADIUS;
    s = s * 1000;
    return s;
}
//计算给定图块级别（level）对应的分辨率
double compute_res(int level) {
    // 定义Web Mercator投影下的常量
    const double meters_per_pixel_at_level_0 = 20037508.3427892;
    const int tile_size = 256; // 瓦片的像素大小
    // 计算特定层级的分辨率
    double resolution = meters_per_pixel_at_level_0 * 2.0 / tile_size / std::pow(2.0, level);
    return resolution;
}

// 二维点坐标（x,y） 结构体
struct T_Point2D
{
    double X, Y; //横 纵 坐标
    T_Point2D() { X = 0.0; Y = 0.0; }
    T_Point2D(double _x, double _y) { X = _x; Y = _y; }
};
//计算三角形面积（三个点）
double Area(T_Point2D p0, T_Point2D p1, T_Point2D p2)
{
    double area = 0;
    area = p0.X * p1.Y + p1.X * p2.Y + p2.X * p0.Y - p1.X * p0.Y - p2.X * p1.Y - p0.X * p2.Y; //顶点坐标的叉积计算
    return area / 2;
}
// 计算一个团中的 节点的经度和纬度范围
void compute_diss(vector<pt>& pts, vector<int>& cliques,double &lon_dis,double &lat_dis) {
    double min_lon = 99999999;
    double min_lat = 99999999;
    double max_lon = -99999999;
    double max_lat = -99999999;
    for (int i = 0; i < cliques.size(); i++) {
        pt& p1 = pts[cliques[i]];
        min_lon = std::min((double)p1.x, min_lon);
        min_lat = std::min((double)p1.y, min_lat);
        max_lon = std::max((double)p1.x, max_lon);
        max_lat = std::max((double)p1.y, max_lat);
    }

    double distance1 = RealDistance(max_lat, min_lon, max_lat, max_lon);  //经纬度 - 地理距离
    double distance2 = RealDistance(max_lat, min_lon, min_lat, min_lon);
    lon_dis = distance1;
    lat_dis = distance2;

}




//判断两个点集合是否可以合并（是否有位置重合的节点）
bool can_combine2(vector<pt>& pts,vector<int>& cliques1, vector<int>& cliques2,int th_same_ptn) {
    //空间范围变化 判断是否合并
    double lon_dis1, lat_dis1;
    double lon_dis2, lat_dis2;
    compute_diss(pts, cliques1, lon_dis1, lat_dis1);  //cliques1 的经度和纬度的矩形范围
    //合并了两个点集合
    vector<int> tmp_cliques2;
    tmp_cliques2 = cliques1;
    for (int i = 0; i < cliques2.size(); i++) {
        tmp_cliques2.push_back(cliques2[i]);
    }
    compute_diss(pts, tmp_cliques2, lon_dis2, lat_dis2); //合并后的点集合的经度和纬度的范围
    //处理距离值为0的情况
    if (lon_dis1 <= 0.0001) {
        lon_dis1 = 0.001;
    }
    if (lat_dis1 <= 0.0001) {
        lat_dis1 = 0.001;
    }
    // 判断是否合并， 1.5倍
    if (lon_dis2 / lon_dis1 > 1.5 || lat_dis2 / lat_dis1 > 1.5)
       return false;
   else
       return true;

}
// 将两个团合并为一个新的点集合combine_clique
vector<int> combine(vector<pt>& pts, vector<int>& clique1, vector<int>& clique2) {
    vector<int> combine_clique = clique1;  // 先复制clique1中的点
    for (int i : clique2) {
        // 检查clique2的点是否已经在combine_clique中
        bool add = true;
        for (int j : clique1) {
            float d = fabs(pts[i].x - pts[j].x) + fabs(pts[i].y - pts[j].y);
            if (d < 0.00001) {
                add = false;
                break;
            }
        }
        if (add) {
            combine_clique.push_back(i);
        }
    }
    return combine_clique;
}
// 将相似团 合并（点信息pts、一组团 cliques、整数确定是否合并）（合并后的团与下一个点集合判断）
vector<vector<int>> combine_clique(vector<pt>& pts, vector<vector<int>>& cliques, int th_same_ptn) {
    // 每个团-邻居团 的映射 i:j1,j2,j3,..
    map<int, vector<int>> clique_relations;  //字典

    // 构建团关系映射
    for (int i = 0; i < cliques.size(); ++i) {
        // 寻找与当前团可以合并的团
        for (int j = 0; j < cliques.size(); ++j) {
            if (can_combine(pts, cliques[i], cliques[j], th_same_ptn)) {
                clique_relations[i].push_back(j); // 为i创建映射，并把j的索引加进去
            }
        }
    }

    //初始化标识符 (能合并的团 给同一个标识符)
    vector<int> clique_ids(cliques.size(), -1);  // 团的标识符 初始化 分配-1
    int current_new_idx = 0; //从0开始分配

    // 分配标识符
    for (int i = 0; i < cliques.size(); ++i) {
        int my_idx = -1;
        // 遍历邻居团  获取邻居团的标识
        for (int neb_id : clique_relations[i]) {
            if (clique_ids[neb_id] >= 0) {
                my_idx = clique_ids[neb_id];
                break; //只要和一个邻居一样的标识符就可以了
            }
        }
        clique_ids[i] = (my_idx >= 0) ? my_idx : current_new_idx++; //如果邻居团里全都没有分配标识符，则产生一个新的标识符
    }

    // 根据标识符合并团
    map<int, vector<int>> new_cliques;  // 用于存储每个团的新标识符及其对应的团
    for (int i = 0; i < clique_ids.size(); i++) {
        int my_id = clique_ids[i]; //获得当前团的新id  my_id
        // 如果 new_cliques 中还没有 my_id，则直接赋值；否则将两个团合并
        if (new_cliques.count(my_id) == 0) {
            new_cliques[my_id] = cliques[i]; //当前id：当前团 { my_id：团i }
        }
        else {
            new_cliques[my_id] = combine(pts, new_cliques[my_id], cliques[i]);  //直接合并两个相同id的团，{ my_id：团x } . 团i   更新
        }
    }

    // 将新映射中的团 保存为团向量
    vector<vector<int>> new_out_cliques;
    for (auto& [_, clique] : new_cliques) {
        new_out_cliques.push_back(clique);
    }
    return new_out_cliques;
}



//*********找到图形中的极大团****************
void find_clipque(Graph& g, int max_cli, int conbine, float topclique) {
    // Step 1: 找到所有最大团1000 1 0.6
    vector<vector<int>> cliques; //团 集合
    igraph_vector_ptr_t result;  //找到的团  存储指针数组

    auto start1 = std::chrono::steady_clock::now(); //计时

    igraph_vector_ptr_init(&result, 0); // 初始化
    igraph_maximal_cliques(&g.graph, &result, 3, 0);
    //igraph_cliques(&g.graph, &result, 3, 0);  //（输入图，团储存，3最少节点，0不额外处理）
    auto end1 = std::chrono::steady_clock::now(); //计时

    int size = igraph_vector_ptr_size(&result); //团数量
    
    // 团 转化为点向量clique结构，并存到cliques中
    for (int i = 0; i < size; i++) {
        igraph_vector_t* p = (igraph_vector_t*)igraph_vector_ptr_e(&result, i); //从指针数组中获取第 i 个团的指针，并将其转换为 igraph_vector_t* 类型的指针 p
        // 将团的结构 转换为clique 点向量
        vector<int> clique(igraph_vector_size(p));
        for (int j = 0; j < clique.size(); j++) {
            clique[j] = (int)VECTOR(*p)[j];  //按顺序取出团中的节点索引，就是点在集合中的顺序号 转换为 vector<int>
        }
        cliques.push_back(clique);
        igraph_vector_destroy(p);
        delete p;
    }
    igraph_vector_ptr_destroy(&result); //销毁原始的团集合
    printf("find %d cliques（time: %d ms）\n", cliques.size(), 
        std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count());

    auto start2 = std::chrono::steady_clock::now(); //计时

    //  Step 2: 计算团的分数sim和size，按大小和分数排序
    {
        vector<clique_info> COSInfos(cliques.size());
        for (int i = 0; i < cliques.size(); i++) {
            float total_score = 0.0;
            for (int pt_idx : cliques[i]) {
                total_score += g.pts[pt_idx].score;
            }
            COSInfos[i] = { i, static_cast<int>(cliques[i].size()), total_score / cliques[i].size() };
        }

        //重排序 先大小 后平均 cos_sim
        std::sort(COSInfos.begin(), COSInfos.end(), [](const clique_info& a, const clique_info& b) {
            if (a.size != b.size) { return a.size > b.size; }  // 优先按大小排序，降序
            else { return a.score > b.score; }  // 如果大小相同，再按平均 cossim 分数排序，降序
            });
        
        vector<vector<int>> sorted_cliques;
        for (const auto& cinfo : COSInfos) {
            sorted_cliques.push_back(cliques[cinfo.idx]);
        }
        cliques = std::move(sorted_cliques);
    }

    //Step 3: 去除不满足大小限制的团
    {
        float max_clique_size = cliques[0].size(); //最大节点数，已经排过序了
        float clique_size_t = max_clique_size * topclique;  //去掉较小节点的团
        int top_clique_n = std::min(max_cli, (int)cliques.size());
        //float clique_size_t = (cliques[0].size()) * topclique;  //去掉较小节点的团  最大节点数*比例
        vector<vector<int>> new_cliques;
        for (const auto& clique : cliques) {
            if (new_cliques.size() >= top_clique_n || clique.size() < clique_size_t)
                break;  //数量限制、大小限制
            new_cliques.push_back(clique);
        }
        cliques = std::move(new_cliques);
        printf("%d cliques ( after cliques 小团去除（%d 以内） ) \n", cliques.size(), max_cli);
    }

    //Step 4: 合并重叠的团
    while (true) {
        vector<vector<int>> updated_cliques = combine_clique(g.pts, cliques, conbine);//合并团 *******************************
        if (updated_cliques.size() == cliques.size()) break;
        cliques = std::move(updated_cliques);
    }
    printf("%d cliques ( after combining ) \n", cliques.size());

    //Step 5: 按图像数量计算团的分数并排序 新的分数规则    （包含的imgname点数量）并重排序
    vector<clique_info> infos(cliques.size());
    for (int i = 0; i < cliques.size(); i++) {
        std::set<string> match_pts;
        int score = 0;
        for (int pt_idx : cliques[i]) {
            if (match_pts.insert(g.pts[pt_idx].imgname).second) {
                score++;  // 尝试插入 imgname，如果成功则增加分数
            }
        }
        infos[i] = { i, static_cast<int>(cliques[i].size()), static_cast<double>(score) };
    }
    // 按分数降序排列
    std::sort(infos.begin(), infos.end(), [](const clique_info& a, const clique_info& b) {
        return a.score > b.score;
        });

    vector<vector<int>> new_cliques2;
    int maxcli = 500;//限制 输出团的数量
    for (int i = 0; i < std::min(maxcli, static_cast<int>(infos.size())); i++) {
        new_cliques2.push_back(cliques[infos[i].idx]);  // 获取排序后团的索引并添加到 new_cliques2
        for (int pt_idx : new_cliques2.back()) {
            g.pts[pt_idx].cli_score = infos[i].score;
        }
    }
    cliques = std::move(new_cliques2);
    g.cliques = cliques;

    auto end2 = std::chrono::steady_clock::now(); //计时

    printf("%d cliques ( after filter, time %d ms ) \n", cliques.size(), 
        std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start1).count());

}


//*********  写入数据  ***************
// 图的团（clique）写入shapefile 文件
void write_clique_to_shpfile(const char* csv_path, const char* target_dir, Graph& graph, int maxSize) {
    // 生成文件名
    std::string filename = std::string(csv_path).substr(std::string(csv_path).find_last_of("/") + 1);  //提取最后一级文件名
    std::string filename_without_extension = filename.substr(0, filename.find_last_of("."));  //只提取非后缀的部分
    std::string aaa = std::string(target_dir) + "/" + filename_without_extension + ".shp";
    const char* saveFileName = aaa.c_str();

    std::vector<pt>& pts = graph.pts;
    std::vector<std::pair<int, int>>& edges = graph.edges;
    const auto& cliques = graph.cliques;

    int nShpTpyeSave = SHPT_POLYGON;
    SHPHandle outShp = SHPCreate(saveFileName, SHPT_POLYGON);
    DBFHandle dbf_h = DBFCreate(saveFileName);
    DBFAddField(dbf_h, "Shape", FTString, 250, 0);
    //遍历所有团
    for (int i = 0; i < std::min(maxSize, static_cast<int>(cliques.size())); i++) {
        const auto& clique = cliques[i];
        std::vector<double> xCoords(clique.size() + 1);
        std::vector<double> yCoords(clique.size() + 1);
        for (int j = 0; j < clique.size(); j++) {
            xCoords[j] = pts[clique[j]].x;
            yCoords[j] = pts[clique[j]].y;
        }
        //图形闭合
        xCoords.back() = xCoords[0];
        yCoords.back() = yCoords[0];
        SHPObject* psShape = SHPCreateObject(SHPT_POLYGON, -1, 0, nullptr, nullptr, clique.size() + 1, xCoords.data(), yCoords.data(), nullptr, nullptr);
        int ishape = SHPWriteObject(outShp, -1, psShape);

        //计算团中 查询点imgname的数量（不同的点）
        stringstream g_id;
        std::set<string> match_pts;
        for (int pt_idx : clique) {
            match_pts.insert(pts[pt_idx].imgname);
        }
        DBFWriteStringAttribute(dbf_h, ishape, 0, std::to_string(match_pts.size()).c_str());
        SHPDestroyObject(psShape);
    }

    std::cout << "finish write " << cliques.size() << " cliques Shape\n\n";

    SHPClose(outShp);
    DBFClose(dbf_h);
}
// 将团 cli 筛选后的候选 写入csv文件 
void write_clique_to_hit(Graph& g, const char* input_csv, const char* output_path) {
    std::string input_csv_str(input_csv);
    std::string output_csv = output_path + std::string("/") + input_csv_str.substr(input_csv_str.find_last_of("/") + 1);

    size_t pos = output_csv.find("hits_");
    if (pos != std::string::npos) {
        output_csv.replace(pos, 5, "clique_");  //std::string("hits_").length()=5
    }

    // 将团中的点索引取出
    vector<vector<int>> cliques = g.cliques;
    std::vector<pt> new_pts;
    for (const auto& clique : cliques) {
        for (int pt_idx : clique) {
            new_pts.push_back(g.pts[pt_idx]);
        }
    }
    std::cout << "new pts point number：" << new_pts.size() << endl;

    //排序整理
    std::sort(new_pts.begin(), new_pts.end(), [](const pt& p1, const pt& p2) {
        if (p1.imgname != p2.imgname) {
            return p1.imgname < p2.imgname;
        }
        else if (p1.cli_score != p2.cli_score) {
            return p1.cli_score > p2.cli_score;
        }
        else {
            return p1.rank < p2.rank;
        }
        });

    // 将new_pts中的p一行一行写进output_csv
    std::ofstream output_file(output_csv);
    if (!output_file) {
        std::cerr << "Error opening output file: " << output_csv << std::endl;
        return;
    }
    output_file << "id,level,col,row,lon,lat,time,rank,score,imgname,center_x,center_y,length,cli_score,isplace\n";
    for (const auto& p : new_pts) {
        output_file << p.patch_id << "," << p.level << "," << p.col << "," << p.row << ",";
        output_file << p.x << "," << p.y << "," << p.time << "," << p.rank << "," << p.score << ",";
        output_file << p.imgname << "," << p.center_x << "," << p.center_y << "," << p.length << "," << p.cli_score << "," << p.isplace << std::endl;
    }
}


// 创建一个图形对象

Graph create_graph(const char* geo_point_file, float res, int pt_edge_max, int max_edges, int strategy) {
    std::vector<pt> pts;
    std::vector<locImageInfo> locimginfos;  // 存储位置信息
    read_file_to_pts(geo_point_file, pts, locimginfos);  // 从文件读取点和位置信息

    // 构建网格索引
    //float grid_w_x = imginfo.grid_w_x;
    //float grid_w_y = imginfo.grid_w_y;
    const float grid_w_x = 0.03; // 网格宽度（x方向） 0.01度 11.0km*6.6km 对应15层级查询范围
    const float grid_w_y = 0.015; // 网格宽度（y方向） 0.005
    int grid_n_x = static_cast<int>(360 / grid_w_x); // 计算x方向上的网格数量
    int grid_n_y = static_cast<int>(180 / grid_w_y); // 计算y方向上的网格数量
    std::map<std::string, std::vector<int>> grids;  //网格字典 存储网格和对应的点索引
    for (int id = 0; id < pts.size(); id++) {
        pt& p = pts[id];
        int gridx = int((p.x + 180) / grid_w_x); // 计算网格x坐标
        int gridy = int((p.y + 90) / grid_w_y);  // 计算网格y坐标
        std::string g_id = std::to_string(gridx) + "_" + std::to_string(gridy);
        grids[g_id].push_back(id); // 将点索引添加到对应网格
    }
    //printf("网格的长宽格子数量： %d x %d\n   使用网格数量：%d\n",  grid_n_x, grid_n_y,grids.size());

    // ******构建边******
    auto start0 = std::chrono::steady_clock::now();  //计时开始
    long cur_finish_n = 0;
    long total_finish_n = pts.size();
    long last_percent = -1;

    std::vector<std::pair<int, int>> edges;  // 存储边
    std::set<std::string> edges_map; // 存储已存在的边，以避免重复
    const int grid_n = 5; //周围网格范围
    
    for (int id1 = 0; id1 < pts.size(); id1++) {
        pt& p = pts[id1];
        set<int> neb_ids; // 存储邻居点的ID
        int gridx = static_cast<int>((p.x + 180) / grid_w_x);
        int gridy = static_cast<int>((p.y + 90) / grid_w_y);
        // 收集邻近点
        for (int g1 = -grid_n; g1 <= grid_n; g1++) {
            for (int g2 = -grid_n; g2 <= grid_n; g2++) {
                std::string g_id = std::to_string(gridx + g1) + "_" + std::to_string(gridy + g2);
                if (grids.find(g_id) != grids.end()) {
                    neb_ids.insert(grids[g_id].begin(), grids[g_id].end());
                }
            }
        }
        
        // ***评估边*** 
        //// 计算与邻近点的距离并筛选边
        std::vector<edge_info> match_edges;  //score,id1,id2
        for (int neb_id : neb_ids) {
            pt& p2 = pts[neb_id];
            bool is_ok = false;
            if (p.patch_id != p2.patch_id && p2.imgname != p.imgname) {
                // 瓦片分辨率
                float tile_res = (compute_res(p.level) + compute_res(p2.level)) / 2.0;
                if (std::abs(p.level - p2.level) > 1) {
                    tile_res = std::max(compute_res(p.level), compute_res(p2.level));
                }

                double tile_dis = RealDistance(p.y, p.x, p2.y, p2.x);  // 瓦片地理距离
                double pic_dis = sqrt(pow((p.center_x - p2.center_x) * res, 2) + pow((p.center_y - p2.center_y) * res, 2));  // 图像地理距离
                double dd = std::fabs(pic_dis - tile_dis); // 距离 偏差
                double th_dis = 1.5 * 256 * tile_res;  //容差阈值**********************
                // 如果距离偏差在阈值内且真实距离大于100，则记录边
                if (dd < th_dis && tile_dis > 100) {
                    is_ok = true;
                    match_edges.push_back({ dd, id1, neb_id }); // 加入边 score,id1,id2
                }
            }

        }
        
        //1-周围 中 将所有边按长度升序，筛选最小偏差的边
        int max_k = pt_edge_max;
        if (match_edges.size() > max_k) {
            std::sort(match_edges.begin(), match_edges.end(), [](const edge_info edge1, const edge_info edge2) {
                return edge1.score < edge2.score;
                }); 
        } //排序
        int want_k = std::min(max_k, static_cast<int>(match_edges.size()));
        for (int i = 0; i < want_k; i++) {
            edge_info& e = match_edges[i];
            std::string edge_id1 = std::to_string(e.id1) + "-" + std::to_string(e.id2);
            std::string edge_id2 = std::to_string(e.id2) + "-" + std::to_string(e.id1);
            if (edges_map.find(edge_id1) == edges_map.end() && edges_map.find(edge_id2) == edges_map.end()) {
                edges_map.insert(edge_id1);
                edges_map.insert(edge_id2);
                //if (edges.size() > 80000) break;
                edges.push_back({ e.id1, e.id2 });//索引对
            }
        }  //去重
        
        // 显示进度
        long finish_percent = (cur_finish_n * 10) / total_finish_n;    //进度显示
        if (finish_percent != last_percent) {
            last_percent = finish_percent;
            printf("finish create graph :%d\n", (int)finish_percent);
        }
        //if (edges.size() > 40000) break;
        cur_finish_n++;
    }
    printf("create graph %d 原始edges \n", edges.size());
    
    
    //筛选边
    if (edges.size() >= max_edges) {
        // 策略1: 取短边
        if (strategy == 1) {
            //计算边的长度，筛选最短的一些边（认为可信度更高）
            std::vector<edge_info>  edge_scores;
            for (const auto& edge : edges) {
                //std::pair<int, int> edge = edges[i];
                pt edge1_pt1 = pts[edge.first];
                pt edge1_pt2 = pts[edge.second];
                double d1 = std::abs(edge1_pt1.x - edge1_pt2.x) + std::abs(edge1_pt1.y - edge1_pt2.y);  //计算两点之间的距离
                edge_scores.push_back({ d1, edge.first, edge.second });
            }
            std::sort(edge_scores.begin(), edge_scores.end(), [](const edge_info edge1, const edge_info edge2) {
                return edge1.score < edge2.score;
                });

            int want_n = min(max_edges, static_cast<int>(edge_scores.size()));
            edges.clear();
            for (int i = 0; i < want_n; i++) {
                edges.push_back({ edge_scores[i].id1, edge_scores[i].id2 });
            }
        }
        // 策略2: 取平均得分高的边
        if (strategy == 2) {
            //计算边的点的平均相似性，筛选较好的一些边（认为可信度更高）
            std::vector<edge_info>  edge_scores;
            for (const auto& edge : edges) {
                pt edge1_pt1 = pts[edge.first];
                pt edge1_pt2 = pts[edge.second];
                double ave_score = (edge1_pt1.score + edge1_pt2.score) / 2;
                edge_scores.push_back({ ave_score, edge.first, edge.second });
            }
            std::sort(edge_scores.begin(), edge_scores.end(), [](const edge_info edge1, const edge_info edge2) {
                return edge1.score > edge2.score;
                });
            int want_n = min(max_edges, static_cast<int>(edge_scores.size()));
            edges.clear();
            for (int i = 0; i < want_n; i++) {
                edges.push_back({ edge_scores[i].id1, edge_scores[i].id2 });
            }
        }

    }
    
    auto end0 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
    printf("create graph %d edges （time: %d ms） \n", edges.size(), duration.count());

    // 构建图形对象
    Graph g;
    g.pts = pts;
    g.edges = edges;
    // 创建igraph图对象
    igraph_t graph;
    igraph_vector_t  v;  //向量类
    igraph_vector_init(&v, edges.size() * 2);  //初始化向量，大小为edges.size() * 2
    for (int i = 0; i < edges.size(); i++) {
        std::pair<int, int>& edge = edges[i];
        VECTOR(v)[i * 2] = edge.first;  //起始顶点索引
        VECTOR(v)[i * 2 + 1] = edge.second;  //结束顶点索引
    }
    igraph_create(&graph, (igraph_vector_t*)&v, 0, IGRAPH_UNDIRECTED);  //创建无向图 （边，顶点数量自动计算，无向图）
    g.graph = graph;
    return g;
    printf("create graph g!!! \n");
}


// 地理位置空间检索 批量
void RS_cliques() {
    float res = compute_res(15);  //4.777; 查询图像的分辨率
    const char* csv_dir = "./output/hits";
    const char* target_dir = "./output/cli";
    std::vector<std::string> csv_files = get_csv_files(csv_dir);
    
    auto start00 = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(start00);
    std::tm* now_tm = std::localtime(&now_c);
    std::cout << "Start time: " << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
    
    auto startaa = std::chrono::steady_clock::now();  //计时
    int TIME_sum = 0;
    for (const std::string& file : csv_files) {
        const char* csv_path = file.c_str();
        std::cout << file << std::endl;
        
        // 判断是否跳过
        std::string output_csv_str = generate_output_csv(csv_path, target_dir);
        if (file_exists(output_csv_str)) {
            std::cout << "clique csv exists " << std::endl;
            continue;
        }

        auto startx = std::chrono::steady_clock::now();  //计时

        int pt_edge_max = 1;  //单点 构建边的最大数量（偏差最小的点对）
        int max_edges = 100000;  //筛选边的数量
        int strategy = 1;  //0:不筛选  1:最短边  2:平均相似性分数最高   边太少无效
        Graph g = create_graph(csv_path, res, pt_edge_max, max_edges, strategy);  // 创建图形对象
    
        int max_cli = 1000;  //合并子团后1000
        int conbine = 1;  // 最大输出数量（去除子团后）
        find_clipque(g, max_cli, conbine,0.6);  // 找到团

        auto endx = std::chrono::steady_clock::now();  //计时
        
        auto end_system = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endx - startx);
        printf("time: %d ms;    find clique %s at %s\n", duration.count(), file.c_str(), std::ctime(&end_system));
        TIME_sum += duration.count();
        
        write_clique_to_hit(g, csv_path, target_dir);  // 将团 返回为检索信息
        //write_clique_to_shpfile(csv_path, target_dir, g, 500);  // 将图形对象的几何形状写入到Shapefile文件中
        
    }
    printf("total TRUE time: %d ms \n", TIME_sum);
    auto endaa = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endaa - startaa);
    printf("total time: %d ms \n", duration.count());
}


int main(int argc, char* argv[]){
    setenv("TZ", "Asia/Shanghai", 1);

    RS_cliques();

    return 0;
}
