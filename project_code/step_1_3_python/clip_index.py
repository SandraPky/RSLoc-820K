from multiprocessing import Process, Queue
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import sqlite3
from torchvision.transforms import transforms
from torchvision.io import decode_image,ImageReadMode
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

db_path = './Gallery/db1.db'
model_path = "./project_code/model/clip-vit-large-patch14"

def extract_db_image_expand(connection,level,col,row):
    trans_image = torch.ones(3,256*3, 256*3) * 255
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            cursor = connection.cursor()  # 游标
            clevel = level
            ccol = col +j
            crow = row +i
            sql = f"select rowid,tile_data  from ge_tiles where zoom_level = {clevel} and tile_column = {ccol} and tile_row = {crow}"
            cursor.execute(sql)
            rows = cursor.fetchmany(1)
            if len(rows) == 0:
                continue
            rowid, tile_data = rows[0]
            nparr = torch.frombuffer(tile_data, dtype=torch.uint8)
            image = decode_image(nparr, ImageReadMode.RGB)
            trans_image[:,(i+1)*256:(i+2)*256,(j+1)*256:(j+2)*256] = image[:, :, :]
    return trans_image

transform2 = transforms.Compose([
        transforms.Resize([224,224], antialias=True),
        transforms.Lambda(lambda x: x.float().div(255)),
        transforms.Normalize(( 0.48145466,  0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def extract_feature(model, org_img, device,processor):
    with torch.no_grad():
        img = org_img.to(device)
        img = transform2(img).unsqueeze(0)
        image = img
        image_features_g = model.get_image_features(image)
        return image_features_g.view(-1).cpu().numpy()

def extract_feature_clip(model, image_path, device,processor):
    tile_data = read_into_buffer(image_path)
    nparr = torch.frombuffer(tile_data, dtype=torch.uint8)
    org_img = decode_image(nparr, ImageReadMode.RGB)
    inputs = processor(images=org_img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features_g = model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features_g, p=2, dim=1)
        return image_features.view(-1).cpu().numpy()

def add_vector_to_index(vector,rowid,level,col,row,time,client):
    vector = vector.tolist()
    pt = PointStruct(
        id=rowid,
        vector=vector,
        payload={"level": level,"col":col,"row":row,"time":time}
    )
    res = client.upsert(
        collection_name="test_geosearch_clip_v2_test",
        points=[
            pt
        ]
    )
    er = 3

def read_into_buffer(filename):
    buf = bytearray(os.path.getsize(filename))
    with open(filename, 'rb') as f:
        f.readinto(buf)
    return buf

def extract_file_feature(model,device,image_path):
    with torch.no_grad():
        tile_data = read_into_buffer(image_path)
        nparr = torch.frombuffer(tile_data, dtype=torch.uint8)
        org_img = decode_image(nparr, ImageReadMode.RGB)
        img = org_img.to(device)
        img = transform2(img).unsqueeze(0)
        image = img
        image_features_g = model.get_image_features(image)
        return image_features_g.view(-1).cpu().numpy()

def index_process_process(q,cpuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = cpuid
    print('Process(%s) is reading...' % os.getpid())
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).to(device)
    client = QdrantClient(host="[server address]", port=6333)

    while True:
        image_path_id = q.get(True)
        if image_path_id is None:
            break
        else:
            try:
                rowid,tile_data,level,col,row,time = image_path_id
                clip_features = extract_feature(model, tile_data, device, processor)
                add_vector_to_index(clip_features, rowid,level,col,row,time, client)
            except Exception as e:
                print(e)

def dispatch_images():
    collection_name = "geosearch_clipvitl14_index"
    client = QdrantClient(host="[server address]", port=6333)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()  # 游标
    cursor2 = connection.cursor()  # 游标
    cursor2.execute("create index if not exists level_col_row_index on ge_tiles(zoom_level,tile_column,tile_row)")

    sql = "select count(*) from ge_tiles"
    cursor.execute(sql)
    rows = cursor.fetchone()
    tile_num = 0
    for row in rows:
        tile_num = row
    print(tile_num)

    q = Queue(200)
    readers = []
    for i in range(5):
        for j in range(1):
            _reader = Process(target=index_process_process, args=(q,str(i)))
            _reader.start()
            readers.append(_reader)

    sql = "select rowid,zoom_level,tile_column,tile_row,time  from ge_tiles"
    cursor.execute(sql)
    rows = cursor.fetchmany(100)
    finish_row = 0
    while rows is not None:
        if len(rows) == 0:
            break
        for row in rows:
            rowid,zoom_level, tile_column, tile_row, time = row
            tile_data = extract_db_image_expand(connection, zoom_level, tile_column, tile_row)
            q.put((rowid,tile_data,zoom_level, tile_column, tile_row, time))
            finish_row = finish_row + 1
            print(f"finsh{finish_row}/{tile_num}")
        rows = cursor.fetchmany(100)

    connection.close()
    for i in range(len(readers)):
        q.put(None)
    for _reader in readers:
        _reader.join()

if __name__=='__main__':
    dispatch_images()


