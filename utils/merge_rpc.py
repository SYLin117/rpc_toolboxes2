import json
import pycocotools
import os
import shutil
from collections import OrderedDict, defaultdict
from tqdm import tqdm


def sort_ann_by_img(json_path):
    with open(json_path, 'r', encoding="utf-8") as fid:
        data = json.load(fid)
    images = OrderedDict()
    for x in data['images']:
        images[x['id']] = x
    annotations = defaultdict(list)
    for x in data['annotations']:
        annotations[x['image_id']].append(x)
    return annotations, images


# 提取一些基本訊息
demo_json_path = r"D:\datasets\retail_product_checkout\instances_train2019.json"
# demo_json_path = r"D:\datasets\tw_rpc\train2019.json"

with open(demo_json_path) as fid:
    demo_data = json.load(fid)

json_list = [
    r"D:\datasets\retail_product_checkout\instances_val2019.json",
    r"D:\datasets\rpc_list3\synthesize_15000_train1.json",
    r"D:\datasets\rpc_list3\synthesize_15000_train2(cyclegan).json"
]
img_list = [
    r"D:\datasets\retail_product_checkout\val2019",
    r"D:\datasets\rpc_list3\synthesize_15000_train1",
    r"D:\datasets\rpc_list3\synthesize_15000_train2(UNIT)"
]
# save_dir_root = r'D:\datasets\rpc_list3\combine'
save_dir_root = r'D:\datasets\rpc_list3\combine'
save_folder = 'val_15000_15000(unit)'
save_path = os.path.join(save_dir_root, save_folder)

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

new_json = defaultdict()
new_images = list()
new_annotations = list()
new_image_idx = 1
new_ann_idx = 1
assert len(json_list) == len(img_list)
for i, (json_path, img_folder) in enumerate(zip(json_list, img_list)):
    assert os.path.exists(json_path)
    annotations, imgs = sort_ann_by_img(json_path)
    for img_id, anns in tqdm(annotations.items()):
        if not os.path.exists(os.path.join(img_folder, imgs[img_id]['file_name'])):
            continue
        shutil.copyfile(os.path.join(img_folder, imgs[img_id]['file_name']),
                        os.path.join(save_path, f"{new_image_idx}.jpg"))
        img_tmp = imgs[img_id]
        img_tmp['id'] = new_image_idx
        img_tmp['file_name'] = f"{new_image_idx}.jpg"
        new_images.append(img_tmp)
        for ann in anns:
            ann['image_id'] = new_image_idx
            ann['id'] = new_ann_idx
            new_annotations.append(ann)
            new_ann_idx += 1
        new_image_idx += 1

new_json['categories'] = demo_data['categories']
new_json['images'] = new_images
new_json['annotations'] = new_annotations
with open(os.path.join(save_dir_root, save_folder + '.json'), 'w') as fid:
    json.dump(new_json, fid)
