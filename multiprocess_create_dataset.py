# ==========================
# 使用multithread製作偽造影像
# ==========================
import glob
import json
import os
import random
from random import randrange
import scipy
import scipy.spatial as T
import time
from argparse import ArgumentParser
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from tqdm import tqdm
import pathlib
import sys
import matplotlib.pyplot as plt
import multiprocessing
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gc
import traceback
from scipy.stats import truncnorm
import math


def get_truncated_normal(mean=0.25, sd=0.05, low=0.0, upp=1.0):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


CATEGORIES = ['__background__', '001_drink', '002_drink', '003_chocolate', '004_chocolate', '005_candy', '006_candy',
              '007_puffed_food', '008_puffed_food', '009_tissue', '010_tissue']

np.random.seed(42)
CAT_COLORS = (1 - (np.random.rand(len(CATEGORIES) + 1, 3)) * 255).astype(np.uint8)
CAT_COLORS[0, :] = [0, 0, 0]
SHADOW_COLOR = [0x393433, 0x2a2727]


def buy_strategic(counter):
    # 回傳每張圖片有幾個類別與物件
    global NUM_CATEGORIES
    categories = [i + 1 for i in range(NUM_CATEGORIES)]
    diff_list = [1, 2, 3]
    dist = [.7, .2, .1]
    difficulty = random.choices(diff_list, weights=dist, k=1, )[0]
    num_categories = 1
    if difficulty == 1:
        num_categories = random.randint(1, 4)
        counter['easy_mode'] += 1
    elif difficulty == 2:
        num_categories = random.randint(4, 7)
        counter['medium_mode'] += 1
    elif difficulty == 3:
        num_categories = random.randint(8, 10)
        counter['hard_mode'] += 1
    num_per_category = {}
    selected_categories = np.random.choice(categories, size=num_categories, replace=False)

    for category in selected_categories:
        count = random.randint(1, 2)
        num_per_category[int(category)] = count
    return num_per_category, difficulty


def check_iou(annotations, box, threshold=0.5):
    """
    Args:
        annotations:
        box: (x, y, w, h)
        threshold:
    Returns: bool
    """

    cx1, cy1, cw, ch = box
    cx2, cy2 = cx1 + cw, cy1 + ch
    carea = cw * ch  # new object
    for ann in annotations:
        x1, y1, w, h = ann['bbox']
        x2, y2 = x1 + w, y1 + h
        area = w * h  # object in ann
        inter_x1 = max(x1, cx1)
        inter_y1 = max(y1, cy1)
        inter_x2 = min(x2, cx2)
        inter_y2 = min(y2, cy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        iou = inter_area / (carea + area - inter_area + 1e-8)  # avoid division by zero

        iou1 = inter_area / (carea + 1e-8)  # 重疊區域佔舊object的比例
        iou2 = inter_area / (area + 1e-8)  # 重疊區域佔新object的比例

        if iou > threshold or iou1 > threshold or iou2 > threshold:
            return False
    return True


def sample_select_object_index(paths):
    """
    隨機回傳該類別其中一個物件
    """

    path = random.choices(paths, weights=None, k=1)[0]
    return path


def generated_position(width, height, w, h, padx=0, pady=0):
    x = random.randint(padx, width - w - padx)
    y = random.randint(pady, height - h - pady)
    while x + w > width:
        x = random.randint(padx, width - w - padx)
    while y + h > height:
        y = random.randint(pady, height - h - pady)
    return x, y


def get_object_bbox(annotation, max_width, max_height):
    bbox = annotation['bbox']
    x, y, w, h = [int(x) for x in bbox]

    # box_pad = max(160, int(max(w, h) * 0.3))
    box_pad = 5
    crop_x1 = max(0, x - box_pad)
    crop_y1 = max(0, y - box_pad)
    crop_x2 = min(x + w + box_pad, max_width)
    crop_y2 = min(y + h + box_pad, max_height)
    x = crop_x1
    y = crop_y1
    w = crop_x2 - crop_x1
    h = crop_y2 - crop_y1
    return x, y, w, h


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # (x,y)
    leaf_size = 2048
    # build kd tree
    tree = T.KDTree(pts.copy(), leafsize=leaf_size)
    # query kd tree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.085
            sigma = min(sigma, 999)  # avoid inf
        else:
            raise NotImplementedError('should not be here!!')
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def trans_paste(bg_img: np.ndarray, fg_img: np.ndarray, mask: np.ndarray, bbox: tuple, trans: bool):
    pos_x, pos_y, w, h = bbox
    try:
        if trans:  # fg_img: shadow
            shadow_prop = random.uniform(0.3, 0.7)
            bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] = \
                bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * (np.ones_like(mask) - mask) + \
                fg_img * mask * shadow_prop + bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * mask * (1 - shadow_prop)
            # fg_img * mask * shadow_prop + bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * mask * (1 - shadow_prop)
        else:
            bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] = \
                bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * (np.ones_like(mask) - mask) + fg_img * mask
    except ValueError as ve:
        print(str(ve))
    return bg_img


def get_random_pos_neg():
    """
    randomly return 1 or -1
    Returns: 1 or -1

    """
    return 1 if random.random() < 0.8 else -1


def create_image(output_dir, output_dir2, object_category_paths, level_dict, image_id, num_per_category,
                 change_background: bool, train_imgs_mask_dir, annotations, lock: Lock):
    """
    製作合成圖
    :param output_dir: 儲存圖片的位置
    :param output_dir2: 儲存陰影圖片的位置
    :param object_category_paths: 每個類別所對應的檔案list
    :param level_dict: 該圖的難度
    :param image_id: 合成編號
    :param num_per_category: dict儲存每個類別有多少個
    :param change_background: 是否變換背景
    :param train_imgs_mask_dir: 物件mask影像儲存位置
    :param annotations:
    :param lock:
    :return:
    """
    try:
        # ----------------- get background image --------------------
        background_id = random.randint(1, 3)
        if not change_background:
            background_id = 1
        bg_img_cv = cv2.imread('bg{}.jpg'.format(background_id), cv2.IMREAD_COLOR)
        bg_img_cv = cv2.cvtColor(bg_img_cv, cv2.COLOR_BGR2RGB)
        bg_img_cv2 = bg_img_cv.copy()
        bg_height, bg_width = bg_img_cv.shape[:2]
        mask_img_cv = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        # ----------------- get background image --------------------
        img_id_num = image_id.split('_')[2]
        obj_in_this_pic = list()
        for category, count in num_per_category.items():
            category = int(category)
            for _ in range(count):
                paths = object_category_paths[category]

                object_path = sample_select_object_index(paths)

                name = os.path.basename(object_path)
                mask_path = os.path.join(train_imgs_mask_dir, '{}.png'.format(name.split('.')[0]))

                obj = Image.open(object_path)
                mask = Image.open(mask_path).convert('1')
                original_width = obj.width
                original_height = obj.height
                # dense object bbox
                # ---------------------------
                # Crop according to json annotation
                # ---------------------------
                x, y, w, h = get_object_bbox(annotations[name], original_width, original_height)
                obj = obj.crop((x, y, x + w, y + h))
                mask = mask.crop((x, y, x + w, y + h))

                scale_mean = 0.8  # 稍微縮小疫點比較不會一直重疊
                std = 0.03
                low = scale_mean - 3 * std
                up = scale_mean + 3 * std
                scale = get_truncated_normal(mean=scale_mean, sd=std, low=low, upp=up).rvs()
                while scale <= 0:
                    scale = get_truncated_normal(mean=scale_mean, sd=std, low=low, upp=up).rvs()
                w, h = int(w * scale), int(h * scale)
                obj = obj.resize((w, h), resample=Image.BILINEAR)
                mask = mask.resize((w, h), resample=Image.BILINEAR)

                # ---------------------------
                # Random rotate
                # ---------------------------
                angle = random.random() * 360
                obj = obj.rotate(angle, resample=Image.BILINEAR, expand=True)
                mask = mask.rotate(angle, resample=Image.BILINEAR, expand=True)

                # ---------------------------
                # Crop according to mask
                # ---------------------------
                where = np.where(np.array(mask))  # value == 255 location
                # where = np.vstack((where[0], where[1]))  ## ian added
                assert len(where[0]) != 0
                assert len(where[1]) != 0
                assert len(where[0]) == len(where[1])
                area = len(where[0])
                y1, x1 = np.amin(where, axis=1)
                y2, x2 = np.amax(where, axis=1)

                obj = obj.crop((x1, y1, x2, y2))
                mask = mask.crop((x1, y1, x2, y2))
                mask_l = mask.convert('L')
                w, h = obj.width, obj.height
                offset = []  # for shadow
                offset.append(np.random.randint(5, 50) * get_random_pos_neg())  # right offset
                offset.append(np.random.randint(10, 50) * get_random_pos_neg())  # down offset

                pos_x, pos_y = generated_position(bg_width, bg_height, w, h, padx=abs(offset[0]), pady=abs(offset[1]))
                start = time.time()
                threshold = 0.2
                while not check_iou(obj_in_this_pic, box=(pos_x, pos_y, w, h), threshold=threshold):
                    if (time.time() - start) > 5:  # cannot find a valid position in 3 seconds
                        start = time.time()
                        threshold += 0.05
                        continue
                    pos_x, pos_y = generated_position(bg_width, bg_height, w, h, padx=abs(offset[0]),
                                                      pady=abs(offset[1]))

                obj_cv = np.array(obj)
                mask_cv = np.array(mask) * 1  # single channel mask
                mask_cv = np.stack((mask_cv, mask_cv, mask_cv), axis=2)  # RGB mask
                blur_mask = mask_cv[:, :, 0].copy() * 255
                blur_mask.astype('float32')
                for i in range(5):
                    blur_mask = cv2.blur(blur_mask, (3, 3), cv2.BORDER_CONSTANT)
                blur_mask = np.divide(blur_mask, np.ones_like(blur_mask).astype('float32') * 255)
                blur_mask = np.stack((blur_mask, blur_mask, blur_mask), axis=2)  # shadow mask
                # blur_mask = cv2.GaussianBlur(blur_mask, (10, 10), 0)
                # blur_mask = cv2.GaussianBlur(blur_mask, (10, 10), 0)
                shadow_indx = randrange(len(SHADOW_COLOR))
                shadow = Image.new('RGB', (w, h), SHADOW_COLOR[shadow_indx])
                shodow_cv = np.array(shadow)

                trans_paste(bg_img_cv, obj_cv, mask_cv, bbox=(pos_x, pos_y, w, h), trans=False)
                # paste shadow
                trans_paste(bg_img_cv2, shodow_cv, blur_mask, bbox=(pos_x + offset[0], pos_y + offset[1], w, h),
                            trans=True)
                trans_paste(bg_img_cv2, obj_cv, mask_cv, bbox=(pos_x, pos_y, w, h), trans=False)

                # ---------------------------
                # Find center of mass
                # ---------------------------
                mask_array = np.array(mask)
                center_of_mass = ndimage.measurements.center_of_mass(mask_array)  # y, x
                center_of_mass = [int(round(x)) for x in center_of_mass]
                center_of_mass = center_of_mass[1] + pos_x, center_of_mass[0] + pos_y  # map to whole image
                if lock:
                    with lock:
                        new_ann = {
                            'bbox': (pos_x, pos_y, w, h),
                            'category_id': category,
                            'center_of_mass': center_of_mass,
                            'area': area,
                            'image_id': int(img_id_num),
                            'iscrowd': 0,
                            'id': ann_idx.value
                        }
                        json_ann.append(new_ann)
                        obj_in_this_pic.append(new_ann)
                        ann_idx.value += 1
        # -------------------------------
        ## save image (mask) and json file
        # -------------------------------
        image_name = '{}.jpg'.format(image_id)
        bg_img = Image.fromarray(bg_img_cv)
        bg_img.save(os.path.join(output_dir, image_name))  # with shadow
        bg_img2 = Image.fromarray(bg_img_cv2)
        bg_img2.save(os.path.join(output_dir2, image_name))  # no shadow

        new_img = {
            'file_name': image_name,
            'id': int(img_id_num),
            'width': bg_width,
            'height': bg_height,
            'level': level_dict[os.path.basename(image_name).split('.')[0]]
        }
        if lock:
            with lock:
                json_img.append(new_img)
        print("\n{} done".format(image_name))
    except Exception as e:
        traceback.print_exc()


def get_object_paths(imgDir):
    object_paths = glob.glob(os.path.join(imgDir, '*.jpg'))
    return object_paths


def init_globals(ann_counter, ann_json, image_json, ):
    global ann_idx, json_ann, json_img
    ann_idx = ann_counter
    json_ann = ann_json
    json_img = image_json


if __name__ == '__main__':
    parser = ArgumentParser(description="Synthesize fake images")
    parser.add_argument('--gen_num', type=int, default=50,
                        help='how many number of images need to create.')
    parser.add_argument('--suffix', type=str, default='test',
                        help='suffix for image folder and json file')
    parser.add_argument('--thread', type=int, default=7,
                        help='using how many thread to create')
    parser.add_argument('--chg_bg', type=bool, default=False,
                        help='use multiple background or not.')

    args = parser.parse_args()
    ###########################################################################################
    NUM_CATEGORIES = 10
    GENERATED_NUM = args.gen_num
    ###########################################################################################
    strategics_name = 'strategics_{}_{}.json'.format(args.gen_num, args.suffix)
    if not os.path.exists(strategics_name):
        counter = {
            'easy_mode': 0,
            'medium_mode': 0,
            'hard_mode': 0
        }
        int_2_diff = {
            1: 'easy',
            2: 'medium',
            3: 'hard'
        }
        level_dict = {}
        strategics = []
        for image_id in tqdm(range(GENERATED_NUM)):
            num_per_category, difficulty = buy_strategic(counter)
            level_dict['synthesized_image_{}'.format(image_id)] = int_2_diff[difficulty]
            strategics.append(('synthesized_image_{}'.format(image_id), num_per_category))
        strategics = sorted(strategics, key=lambda s: s[0])
        save_data = dict()
        save_data['strategics'] = strategics
        save_data['level_dict'] = level_dict
        if os.path.exists(strategics_name):
            os.remove(strategics_name)
        with open(strategics_name, 'w') as f:
            json.dump(save_data, f)

    else:
        with open(strategics_name, 'r') as fid:
            data = json.load(fid)
        strategics = data['strategics']
        strategics = sorted(strategics, key=lambda s: s[0])
        level_dict = data['level_dict']
    strategics = sorted(strategics, key=lambda s: int(os.path.splitext(s[0])[0].split('_')[2]))
    ###########################################################################################

    version = str(GENERATED_NUM)

    output_dir = os.path.join(sys.path[0], 'synthesize', 'synthesize_{}_{}'.format(version, args.suffix))
    os.makedirs(output_dir, exist_ok=True)
    output_dir2 = os.path.join(sys.path[0], 'synthesize', 'synthesize_{}_{}_shadow'.format(version, args.suffix))
    os.makedirs(output_dir2, exist_ok=True)

    DATASET_ROOT = r'D:\datasets\tw_rpc'
    CURRENT_ROOT = str(pathlib.Path().resolve())
    save_json_file = os.path.join(sys.path[0], 'synthesize',
                                  'synthesize_{}_{}.json'.format(version, args.suffix))  # synthesis images json檔案
    train_json = os.path.join(DATASET_ROOT, 'train2019.json')  # rpc的原始train.json
    train_imgs_dir = os.path.join(DATASET_ROOT, 'train2019')  # rpc 的train影像資料夾
    train_imgs_mask_dir = os.path.join(DATASET_ROOT, 'train2019_mask')  # 擷取的mask影像資料夾
    save_mask = False

    with open(train_json) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x
    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    # ---------------------------
    # get image list
    # ---------------------------
    object_paths = get_object_paths(train_imgs_dir)
    # ---------------------------
    # items dict: Key: item_ID, value: item_images(list)
    # ---------------------------
    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)  # store each categories all single images

    m = multiprocessing.Manager()
    lock = m.Lock()
    json_color = m.dict()
    json_ann = m.list()
    json_img = m.list()
    ann_idx = m.Value('i', 1)
    image_left = args.gen_num
    image_cnt = 1
    MAX_JOBS_IN_QUEUE = args.thread
    strategics_iter = iter(strategics)
    # print(strategics)
    jobs = list()
    with ProcessPoolExecutor(max_workers=MAX_JOBS_IN_QUEUE, initializer=init_globals,
                             initargs=(ann_idx, json_ann, json_img,)) as executor:
        results = [executor.submit(create_image, output_dir, output_dir2, object_category_paths, level_dict, image_id,
                                   num_per_category, args.chg_bg, train_imgs_mask_dir, annotations, lock)
                   for image_id, num_per_category in strategics_iter]
    # single core (for debug)
    # results = [create_image(output_dir, output_dir2, object_category_paths, level_dict, image_id, num_per_category,
    #                         args.chg_bg, train_imgs_mask_dir, annotations, None) for image_id, num_per_category in
    #            strategics_iter]
    json_cat = list()
    for idx, val in enumerate(CATEGORIES):
        if idx == 0:
            continue
        name_split = val.split('_', 1)
        id = name_split[0]
        super_cat = name_split[1]
        json_cat.append({'supercategory': super_cat, 'id': int(id), 'name': val})
    new_json = dict()
    new_json['images'] = list(json_img)
    new_json['annotations'] = list(json_ann)
    new_json['categories'] = json_cat
    # new_json['color'] = json_color
    if save_json_file:
        with open(save_json_file, 'w') as fid:
            json.dump(new_json, fid)
