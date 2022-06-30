import os
import glob
import exifread
import datetime
import shutil
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import os.path as ospath
import random
import itertools
from multiprocessing import Pool, freeze_support
from p_tqdm import p_map
import pandas as pd

def single_rename(rootDir: str, saveDir: str):
    """
    對單商品照片檔案重新命名
    資料夾結構
    root
        001_drink
            ...
            ...
        002_drink
    :param type: s(single) or c(checkout)
    :param dir: 跟目錄
    :return:
    """
    os.makedirs(saveDir, exist_ok=True)
    images_path = []
    for root, dirs, files in os.walk(rootDir):
        if files:
            file_type = ospath.basename(root)
        for file in files:
            if file.endswith('.JPG'):
                f = open(ospath.join(root, file), 'rb')
                exif_tag = exifread.process_file(f)
                datetime_obj = datetime.datetime.strptime(exif_tag['EXIF DateTimeOriginal'].values, '%Y:%m:%d %H:%M:%S')
                timestamp_str = str(int(datetime_obj.timestamp()))
                new_file_name = 's_' + file_type + '_' + timestamp_str + '.jpg'
                images_path.append(file)
                shutil.copyfile(ospath.join(rootDir, root, file), ospath.join(saveDir, new_file_name))
    # print(images_path)


def checkout_rename(rootDir: str, saveDir: str):
    """
    對checkout照片檔案重新命名
    :param dir: 跟目錄
    :return:
    """
    os.makedirs(saveDir, exist_ok=True)
    files = glob.glob(ospath.join(rootDir, '*.JPG'))
    for file in files:
        f = open(file, 'rb')
        exif_tag = exifread.process_file(f)
        datetime_obj = datetime.datetime.strptime(exif_tag['EXIF DateTimeOriginal'].values, '%Y:%m:%d %H:%M:%S')
        timestamp_str = str(int(datetime_obj.timestamp()))
        new_file_name = 'c_' + timestamp_str + '.jpg'  # 使用timestamp重新命名
        shutil.copyfile(file, ospath.join(saveDir, new_file_name))


def save_bbox_image(imgDir: str, jsonDir: str, saveBboxDir: str, saveMaskDir: str):
    """
    將bbox的範圍儲存成影像 (對train2019使用)
    :param imgDir:
    :param jsonDir:
    :param saveDir:
    :return:
    """
    os.makedirs(saveBboxDir, exist_ok=True)
    os.makedirs(saveMaskDir, exist_ok=True)
    coco = COCO(jsonDir)
    with open(jsonDir, 'r') as f:
        jsonData = json.load(f)
        anns = jsonData['annotations']
        imgs = jsonData['images']

        anns = sorted(anns, key=lambda i: (i['image_id'], i['id']))

        image_id_2_filename = {img['id']: img['file_name'] for img in imgs}

        current_image_id = None
        img_cv = None
        ann_idx = 1
        for ann in tqdm(anns):
            if current_image_id is None:
                img_cv = cv2.imread(ospath.join(imgDir, image_id_2_filename[ann['image_id']]))
            elif current_image_id != ann['image_id']:
                img_cv = cv2.imread(ospath.join(imgDir, image_id_2_filename[ann['image_id']]))
            current_image_id = ann['image_id']
            bbox = ann['bbox']
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = x0 + int(bbox[2])
            y1 = y0 + int(bbox[3])
            crop_img = img_cv[y0:y1, x0:x1, :]
            cv2.imwrite(ospath.join(saveBboxDir, image_id_2_filename[ann['image_id']]), crop_img)

            annIds = coco.getAnnIds(imgIds=current_image_id)
            anns = coco.loadAnns(annIds)
            mask = coco.annToMask(anns[0])
            # crop_mask = mask[y0:y1, x0:x1]
            cv2.imwrite(ospath.join(saveMaskDir, ospath.splitext(image_id_2_filename[ann['image_id']])[0] + ".png"),
                        mask * 255)

            ann_idx += 1


def split_checkout_images(imgDir: str):
    random.seed(42)
    files = glob.glob(ospath.join(imgDir, '*.jpg'))
    parentDir = os.path.abspath(os.path.join(imgDir, '..'))
    valDir = ospath.join(parentDir, 'val2019')
    testDir = ospath.join(parentDir, 'test2019')
    random.shuffle(files)
    mid = 300
    first_half = files[:mid]
    second_half = files[mid:]
    if valDir:
        if ospath.exists(valDir):
            shutil.rmtree(valDir)
        os.makedirs(valDir)
        with Pool(6) as pool:
            a = list(zip(itertools.repeat(valDir), first_half))
            pool.map(copy_img, a)
    if testDir:
        if ospath.exists(testDir):
            shutil.rmtree(testDir)
        os.makedirs(testDir)
        with Pool(6) as pool:
            a = list(zip(itertools.repeat(testDir), second_half))
            pool.map(copy_img, a)


def copy_img(data):
    """
    data[0]: save root path
    data[1]: file path
    """
    shutil.copyfile(data[1], ospath.join(data[0], ospath.basename(data[1])))


def split_checkout_half(imgDir: str, saveDir1: str = None, saveDir2: str = None):
    """
    將資料夾中的影像拆成兩半
    :param imgDir:
    :param saveDir1:
    :param saveDir2:
    :return:
    """
    import time
    random.seed(42)

    files = glob.glob(ospath.join(imgDir, '*.jpg'))
    random.shuffle(files)
    mid = int(len(files) / 2)
    first_half = files[:mid]
    second_half = files[mid:]
    start = time.time()
    if saveDir1:
        if ospath.exists(saveDir1):
            shutil.rmtree(saveDir1)
        os.makedirs(saveDir1)
        with Pool(6) as pool:
            a = list(zip(itertools.repeat(saveDir1), first_half))
            pool.map(copy_img, a)

    if saveDir2:
        if ospath.exists(saveDir2):
            shutil.rmtree(saveDir2)
        os.makedirs(saveDir2)
        with Pool(6) as pool:
            a = list(zip(itertools.repeat(saveDir2), second_half))
            pool.map(copy_img, a)
    print('cost time: {}'.format(time.time() - start))


def split_twrpc_json(jsonDir: str):
    """
    將json資料依照train2019 val2019 test2019分類
    :param jsonDir:
    :return:
    """
    with open(jsonDir) as f:
        jsonData = json.load(f)
    parentDir = os.path.abspath(os.path.join(jsonDir, '..'))
    saveTrainDir = ospath.join(parentDir, 'train2019.json')
    saveValDir = ospath.join(parentDir, 'val2019.json')
    saveTestDir = ospath.join(parentDir, 'test2019.json')

    trainFiles = glob.glob(ospath.join(parentDir, 'train2019', '*.jpg'))
    trainFiles = [ospath.basename(i) for i in trainFiles]

    valFiles = glob.glob(ospath.join(parentDir, 'val2019', '*.jpg'))
    valFiles = [ospath.basename(i) for i in valFiles]

    testFiles = glob.glob(ospath.join(parentDir, 'test2019', '*.jpg'))
    testFiles = [ospath.basename(i) for i in testFiles]

    annotations = jsonData['annotations']
    images = jsonData['images']
    new_annotations = []
    new_images = []

    trainImageId = []
    for image in images:
        if image['file_name'] in trainFiles:
            new_images.append(image)
            trainImageId.append(image['id'])
    for ann in annotations:
        if ann['image_id'] in trainImageId:
            new_annotations.append(ann)

    saveTrainJson = {}
    saveTrainJson['categories'] = jsonData['categories']
    saveTrainJson['images'] = new_images
    saveTrainJson['annotations'] = new_annotations
    with open(saveTrainDir, "w") as outfile:
        json.dump(saveTrainJson, outfile, )

    new_images.clear()
    new_annotations.clear()

    valImageId = []
    for image in images:
        if image['file_name'] in valFiles:
            new_images.append(image)
            valImageId.append(image['id'])
    for ann in annotations:
        if ann['image_id'] in valImageId:
            new_annotations.append(ann)

    saveValJson = {}
    saveValJson['categories'] = jsonData['categories']
    saveValJson['images'] = new_images
    saveValJson['annotations'] = new_annotations
    with open(saveValDir, "w") as outfile:
        json.dump(saveValJson, outfile, )

    new_images.clear()
    new_annotations.clear()
    testImageId = []
    for image in images:
        if image['file_name'] in testFiles:
            new_images.append(image)
            testImageId.append(image['id'])
    for ann in annotations:
        if ann['image_id'] in testImageId:
            new_annotations.append(ann)
    saveTestJson = {}
    saveTestJson['categories'] = jsonData['categories']
    saveTestJson['images'] = new_images
    saveTestJson['annotations'] = new_annotations
    with open(saveTestDir, "w") as outfile:
        json.dump(saveTestJson, outfile, )


def split_twrpc_json_by_folder(jsonDir: str, imgDir1: str, saveJson1: str, imgDir2: str, saveJson2: str):
    """
    將json資料依照train2019 val2019 test2019分類
    :param jsonDir:
    :return:
    """
    with open(jsonDir) as f:
        jsonData = json.load(f)

    firstFiles = glob.glob(ospath.join(imgDir1, '*.jpg'))
    firstFiles = [ospath.basename(i) for i in firstFiles]

    annotations = jsonData['annotations']
    images = jsonData['images']

    new_annotations = []
    new_images = []
    firstImageId = []
    for image in images:
        if image['file_name'] in firstFiles:
            new_images.append(image)
            firstImageId.append(image['id'])
    for ann in annotations:
        if ann['image_id'] in firstImageId:
            new_annotations.append(ann)

    secondFiles = glob.glob(ospath.join(imgDir2, '*.jpg'))
    secondFiles = [ospath.basename(i) for i in secondFiles]
    saveFirstJson = {}
    saveFirstJson['categories'] = jsonData['categories']
    saveFirstJson['images'] = new_images
    saveFirstJson['annotations'] = new_annotations
    with open(saveJson1, "w") as outfile:
        json.dump(saveFirstJson, outfile, )

    new_images.clear()
    new_annotations.clear()
    secondImageId = []
    for image in images:
        if image['file_name'] in secondFiles:
            new_images.append(image)
            secondImageId.append(image['id'])
    for ann in annotations:
        if ann['image_id'] in secondImageId:
            new_annotations.append(ann)

    saveSecondJson = {}
    saveSecondJson['categories'] = jsonData['categories']
    saveSecondJson['images'] = new_images
    saveSecondJson['annotations'] = new_annotations
    with open(saveJson2, "w") as outfile:
        json.dump(saveSecondJson, outfile, )


def delete_duplecate_download(rootDir: str):
    """
    刪除重複下載的檔案
    :param rootDir:
    :return:
    """
    files = glob.glob(os.path.join(rootDir, "*(*)*.jpg"))
    for file in files:
        os.remove(file)


def coco_shrink_data(json_path, save_path, ratio, k):
    """
    縮減json資料
    :param json_path:
    :param save_path:
    :param ratio: remove portion
    :return:
    """
    with open(json_path, 'r') as fid:
        data = json.load(fid)
    assert data is not None
    annotations = data['annotations']
    images = data['images']
    new_annotations = list()
    new_images = list()
    # remain_images = random.choices(images, k=int(len(images) * ratio) if k is None else k)
    remain_images = random.sample(images, int(len(images) * ratio) if k is None else k)
    remain_images = sorted(remain_images, key=lambda x: x['id'])
    remain_img_ids = [i['id'] for i in remain_images]
    for i in range(len(images)):
        if images[i]['id'] in remain_img_ids:
            new_images.append(images[i])
    for ann in annotations:
        if ann['image_id'] in remain_img_ids:
            new_annotations.append(ann)
    new_data = {}
    new_data['categories'] = data['categories']
    new_data['annotations'] = new_annotations
    new_data['images'] = new_images
    with open(save_path, "w") as outfile:
        json.dump(new_data, outfile, )


def resize_images(files: list, target_size: int, interpolation: int, target_folder: str = None, isMask: bool = False):
    """
    Args:
        files: list of images path
        target_size: int
        interpolation: cv2.INTER_...

    Returns:
    """
    if target_folder:
        os.makedirs(target_folder, exist_ok=True)
    if not isMask:
        for file in tqdm(files):
            filename = os.path.basename(file)
            img_cv = cv2.imread(file, cv2.IMREAD_COLOR)
            img_cv = cv2.resize(img_cv, (target_size, target_size), interpolation=interpolation)
            if not target_folder:
                cv2.imwrite(file, img_cv)
            else:
                cv2.imwrite(os.path.join(target_folder, filename), img_cv)
    else:
        for file in tqdm(files):
            filename = os.path.basename(file)
            img_cv = cv2.imread(file, cv2.IMREAD_COLOR)
            img_cv = cv2.resize(img_cv, (target_size, target_size), interpolation=interpolation)
            # kernel =
            # img_cv = skimage.transform.resize(img_cv,
            #                                   (target_size, target_size),
            #                                   mode='edge',
            #                                   anti_aliasing=False,
            #                                   anti_aliasing_sigma=None,
            #                                   preserve_range=True,
            #                                   order=0)
            if not target_folder:
                cv2.imwrite(file, img_cv)
            else:
                cv2.imwrite(os.path.join(target_folder, filename), img_cv)


def rescale_coco_data(source_image_folder: str,
                      source_mask_folder: str = None,
                      source_json_file: str = None,
                      target_size: int = 512,
                      target_image_folder: str = None,
                      target_mask_folder: str = None,
                      target_json_file: str = None,
                      resize_file: bool = False):
    """
    rescale images, masks and bounding box in json file of coco style data
    after small the dataset
    could use rpc_mask_2_coco.py convert mask to seg datas
    Args:
        image_folder: path of images folder
        mask_folder:  path of masks folder
        json_file: path of json file
    Returns: None
    """

    our_coco = COCO(source_json_file)
    images = glob.glob(os.path.join(source_image_folder, "*.jpg"))
    if source_mask_folder:
        masks = glob.glob(os.path.join(source_mask_folder, "*.png"))
    with open(source_json_file) as fid:
        json_data = json.load(fid)
    new_anns = []
    ann_idx = 0
    imgToAnns = our_coco.imgToAnns
    for k, v in tqdm(imgToAnns.items()):  # k: img_id, v: anns
        width = our_coco.imgs[k]['width']
        height = our_coco.imgs[k]['height']
        width_scale = target_size / width
        height_scale = target_size / height
        for i, ann in enumerate(v):
            new_anns.append({
                'bbox': (
                    int(ann['bbox'][0] * width_scale),
                    int(ann['bbox'][1] * height_scale),
                    int(ann['bbox'][2] * width_scale),
                    int(ann['bbox'][3] * height_scale)),
                'category_id': ann['category_id'],
                'area': int(ann['bbox'][2] * width_scale) * int(ann['bbox'][3] * height_scale),
                'center_of_mass': (int(ann['bbox'][0] * width_scale + ann['bbox'][2] * width_scale / 2),
                                   int(ann['bbox'][1] * height_scale + ann['bbox'][3] * height_scale / 2)),
                'image_id': k,
                'iscrowd': 0,
                'id': ann_idx
            })
            ann_idx += 1

    if resize_file:
        print("... resize images ...")
        resize_images(images, target_size, cv2.INTER_CUBIC, target_image_folder)
        print("... resize mask ...")
        if source_mask_folder is not None:
            resize_images(masks, target_size, cv2.INTER_NEAREST_EXACT, target_mask_folder, isMask=True)

    json_images = json_data['images']
    for img in json_images:
        img['width'] = target_size
        img['height'] = target_size
    json_data['images'] = json_images
    json_data['annotations'] = new_anns
    with open(target_json_file, 'w') as fid:
        json.dump(json_data, fid)



if __name__ == "__main__":
    print()

    # single_rename(rootDir=r"D:\datasets\tw_rpc\products", saveDir=r'D:\datasets\tw_rpc\train2019')

    # checkout_rename(rootDir=r"D:\datasets\tw_rpc\new_checkout_original", saveDir=r'D:\datasets\tw_rpc\checkout')

    # save_bbox_image(imgDir=r'D:\datasets\tw_rpc\train2019', jsonDir=r'D:\datasets\tw_rpc\train2019.json',
    #                 saveBboxDir=r'D:\datasets\tw_rpc\train2019_crop', saveMaskDir=r'D:\datasets\tw_rpc\train2019_mask')

    # coco_shrink_data(r"D:\datasets\tw_rpc\val2019.json", r"D:\datasets\tw_rpc\val2019_100.json", None, 100)
    # delete_duplecate_download(r'C:\Users\newia\Downloads\G 奶甜蜜女友郭鬼鬼躺在你身邊')

    ## 切割資料
    # split_checkout_images(r'D:\datasets\tw_rpc\checkout')
    split_twrpc_json(r'D:\datasets\tw_rpc\tw_rpc-4.json')

    # split_checkout_half(r"D:\datasets\rpc_list\synthesize_100000_train",
    #                     r'D:\datasets\rpc_list3\synthesize_50000_train1',
    #                     r'D:\datasets\rpc_list3\synthesize_50000_train2')

    # split_twrpc_json_by_folder(r"D:\datasets\rpc_list\synthesize_30000_train.json",
    #                            r"D:\datasets\rpc_list3\synthesize_15000_train1",
    #                            r"D:\datasets\rpc_list3\synthesize_15000_train1.json",
    #                            r"D:\datasets\rpc_list3\synthesize_15000_train2",
    #                            r"D:\datasets\rpc_list3\synthesize_15000_train2.json")

    # name = 'synthesize_15000_train2'
    # rescale_coco_data(source_image_folder=r"D:\datasets\rpc_list3\{}".format(name),
    #                   source_mask_folder=None,
    #                   source_json_file=r"D:\datasets\rpc_list3\{}.json".format(name),
    #                   target_size=512,
    #                   target_image_folder=r'D:\datasets\rpc_list3\{}'.format(name),
    #                   target_mask_folder=None,
    #                   target_json_file=r'D:\datasets\rpc_list3\{}(cyclegan).json'.format(name),
    #                   resize_file=False)
