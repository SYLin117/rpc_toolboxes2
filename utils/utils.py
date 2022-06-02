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
    將bbox的範圍儲存成影像
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
    files = glob.glob(ospath.join(imgDir, '*.jpg'))
    parentDir = os.path.abspath(os.path.join(imgDir, '..'))
    valDir = ospath.join(parentDir, 'val2019')
    testDir = ospath.join(parentDir, 'test2019')
    for file in files:
        val_or_test = random.randint(1, 2)
        if val_or_test == 1:
            shutil.copyfile(file, ospath.join(valDir, ospath.basename(file)))
        else:
            shutil.copyfile(file, ospath.join(testDir, ospath.basename(file)))


def split_twrpc_json(jsonDir: str):
    """
    將json資料依照val2019 test2019分類
    :param jsonDir:
    :return:
    """
    with open(jsonDir) as f:
        jsonData = json.load(f)
    parentDir = os.path.abspath(os.path.join(jsonDir, '..'))
    saveValDir = ospath.join(parentDir, 'val2019.json')
    saveTestDir = ospath.join(parentDir, 'test2019.json')

    valFiles = glob.glob(ospath.join(parentDir, 'val2019', '*.jpg'))
    valFiles = [ospath.basename(i) for i in valFiles]

    testFiles = glob.glob(ospath.join(parentDir, 'test2019', '*.jpg'))
    testFiles = [ospath.basename(i) for i in testFiles]

    annotations = jsonData['annotations']
    images = jsonData['images']
    new_annotations = []
    new_images = []

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


if __name__ == "__main__":
    print()

    # rename_images_by_time(rootDir=r"D:\datasets\my_rpc\products", saveDir=r'D:\datasets\my_rpc\train2019')

    # checkout_rename(rootDir=r"D:\datasets\tw_rpc\chechout_original", saveDir=r'D:\datasets\tw_rpc\checkout')

    # save_bbox_image(imgDir=r'D:\datasets\my_rpc\train2019', jsonDir=r'D:\datasets\my_rpc\train2019.json',
    #                 saveBboxDir=r'D:\datasets\my_rpc\train2019_crop', saveMaskDir=r'D:\datasets\my_rpc\train2019_mask')

    # split_checkout_images(r'D:\datasets\tw_rpc\checkout')
    split_twrpc_json(r'D:\datasets\tw_rpc\tw_rpc.json')
