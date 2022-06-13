import argparse
import os
import time
import json
import cv2
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

"""
為測試資料集的影像加上標籤(bbox)(在畫面中)
"""
COCO_CLASSES = ('001_drink', '002_drink', '003_chocolate', '004_chocolate', '005_candy',
                '006_candy', '007_puffed_food', '008_puffed_food', '009_tissue', '010_tissue',)

np.random.seed(42)
_COLORS = np.random.rand(len(COCO_CLASSES), 3)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("visualize coco format dataset!")
    parser.add_argument("--use_default", default=True, help="use default value? true or false")
    parser.add_argument("--img_path", default=None, help="path to images or video")
    parser.add_argument("--json_path", default=None, help="path to json file")
    parser.add_argument("--save_path", default=None, help="path to save file")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def vis(img, box, cls_id, class_names=None):
    cls_id = cls_id - 1

    x0 = int(box[0])
    y0 = int(box[1])
    x1 = x0 + int(box[2])
    y1 = y0 + int(box[3])

    color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
    text = '{}'.format(class_names[cls_id], )
    txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    txt_size = cv2.getTextSize(text, font, 1, 1)[0]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        img,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 1, txt_color, thickness=1)

    return img


def main(args):
    print("__main__")
    Path.mkdir(Path(args.save_path).resolve(), parents=True, exist_ok=True)

    with open(args.json_path, 'rb') as json_file:
        json_data = json.load(json_file)
    imgs = json_data['images']
    anns = json_data['annotations']
    anns = sorted(anns, key=lambda i: (i['image_id'], i['id']))
    image_id_2_filename = {img['id']: img['file_name'] for img in imgs}
    current_image_id = None
    img_cv = None
    # idx = 0
    for ann in anns:
        if current_image_id != ann['image_id']:
            if current_image_id is not None:
                cv2.imwrite(os.path.join(args.save_path, image_id_2_filename[current_image_id]), img_cv)
            current_image_id = ann['image_id']
            img_cv = cv2.imread(os.path.join(args.img_path, image_id_2_filename[current_image_id]))
            vis(img_cv, ann['bbox'], ann['category_id'], COCO_CLASSES)
        else:
            vis(img_cv, ann['bbox'], ann['category_id'], COCO_CLASSES)
        # idx += 1
        # if idx == 10:
        #     break


if __name__ == "__main__":
    # DATASET_DIR = r"D:\datasets\tw_rpc"
    DATASET_DIR = r'D:\PycharmProject\rpc_toolboxes2\synthesize'
    args = make_parser().parse_args()
    if args.use_default:
        args.img_path = os.path.join(DATASET_DIR, 'synthesize_3000_test')
        args.json_path = os.path.join(DATASET_DIR, 'synthesize_3000_test.json')
        args.save_path = os.path.join(DATASET_DIR, 'synthesize_3000_test_labled')

        # args.img_path = os.path.join(DATASET_DIR, 'test2019')
        # args.json_path = os.path.join(DATASET_DIR, 'test2019.json')
        # args.save_path = os.path.join(DATASET_DIR, 'test2019_labeled')

    main(args)
