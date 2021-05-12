'''
    Thanks to: Tiago Miguel Rodrigues de Almeida (https://github.com/tmralmeida)
    The copy borrowed from:
        https://github.com/tmralmeida/bag-of-models
    and then modified.
'''

import os
import sys
import pandas as pd
from pathlib import Path
import copy
import shutil
import random

_SMALL = 32*32
_LARGE = 96*96


def box2d_to_yolo(box2d, IMG_WIDTH=1280, IMG_HEIGHT=720):
    # box2d = [x1 y1 x2 y2]
    x1 = box2d[0] / IMG_WIDTH
    x2 = box2d[2] / IMG_WIDTH
    y1 = box2d[1] / IMG_HEIGHT
    y2 = box2d[3] / IMG_HEIGHT

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return cx, cy, width, height


class BDD100K2YOLO():
    def __init__(self, path, objects):
        self.PATH = path
        self.TRAINING_SIZE = '100k'
        self.PATH_IMAGES = os.path.join(
            self.PATH, 'images/' + self.TRAINING_SIZE)
        self.PATH_LABELS = os.path.join(self.PATH, "labels")

        self.objects = objects

        self.imgs = {}
        self.lbls = {}
        self.labels_file = {}
        self.len_per_mode = {}

        for mode in ['train', 'val']:
            print("-> Opening the JSON file for {} mode.".format(mode))
            self.imgs[mode] = os.listdir(os.path.join(self.PATH_IMAGES, mode))
            self.lbls[mode] = os.path.join(
                self.PATH_LABELS, 'bdd100k_labels_images_' + mode + '.json')
            self.labels_file[mode] = pd.read_json(self.lbls[mode])
            self.len_per_mode[mode] = len(self.labels_file[mode]["name"])

    def check_val_img_id(self, f):
        set_of_idx = dict()
        for idx, img_name in enumerate(self.labels_file['val']["name"]):
            x = f(img_name)
            if x in set_of_idx.keys():
                raise(ValueError(
                    "ID:{} for {} has already recored!!".format(x, img_name)))
            else:
                set_of_idx[x] = img_name
        assert(len(set_of_idx) == self.len_per_mode['val'])

    def convert(self, db_class_name_path, abs_adress=True, round_to_int=False):

        for mode in ['train', 'val']:
            file_handler = open(os.path.join(
                self.PATH_IMAGES, mode, '{}.txt'.format(mode)), 'w')
            for idx, img_name in enumerate(self.labels_file[mode]["name"]):
                lbl_file_row = self.labels_file[mode][self.labels_file[mode].index == idx]
                num_objs = len(lbl_file_row["labels"][idx])

                if abs_adress:
                    data = "{}".format(
                        str(os.path.join(self.PATH_IMAGES, mode, img_name)))
                else:
                    data = "{}".format(str(os.path.join(mode, img_name)))

                have_seen_somthing = False
                for i in range(num_objs):
                    object_ = lbl_file_row['labels'][idx][i]
                    if object_['category'] in self.objects.values() and 'box2d' in object_:
                        have_seen_somthing = True
                        coordinates = object_['box2d']
                        lbl_mapped = list(self.objects.keys())[list(
                            self.objects.values()).index(object_['category'])]
                        x1, y1, x2, y2 = (coordinates['x1'], coordinates['y1'], coordinates['x2'], coordinates['y2']) if not round_to_int else (
                            round(coordinates['x1']), round(coordinates['y1']), round(coordinates['x2']), round(coordinates['y2']))
                        data = "{} {},{},{},{},{}".format(
                            data, x1, y1, x2, y2, lbl_mapped)
                if have_seen_somthing:
                    file_handler.writelines("{}\n".format(data))
                else:
                    print("{} is ignored.".format(img_name))
                print("-> Creating {}.txt: {:.2f}%".format(mode,
                                                           ((idx + 1)*100/self.len_per_mode[mode])), end='\r')
            print()
            file_handler.close()
        print('-> Writing bdd100k.name...')
        f = open(os.path.join(db_class_name_path, "bdd100k.name"), 'w')
        for k in self.objects.values():
            f.write("{}\n".format(k))
        f.close()

    def find_images(self, scene, time_of_day, folder_path, how_many=10):
        list_of_images = []
        print('-> Finding files based on your criteria...')
        for idx, img_name in enumerate(self.labels_file['val']["name"]):
            lbl_file_row = self.labels_file['val'][self.labels_file['val'].index == idx]
            attr = lbl_file_row['attributes'][idx]
            if (attr['scene'] in scene) and (attr['timeofday'] in time_of_day):
                addr = "{}".format(
                    str(os.path.join(self.PATH_IMAGES, 'val', img_name)))
                list_of_images.append(addr)
        print('-> Search is done. Found {} files.'.format(len(list_of_images)))
        try:
            final_list = random.sample(list_of_images, how_many)
        except:
            print('x> Please check your criteria. Could find any file or you requested more files than I found.')
        for file_val in final_list:
            shutil.copy(file_val, folder_path)
        print('Done.')

    def convert_yolo_format(self, db_class_name_path, abs_adress=False):

        for mode in ['train', 'val']:
            file_handler = open(os.path.join(
                self.PATH_IMAGES, mode, '{}.txt'.format(mode)), 'w')
            for idx, img_name in enumerate(self.labels_file[mode]["name"]):
                lbl_file_row = self.labels_file[mode][self.labels_file[mode].index == idx]
                num_objs = len(lbl_file_row["labels"][idx])

                if abs_adress:
                    file_name = "{}".format(
                        str(os.path.join(self.PATH_IMAGES, mode, img_name)))
                else:
                    file_name = "{}".format(str(os.path.join(mode, img_name)))

                have_seen_somthing = False
                data = ''
                for i in range(num_objs):
                    object_ = lbl_file_row['labels'][idx][i]
                    if object_['category'] in self.objects.values() and 'box2d' in object_:
                        have_seen_somthing = True
                        coordinates = object_['box2d']
                        lbl_mapped = list(self.objects.keys())[list(
                            self.objects.values()).index(object_['category'])]
                        x1, y1, x2, y2 = (
                            coordinates['x1'], coordinates['y1'], coordinates['x2'], coordinates['y2'])
                        cx, cy, width, height = box2d_to_yolo((x1, y1, x2, y2))
                        data = "{}{} {} {} {} {}\n".format(
                            data, lbl_mapped, cx, cy, width, height)
                if have_seen_somthing:
                    file_handler.writelines("{}\n".format(file_name))
                    label_file_name = (os.path.splitext(file_name)[
                                       0]).replace('images', 'labels')
                    file_handler_label = open(
                        '{}.txt'.format(label_file_name), 'w')
                    file_handler_label.write(data)
                    file_handler_label.close()
                else:
                    print("{} is ignored.".format(img_name))
                print("-> Creating {}.txt: {:.2f}%".format(mode,
                                                           ((idx + 1)*100/self.len_per_mode[mode])), end='\r')
            print()
            file_handler.close()
        print('-> Writing bdd100k.name...')
        f = open(os.path.join(db_class_name_path, "bdd100k.name"), 'w')
        for k in self.objects.values():
            f.write("{}\n".format(k))
        f.close()

    def analyze_db(self, xls_root_file_path):
        self.stats = {'train': {}, 'val': {}}
        self.area = {'train': {'small': 0, 'medium': 0, 'large': 0},
                     'val': {'small': 0, 'medium': 0, 'large': 0}}

        writer_obj = pd.ExcelWriter(os.path.join(
            xls_root_file_path, 'obj.xlsx'), engine='xlsxwriter')
        writer_area = pd.ExcelWriter(os.path.join(
            xls_root_file_path, 'area.xlsx'), engine='xlsxwriter')

        for obj in self.objects.values():
            entery = {obj: {'small': 0, 'medium': 0, 'large': 0, 'total': 0}}
            for mode in ['train', 'val']:
                self.stats[mode].update(copy.deepcopy(entery))
        for mode in ['train', 'val']:
            for idx, img_name in enumerate(self.labels_file[mode]["name"]):
                lbl_file_row = self.labels_file[mode][self.labels_file[mode].index == idx]
                num_objs = len(lbl_file_row["labels"][idx])
                for i in range(num_objs):
                    object_ = lbl_file_row['labels'][idx][i]
                    if object_['category'] in self.objects.values() and 'box2d' in object_:
                        coordinates = object_['box2d']
                        area = get_area(coordinates)

                        if area <= _SMALL:
                            self.area[mode]['small'] += 1
                            self.stats[mode][object_['category']]['small'] += 1
                        elif area >= _LARGE:
                            self.area[mode]['large'] += 1
                            self.stats[mode][object_['category']]['large'] += 1
                        else:
                            self.area[mode]['medium'] += 1
                            self.stats[mode][object_[
                                'category']]['medium'] += 1

                        self.stats[mode][object_[
                            'category']]['total'] += 1

                print("-> Working on {}: {:.2f}%".format(mode,
                                                         ((idx + 1)*100/self.len_per_mode[mode])), end='\r')
            print()
        print('-> Writing {}.'.format(writer_obj.path))
        for mode in ['train', 'val']:
            df = pd.DataFrame(self.stats[mode])
            df.to_excel(writer_obj, sheet_name=mode, index=True)
        writer_obj.close()

        print('-> Writing {}.'.format(writer_area.path))
        for mode in ['train', 'val']:
            df = pd.DataFrame(self.area[mode], index=[0])
            df.to_excel(writer_area, sheet_name=mode, index=False)
        writer_area.close()


def get_area(coordinates):
    return abs((coordinates['x2']-coordinates['x1']) * (coordinates['y2']-coordinates['y1']))


if __name__ == "__main__":
    path = "/mnt/AI_2TB/dataset/bdd100k"
    filtered_objects = {
        0: 'bus',
        1: 'person',
        2: 'bike',
        3: 'truck',
        4: 'motor',
        5: 'car',
        6: 'train',
        7: 'rider'
    }
    file = Path(__file__).resolve()
    package_root_directory = file.parents[1]
    sys.path.append(str(package_root_directory))
    #from dataset import get_image_id
    bdd100k = BDD100K2YOLO(path, objects=filtered_objects)
    bdd100k.find_images(['highway'], ['daytime', 'night'],
                        os.path.join(package_root_directory, 'val_samples'))
    #bdd100k.analyze_db(os.path.join(os.getcwd(), 'stats'))
    # bdd100k.check_val_img_id(get_image_id)
    # bdd100k.convert_yolo_format(os.path.join(
    #    os.getcwd(), 'pytorch_YOLOv4', 'data'), abs_adress=True)
