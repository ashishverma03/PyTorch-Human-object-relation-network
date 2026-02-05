import torch
from torch.utils.data import Dataset
import os
import pickle as pkl
from matplotlib import pyplot as plt
import cv2
import logging
import numpy as np
from random import randint
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import pdb


class ClassProperty(object):
    """Readonly @ClassProperty descriptor for internal usage."""
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class VisionDataset(Dataset):
    """Base Dataset with directory checker.

    Parameters
    ----------
    root : str
        The root path of xxx.names, by defaut is '~/.mxnet/datasets/foo', where
        `foo` is the name of the dataset.
    """
    def __init__(self, root):
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

class VOCAction(VisionDataset):
    CLASSES = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
                'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')
    
    def __init__(self, root=os.path.join('~', 'data', 'VOCdevkit'), split='train', index_map=None, preload_label=True, augment_box=False, load_box=False, random_cls=False,transform = None):
        super(VOCAction,self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.join(os.path.expanduser(root), 'VOC2012')
        self._augment_box = augment_box
        self._load_box = load_box
        self._random_cls = random_cls
        self._split = split
        if self._split.lower() == 'val':
            self._jumping_start_pos = 307
        elif self._split.lower() == 'test':
            self._jumping_start_pos = 613
        else:
            self._jumping_start_pos = 0
        self._items = self._load_items(split)
        self._anno_path = os.path.join(self._root, 'Annotations', '{}.xml')
        self._box_path = os.path.join(self._root, 'Boxes', '{}.pkl')
        self._image_path = os.path.join(self._root, 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None
        self.transform = transform



    def __str__(self):
        return self.__class__.__name__ + '(' + self._split + ')'


    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    
    def img_path(self, idx):
        img_id = self._items[idx]
        return self._image_path.format(img_id)


    def save_boxes(self, idx, boxes):
        img_id = self._items[idx]
        box_path = self._box_path.format(img_id)
        with open(box_path, 'wb') as f:
            pkl.dump(boxes, f)


    def __len__(self):
        return len(self._items)


    def augment(self,bbox, img_w, img_h, output_num=16, iou_thresh=0.7):
        # print("-------------xxxxxxxxxxxxxxxx-----------------")
        # print("bbox:",bbox)
        # print("img_w:",img_w)
        # print("img_h:",img_h)
        # # print(img.shape)
        # # print(h,w)
        # print("-------------xxxxxxxxxxxxxxxx-----------------")
        #pdb.set_trace()
        bbox = bbox.copy()
        np.random.shuffle(bbox)
        ori_num = bbox.shape[0]
        aug_num = ori_num
        if aug_num >= output_num:
            return bbox[0:output_num, :]

        time_count = 1
        ori_index = 0
        boxes = [bbox, ]
        while aug_num < output_num:
            if time_count > 150:
                while aug_num < output_num:
                    boxes.append(bbox[randint(0, ori_num-1)].copy())
                    aug_num += 1
                break
            aug_box = bbox[ori_index].copy()
            height = aug_box[3] - aug_box[1]
            width = aug_box[2] - aug_box[0]

            height_cen = (aug_box[3] + aug_box[1]) / 2
            width_cen = (aug_box[2] + aug_box[0]) / 2

            ratio = 1 + randint(-10, 10) * 0.01

            height_shift = randint(-np.floor(height), np.floor(height)) * 0.1
            width_shift = randint(-np.floor(width), np.floor(width)) * 0.1

            aug_box[0] = max(0, width_cen + width_shift - ratio * width / 2)
            aug_box[2] = min(img_w - 1, width_cen + width_shift + ratio * width / 2)
            aug_box[1] = max(0, height_cen + height_shift - ratio * height / 2)
            aug_box[3] = min(img_h - 1, height_cen + height_shift + ratio * height / 2)


            boxA = bbox[ori_index]
            boxB = aug_box

            ixmin = np.maximum(boxA[0], boxB[0])
            iymin = np.maximum(boxA[1], boxB[1])
            ixmax = np.minimum(boxA[2], boxB[2])
            iymax = np.minimum(boxA[3], boxB[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
                (boxA[2] - boxA[0] + 1.) *
                (boxA[3] - boxA[1] + 1.) - inters)

            iou = inters / uni


            if iou > iou_thresh:
                boxes.append(aug_box.reshape(1, -1))
                aug_num += 1
                ori_index = (ori_index + 1) % ori_num
            time_count += 1

        return np.concatenate(boxes, axis=0)


    def __getitem__(self, idx):
        fid = self._items[idx]
        img_id = fid
        img_path = self._image_path.format(img_id)
        label, h_x, h_y = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        # label = np.array(label)
        # print("-------------xxxxxxxxxxxxxxxx-----------------")
        # print(img_id)
        # print(img_path)
        # print(label)
        # print("-------------xxxxxxxxxxxxxxxx-----------------")
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._random_cls:
            for i, cls in enumerate(label[:, 5:]):
                candidate_cls = np.array(np.where(cls == 1)).reshape((-1,))
                label[i, 4] = np.random.choice(candidate_cls)
        if self._augment_box:
            h, w, _ = img.shape
            # print("-------------xxxxxxxxxxxxxxxx-----------------")
            # print(type(w))
            # print(img.shape)
            # print(h,w)
            # print("w:",w)
            # print("h:",h)
            # print("-------------xxxxxxxxxxxxxxxx-----------------")
            label = self.augment(label, img_w=w, img_h=h, output_num=16)
        if self._load_box:
            box_path = self._box_path.format(img_id)
            with open(box_path, 'rb') as f:
                box = pkl.load(f)
                # print(box)

            if self.transform:
                img,label,box = self.transform(img,label,box)
            return img, label, box
            img,label = self.transform(img,label)
        return img, label

    
    def _load_items(self, split):
        """Load individual image indices from split."""
        ids = []
        set_file = os.path.join(self._root, 'ImageSets', 'Action', split + '.txt')
        with open(set_file, 'r') as f:
            ids += [line.strip() for line in f.readlines()]
        return ids


    def _load_label(self, idx):
        """Parse xml file and return labels."""
        fid = self._items[idx]
        # img_id = self._items[idx]
        img_id = fid
        anno_path = self._anno_path.format(img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name != 'person':
                continue

            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            h_x = xmax - xmin
            h_y = ymax - ymin
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))

            cls_id = -1
            act_cls = obj.find('actions')
            cls_array = [0] * len(self.classes)
            if idx < self._jumping_start_pos:
                # ignore jumping class according to voc offical code
                cls_array[0] = -1
            if act_cls is not None:
                for i, cls_name in enumerate(self.classes):
                    is_action = float(act_cls.find(cls_name).text)
                    if is_action > 0.5:
                        cls_id = i
                        cls_array[i] = 1
            anno = [xmin, ymin, xmax, ymax, cls_id]
            anno.extend(cls_array)
            label.append(anno)
        return np.array(label), h_x, h_y


    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)


    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]



    
