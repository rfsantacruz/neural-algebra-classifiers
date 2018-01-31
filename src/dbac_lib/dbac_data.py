import abc
from dbac_lib import dbac_util
import numpy as np
import pandas as pd
import os
import logging
import json

logger = logging.getLogger(__name__)
DB_NAMES = ['cub200', 'awa2']
_DB_SPLIT_KEYS = ['train_exps', 'test_exps', 'train_imgs', 'val_imgs', 'test_imgs', 'valid_prims',
                  'train_combs', 'test_combs']
DB_IMAGE_SPLITS = ['discarded', 'train', 'val', 'test']
DB_EXP_SPLITS = ['train', 'test']
DB_COMB_SPLITS = ['train', 'test']


class IDataset(metaclass=abc.ABCMeta):
    def __init__(self, name, root_path):

        # dataset name
        self.name = name
        # path to dataset root directory
        self.root_path = root_path

        # Placeholders
        # array of labels names [M]
        self.labels_names = None
        # array of images path [N]
        self.images_path = None
        # array of labels [NXM]
        self.labels = None
        # array of labels group names [G]
        self.labels_group_names = None
        # array of labels groups [M] => [G]
        self.labels_group = None

        # Placeholder for the split file
        # boolean array of valid primitives [M]
        self.valid_primitives = None
        # Valid Expression
        # array of valid expressions (op, p1, p2) [E]
        self.expressions = None
        # array of expressions split [E] (0,1)
        self.expressions_split = None
        # array for split the images in train, val and test [N]
        self.images_split = None

        # Place holders for combinations
        # Combinations of expressions ((),()...) [C]
        self.combinations = None
        # Combinations splits [C] (0, 1, 2)
        self.combinations_split = None

    @abc.abstractmethod
    def load_split(self, split_file, comb_file=None):
        raise NotImplementedError()

    @staticmethod
    def factory(name, root_path):
        db = None
        if name == DB_NAMES[0]:
            db = CUB200(root_path)
        elif name == DB_NAMES[1]:
            db = AWA2(root_path)
        else:
            raise ValueError("Dataset {} in directory {} is not defined.".format(name, root_path))
        return db


class CUB200(IDataset):
    def __init__(self, root_path):
        super().__init__(DB_NAMES[0], root_path)

        # read general info
        df_att = pd.read_csv(os.path.join(self.root_path, 'attributes/attributes.txt'), sep='\s+',
                             names=['att_id', 'att_name'])
        df_att_ant = pd.read_csv(os.path.join(self.root_path, 'attributes/image_attribute_labels.txt'), sep='\s+',
                                 names=['img_id', 'att_id', 'is_pres', 'cert_id', 'time'])
        df_images = pd.read_csv(os.path.join(self.root_path, 'images.txt'), sep='\s+', names=['img_id', 'img_path'])
        df_labels = pd.read_csv(os.path.join(self.root_path, 'classes.txt'), sep='\s+', names=['cls_id', 'cls_name'])
        df_is_train = pd.read_csv(os.path.join(self.root_path, 'train_test_split.txt'), sep='\s+',
                                  names=['img_id', 'is_train'])
        df_data = pd.read_csv(os.path.join(self.root_path, 'image_class_labels.txt'), sep='\s+',
                              names=['img_id', 'cls_id'])

        # merge informations
        df_data = pd.merge(df_images, df_data, on='img_id', how='left')
        df_data = pd.merge(df_data, df_labels, on='cls_id', how='left')
        df_data = pd.merge(df_data, df_is_train, on='img_id', how='left')
        df_data_att = pd.merge(df_att_ant, df_att, on='att_id', how='left')
        df_data_att = df_data_att.loc[(df_data_att['is_pres'] == 1) & (df_data_att['cert_id'] > 2)]

        # Fill placeholders
        self.labels_group_names = np.array(['class', 'attribute'], np.str)
        self.labels_group = np.hstack([np.ones(df_labels['cls_name'].size, np.int) * 0,
                                       np.ones(df_att['att_name'].size, np.int) * 1])
        self.labels_names = np.hstack([df_labels['cls_name'].values.astype(np.str),
                                       df_att['att_name'].values.astype(np.str)])
        self.images_path = []
        self.labels = np.zeros((df_data.shape[0], self.labels_names.size), np.bool)
        for i, (_, row) in enumerate(df_data.iterrows()):
            self.images_path.append(os.path.join(self.root_path, 'images', row['img_path']))
            labels = list(df_data_att.loc[(df_data_att['img_id'] == row['img_id'])]['att_name'].values)
            labels.append(row['cls_name'])
            labels = [np.where(self.labels_names == label)[0][0] for label in labels]
            self.labels[i, labels] = 1.0
        self.images_path = np.array(self.images_path, np.str)

        logger.info("Dataset {} with {} images and {} labels read from {}".format(
            self.name, self.images_path.size, self.labels_names.size, self.root_path))

    def load_split(self, split_file, comb_file=None):
        # read json of partition
        split_dic = None
        with open(split_file, 'r') as f:
            split_dic = json.load(f)

        # fill placeholders
        self.valid_primitives = np.zeros_like(self.labels_names, dtype=np.bool)
        self.valid_primitives[split_dic[_DB_SPLIT_KEYS[5]]] = 1
        self.expressions = np.vstack([split_dic[_DB_SPLIT_KEYS[0]], split_dic[_DB_SPLIT_KEYS[1]]])
        self.expressions_split = np.hstack([np.zeros(len(split_dic[_DB_SPLIT_KEYS[0]]), dtype=np.int),
                                            np.ones(len(split_dic[_DB_SPLIT_KEYS[1]]), dtype=np.int)])
        self.images_split = np.zeros(self.images_path.size, dtype=np.int)
        self.images_split[split_dic[_DB_SPLIT_KEYS[2]]] = 1
        self.images_split[split_dic[_DB_SPLIT_KEYS[3]]] = 2
        self.images_split[split_dic[_DB_SPLIT_KEYS[4]]] = 3

        if comb_file:
            comb_dic = None
            with open(comb_file, 'r') as f:
                comb_dic = json.load(f)
            self.combinations = np.vstack([np.array(comb_dic[_DB_SPLIT_KEYS[6]], dtype=object),
                                           np.array(comb_dic[_DB_SPLIT_KEYS[7]], dtype=object)])
            self.combinations_split = np.hstack([0 * np.ones(len(comb_dic[_DB_SPLIT_KEYS[6]]), dtype=np.int),
                                                 1 * np.ones(len(comb_dic[_DB_SPLIT_KEYS[7]]), dtype=np.int)])


class AWA2(IDataset):
    def __init__(self, root_path):
        super().__init__(DB_NAMES[1], root_path)

        # Read Informations
        df_cls = pd.read_csv(os.path.join(self.root_path, 'classes.txt'), sep='\s+', names=['dummy', 'cls_name'])
        df_att = pd.read_csv(os.path.join(self.root_path, 'predicates.txt'), sep='\s+', names=['dummy', 'att_name'])
        att_mat = np.loadtxt(os.path.join(self.root_path, 'predicate-matrix-binary.txt'))
        images_path, labels = [], []
        for label_idx, cls_name in enumerate(df_cls['cls_name']):
            for img_path in dbac_util.list_pictures(os.path.join(self.root_path, 'JPEGImages', cls_name)):
                images_path.append(img_path)
                labels.append(att_mat[label_idx])

        # Fill placeholders
        self.labels_group_names = np.array(['attribute'], np.str)
        self.labels_names = df_att['att_name'].values.astype(np.str)
        self.labels = np.vstack(labels).astype(np.bool)
        self.labels_group = np.zeros(self.labels.shape[1], dtype=np.int)
        self.images_path = np.array(images_path, dtype=np.str)
        logger.info("Dataset {} with {} images and {} labels read from {}".format(
            self.name, self.images_path.size, self.labels_names.size, self.root_path))

    def load_split(self, split_file, comb_file=None):
        # read json of partition
        split_dic = None
        with open(split_file, 'r') as f:
            split_dic = json.load(f)

        # fill placeholders
        self.valid_primitives = np.zeros_like(self.labels_names, dtype=np.bool)
        self.valid_primitives[split_dic[_DB_SPLIT_KEYS[5]]] = 1
        self.expressions = np.vstack([split_dic[_DB_SPLIT_KEYS[0]], split_dic[_DB_SPLIT_KEYS[1]]])
        self.expressions_split = np.hstack([np.zeros(len(split_dic[_DB_SPLIT_KEYS[0]]), dtype=np.int),
                                            np.ones(len(split_dic[_DB_SPLIT_KEYS[1]]), dtype=np.int)])
        self.images_split = np.zeros(self.images_path.size, dtype=np.int)
        self.images_split[split_dic[_DB_SPLIT_KEYS[2]]] = 1
        self.images_split[split_dic[_DB_SPLIT_KEYS[3]]] = 2
        self.images_split[split_dic[_DB_SPLIT_KEYS[4]]] = 3

        if comb_file:
            comb_dic = None
            with open(comb_file, 'r') as f:
                comb_dic = json.load(f)
            self.combinations = np.vstack([np.array(comb_dic[_DB_SPLIT_KEYS[6]], dtype=object),
                                           np.array(comb_dic[_DB_SPLIT_KEYS[7]], dtype=object)])
            self.combinations_split = np.hstack([0 * np.ones(len(comb_dic[_DB_SPLIT_KEYS[6]]), dtype=np.int),
                                                 1 * np.ones(len(comb_dic[_DB_SPLIT_KEYS[7]]), dtype=np.int)])
