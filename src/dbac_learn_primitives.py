from dbac_lib import dbac_util, dbac_data, dbac_primitives, dbac_feature_ext
import numpy as np
import logging
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


logger = logging.getLogger(__name__)


def _learn_primitives(db_name, db_dir, split_file, prim_rpr_file, ex_size=10, num_ex=10, subset_prim_ids=None,
                      kwargs_str=None):
    # processing kwargs
    kwargs_dic = dbac_util.get_kwargs_dic(kwargs_str)
    logger.info("Kwargs dictionary: {}".format(kwargs_dic))

    # read dataset and partitions
    logger.info("Reading dataset and split")
    db = dbac_data.IDataset.factory(db_name, db_dir)
    db.load_split(split_file)
    train_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    train_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]

    # select subset of primitives
    if subset_prim_ids is None:
        subset_prim_ids = np.where(db.valid_primitives)[0].tolist()
    logger.info("Selected Primitives: {}".format(subset_prim_ids))

    # set up feature extractor function
    logger.info("Configuring Features Extractor")
    feat_extractor = dbac_feature_ext.IFeatureExtractor.factory(dbac_feature_ext.FEAT_TYPE[1], **kwargs_dic)
    feat_extractor.load()

    # Learning exemplar SVMS for primitives
    prims = dbac_primitives.IPrimitiveCollection.factory(dbac_primitives.PRIMITIVE_TYPES[0], **kwargs_dic)
    logger.info("Learning Primitives...")
    prims.learn(train_imgs_path, train_labels, feat_extractor,
                num_ex=num_ex, ex_size=ex_size, prim_ids=subset_prim_ids, **kwargs_dic)
    prims.save(prim_rpr_file)
    logger.info("Primitives saved to {}.".format(prim_rpr_file))


def _test_primitives(db_name, db_dir, split_file, prim_rpr_file, subset_prim_ids=None, kwargs_str=None):
    # processing kwargs
    kwargs_dic = dbac_util.get_kwargs_dic(kwargs_str)
    logger.info("Kwargs dictionary: {}".format(kwargs_dic))

    # read dataset and partitions
    logger.info("Reading dataset and split")
    db = dbac_data.IDataset.factory(db_name, db_dir)
    db.load_split(split_file)
    train_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    train_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    test_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('test')]
    test_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('test')]
    val_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('val')]
    val_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('val')]

    # set up feature extractor function
    logger.info("Configuring Features Extractor")
    feat_extractor = dbac_feature_ext.IFeatureExtractor.factory(dbac_feature_ext.FEAT_TYPE[1], **kwargs_dic)
    feat_extractor.load()

    # Learning exemplar SVMS for primitives
    prims = dbac_primitives.IPrimitiveCollection.factory(dbac_primitives.PRIMITIVE_TYPES[0], **kwargs_dic)
    logger.info("Loading Primitive collection")
    prims.load(prim_rpr_file)

    # select subset of primitives
    if subset_prim_ids is None:
        subset_prim_ids = prims.get_ids()
    else:
        subset_prim_ids = list(set(subset_prim_ids).intersection(set(prims.get_ids())))
    logger.info("Selected Primitives: {}".format(subset_prim_ids))

    # test primitives
    report_dic = dict()
    for key, images, labels in zip(['train', 'val', 'test'], [train_imgs_path, val_imgs_path, test_imgs_path],
                                    [train_labels, val_labels, test_labels]):
        logger.info("Testing partition: {}".format(key))
        images_feats = feat_extractor.compute(images)
        # considering uncalibrated scores
        #rprs = np.vstack([prims.get_rpr(pid)[0] for pid in subset_prim_ids])
        #scores = rprs[:, 0].reshape((-1, 1)) + np.dot(rprs[:, 1:], images_feats.T)
        # considering calibrated scores
        scores = np.vstack([prims.get_cls(pid)[0].predict_proba(images_feats)[:, 1] for pid in subset_prim_ids])
        # fill report dictionary
        assert scores.shape == labels[:, subset_prim_ids].T.shape
        report_dic['_'.join([key, 'exps'])] = subset_prim_ids
        report_dic['_'.join([key, 'imgs'])] = images
        report_dic['_'.join([key, 'gt'])] = labels[:, subset_prim_ids].T
        report_dic['_'.join([key, 'pred'])] = scores

    result_file = "{}.results.npy".format(os.path.splitext(prim_rpr_file)[0])
    np.save(result_file, report_dic)
    logger.info("Results file saved to {}.".format(result_file))


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import os

    parser = argparse.ArgumentParser(description="Script to Learn Primitives Representation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='commands', dest='cmd_name', help='additional help')

    # parser for learning
    parser_learn = subparsers.add_parser('learn', help='Learn primitives representation',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_learn.add_argument('db_name', type=str, help='Name of the dataset.', choices=dbac_data.DB_NAMES)
    parser_learn.add_argument('db_dir', type=str, help='Path to the dataset main directory.')
    parser_learn.add_argument('split_file', type=str, help='Path to the split json file.')
    parser_learn.add_argument('file_name', type=str, help='Path to the output file .npy .')
    parser_learn.add_argument('-ex_size', default=10, type=int, help='Number of positive samples per exemplar.')
    parser_learn.add_argument('-num_ex', default=10, type=int, help='Number of exemplars per primitive.')
    parser_learn.add_argument('-gpu_str', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser_learn.add_argument('-prim_subset_ids', nargs='*', default=None, type=int, help='Subset of primitives.')
    parser_learn.add_argument('-kwargs', type=str, default=None, help="Kwargs for the feature extractor k1=v1; k2=v2; ...")

    # parser for test
    parser_test = subparsers.add_parser('test', help='Test primitives representation',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_test.add_argument('db_name', type=str, help='Name of the dataset.', choices=dbac_data.DB_NAMES)
    parser_test.add_argument('db_dir', type=str, help='Path to the dataset main directory.')
    parser_test.add_argument('split_file', type=str, help='Path to the split json file.')
    parser_test.add_argument('file_name', type=str, help='Path to the output file .npy .')
    parser_test.add_argument('-gpu_str', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser_test.add_argument('-prim_subset_ids', nargs='*', default=None, type=int, help='Subset of primitives.')
    parser_test.add_argument('-kwargs', type=str, default=None, help="Kwargs for the feature extractor k1=v1; k2=v2; ...")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    if args.cmd_name == 'learn':
        log_file = "{}_{}.learn.log".format(os.path.splitext(args.file_name)[0], datetime.now().strftime("%Y%m%d-%H%M%S"))
        dbac_util.init_logging(log_file)
        logger.info(args)
        _learn_primitives(args.db_name, args.db_dir, args.split_file, args.file_name, args.ex_size, args.num_ex,
                            args.prim_subset_ids, args.kwargs)
    elif args.cmd_name == 'test':
        log_file = "{}_{}.test.log".format(os.path.splitext(args.file_name)[0],
                                            datetime.now().strftime("%Y%m%d-%H%M%S"))
        dbac_util.init_logging(log_file)
        logger.info(args)
        _test_primitives(args.db_name, args.db_dir, args.split_file, args.file_name, args.prim_subset_ids, args.kwargs)
    else:
        raise ValueError('Not well formatted command line arguments. Parsed arguments {}'.format(args))
