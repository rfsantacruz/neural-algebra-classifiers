from dbac_lib import dbac_util, dbac_data, dbac_feature_ext
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _extract_features(db_name, db_path, feat_type, output_path, kwargs_str=None):

    # read dataset and partitions
    logger.info("Reading dataset and split")
    db = dbac_data.IDataset.factory(db_name, db_path)

    # set up feature extractor function
    logger.info("Configuring Features Extractor")
    # read kwargs
    kwargs_dic = dbac_util.get_kwargs_dic(kwargs_str)
    logger.info("Kwargs dictionary: {}".format(kwargs_dic))
    feat_extractor = dbac_feature_ext.IFeatureExtractor.factory(feat_type, **kwargs_dic)
    feat_extractor.load()

    # compute features
    logger.info("Computing features...")
    feat_dic = dict()
    for idx, image_path in enumerate(db.images_path):
        feat = feat_extractor.compute(image_path)
        feat_dic[image_path] = feat
        if idx % 1000 == 0:
            logger.info("Cached features for {}/{} images.".format(idx, len(db.images_path)))

    # save dictionary of features
    np.save(output_path, feat_dic)
    logger.info("Dictionary of cached features saved to {}.".format(output_path))


if __name__ == '__main__':
    import argparse
    import os
    from datetime import datetime

    # Parse arguments
    parser = argparse.ArgumentParser(description="Script for generating feature cache.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('db_name', type=str, help="Dataset name", choices=dbac_data.DB_NAMES)
    parser.add_argument('db_path', type=str, help="Dataset directory path")
    parser.add_argument('feat_type', type=str, help='Feature type', choices=dbac_feature_ext.FEAT_TYPE)
    parser.add_argument('output_path', type=str, help="Path to output feature cache dictionary .npz")
    parser.add_argument('-kwargs', type=str, default=None, help="Kwargs for the feature extractor k1=v1; k2=v2; ...")
    parser.add_argument('-gpu_str', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    dbac_util.init_logging('_'.join([os.path.splitext(args.output_path)[0], '{}.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))]))
    logger.info(args)
    _extract_features(args.db_name, args.db_path, args.feat_type, args.output_path, args.kwargs)
