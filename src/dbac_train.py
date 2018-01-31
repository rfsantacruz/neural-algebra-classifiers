from dbac_lib import dbac_data, dbac_model, dbac_feature_ext, dbac_primitives, dbac_util
import logging

logger = logging.getLogger(__name__)


def _train(db_name, db_dir, db_split_file, db_comb_file, primitives_file, model_name, output_dir, kwargs_str=None):

    # processing kwargs
    kwargs_dic = dbac_util.get_kwargs_dic(kwargs_str)
    logger.info("Kwargs dictionary: {}".format(kwargs_dic))

    # read dataset and partitions
    logger.info("Reading dataset and split")
    db = dbac_data.IDataset.factory(db_name, db_dir)
    db.load_split(db_split_file, db_comb_file)
    train_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    train_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]

    if db_comb_file:
        logger.info("Loading compositions...")
        train_exps = db.combinations[db.combinations_split == dbac_data.DB_COMB_SPLITS.index('train')]
    else:
        logger.info("Loading single expressions...")
        train_exps = db.expressions[db.expressions_split == dbac_data.DB_EXP_SPLITS.index('train')]

    # Set up feature extractor
    logger.info("Configuring Features Extractor")
    feat_extractor = dbac_feature_ext.IFeatureExtractor.factory(dbac_feature_ext.FEAT_TYPE[1], **kwargs_dic)
    feat_extractor.load()

    # set up primitive collection
    logger.info("Configuring Primitive Collection")
    prim_collection = dbac_primitives.IPrimitiveCollection.factory(dbac_primitives.PRIMITIVE_TYPES[0], **kwargs_dic)
    prim_collection.load(primitives_file)

    # setup model
    logger.info("Configuring Model")
    model = dbac_model.IModel.factory(model_name, feat_extractor, prim_collection, **kwargs_dic, is_train=True)
    logger.info("Training...")
    model.learning(train_imgs_path, train_labels, train_exps, **kwargs_dic)
    model_file = os.path.join(output_dir, 'model.npy')
    model.save(model_file)
    logger.info("Model Saved to {}".format(model_file))


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import os

    parser = argparse.ArgumentParser(description="Script for Learning Neural Algebra of Classifiers models.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('db_name', type=str, choices=dbac_data.DB_NAMES, help='Name of the Dataset.')
    parser.add_argument('db_dir', type=str, help='Path to the dataset main directory.')
    parser.add_argument('split_file', type=str, help='Path to the split json file.')
    parser.add_argument('-comb_file', default=None, type=str, help='Path to the compositions json file.')
    parser.add_argument('primitives_file', type=str, help='Path to the primitives collection file.')
    parser.add_argument('model_name', type=str, choices=dbac_model._MODEL_NAMES, help='Model name.')
    parser.add_argument('output_dir', type=str, help='Output directory path.')
    parser.add_argument('-gpu_str', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('-kwargs', type=str, default=None, help="Kwargs for the feature extractor k1=v1; k2=v2; ...")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    log_file = os.path.join(args.output_dir, "train_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    dbac_util.init_logging(log_file)
    logger.info(args)
    _train(args.db_name, args.db_dir, args.split_file, args.comb_file, args.primitives_file, args.model_name, args.output_dir, args.kwargs)
