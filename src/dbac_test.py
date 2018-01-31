from dbac_lib import dbac_data, dbac_model, dbac_feature_ext, dbac_primitives, dbac_util, dbac_expression
import logging
import numpy as np

from dbac_lib.dbac_expression import eval_op
from dbac_lib.dbac_model import exp_members

logger = logging.getLogger(__name__)


def _test(db_name, db_dir, db_split_file, db_comb_file, primitives_file, model_name, model_file, output_dir, kwargs_str=None):
    # processing kwargs
    kwargs_dic = dbac_util.get_kwargs_dic(kwargs_str)
    logger.info("Kwargs dictionary: {}".format(kwargs_dic))

    # read dataset and partitions
    logger.info("Reading dataset and split")
    db = dbac_data.IDataset.factory(db_name, db_dir)
    db.load_split(db_split_file, db_comb_file)
    train_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    train_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    val_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('val')]
    val_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('val')]
    test_imgs_path = db.images_path[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('test')]
    test_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('test')]

    if db_comb_file:
        logger.info("Loading compositions...")
        train_exps = db.combinations[db.combinations_split == dbac_data.DB_COMB_SPLITS.index('train')]
        test_exps = db.combinations[db.combinations_split == dbac_data.DB_COMB_SPLITS.index('test')]
    else:
        logger.info("Loading single expressions...")
        train_exps = db.expressions[db.expressions_split == dbac_data.DB_EXP_SPLITS.index('train')]
        test_exps = db.expressions[db.expressions_split == dbac_data.DB_EXP_SPLITS.index('test')]

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
    model = dbac_model.IModel.factory(model_name, feat_extractor, prim_collection, **kwargs_dic, is_train=False)
    model.load(model_file)

    # test model
    logger.info("Testing on seen expressions on training images...")
    train_scores = model.score(train_imgs_path, train_exps, **kwargs_dic)
    logger.info("Testing on seen expressions on validation images...")
    val_scores = model.score(val_imgs_path, train_exps, **kwargs_dic)
    logger.info("Testing on unseen expressions on test images...")
    test_scores = model.score(test_imgs_path, test_exps, **kwargs_dic)

    # save results
    logger.info("Computing results.")
    report_dic = dict()
    results_iter = zip(['train', 'val', 'test'], [train_exps, train_exps, test_exps],
                       [train_labels, val_labels, test_labels], [train_scores, val_scores, test_scores],
                       [train_imgs_path, val_imgs_path, test_imgs_path])
    for key, exps, labels, scores, images in results_iter:
        # compute ground truth labels
        ground_truth = np.zeros_like(scores)
        for idx, exp_lst in enumerate(exps):
            exp_tree = dbac_expression.list2exp_parse(exp_lst)
            var_dic = {p: labels[:, int(p)] for p in dbac_expression.get_vars(exp_tree)}
            ground_truth[idx] = dbac_expression.eval_exp(exp_tree, var_dic)
        # fill report dictionary
        report_dic['_'.join([key, 'exps'])] = exps
        report_dic['_'.join([key, 'imgs'])] = images
        report_dic['_'.join([key, 'gt'])] = ground_truth
        report_dic['_'.join([key, 'pred'])] = scores
    result_file = os.path.join(output_dir, 'results.npy')
    np.save(result_file, report_dic)
    logger.info("Results file saved to {}.".format(result_file))


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import os

    parser = argparse.ArgumentParser(description="Script for test Neural Algebra of Classifiers models.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('db_name', type=str, choices=dbac_data.DB_NAMES, help='Name of the Dataset.')
    parser.add_argument('db_dir', type=str, help='Path to the dataset main directory.')
    parser.add_argument('split_file', type=str, help='Path to the split json file.')
    parser.add_argument('-comb_file', default=None, type=str, help='Path to the compositions json file.')
    parser.add_argument('primitives_file', type=str, help='Path to the primitives collection file.')
    parser.add_argument('model_name', type=str, choices=dbac_model._MODEL_NAMES, help='Model name.')
    parser.add_argument('output_dir', type=str, help='Output directory path.')
    parser.add_argument('-model_file', type=str, default=None, help='Path to model file.')
    parser.add_argument('-gpu_str', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('-kwargs', type=str, default=None, help="Kwargs for the feature extractor k1=v1; k2=v2; ...")
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    log_file = os.path.join(args.output_dir, "test_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    dbac_util.init_logging(log_file)
    logger.info(args)
    _test(args.db_name, args.db_dir, args.split_file, args.comb_file, args.primitives_file, args.model_name, args.model_file,
          args.output_dir, args.kwargs)
