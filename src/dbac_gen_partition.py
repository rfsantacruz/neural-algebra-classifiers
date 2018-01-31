from dbac_lib import dbac_data, dbac_expression, dbac_util
import numpy as np
import json
import itertools
import logging
import copy

logger = logging.getLogger(__name__)


def _compute_splits(db_name, db_path, operations, output_path, min_per_prim=1, min_per_exp=1, num_exps=4000,
                    min_pos_exp_test=10, nogroup=False):
    # load dataset
    logger.info("Loading dataset...")
    dataset = dbac_data.IDataset.factory(db_name, db_path)

    # configure primitives - by groups
    prim_ids = []
    for group_id, group_name in enumerate(dataset.labels_group_names):
        label_ids = np.where(dataset.labels_group == group_id)[0].tolist()
        prim_ids.append(label_ids)
    logger.info("Number of primitives per group: {}".format([len(p) for p in prim_ids]))

    # Compute valid expression - minimum of samples per expression and primitive
    # each primitive must be present in more than one expression
    valid_exps, valid_prims = set(), set()
    for exp in dbac_expression.all_ops(prim_ids, operations, nogroup):
        op, v_a, v_b = exp[0], exp[1], exp[2]
        res = dataset.labels[:, v_a]
        if np.sum(res) < min_per_prim or np.sum(np.logical_not(res)) < min_per_prim:
            continue
        res = dataset.labels[:, v_b] if v_b is not None else None
        if v_b is not None and (np.sum(res) < min_per_prim or np.sum(np.logical_not(res)) < min_per_prim):
            continue
        res = dbac_expression.eval_op(op, dataset.labels[:, v_a], dataset.labels[:, v_b])
        if np.sum(res) < min_per_exp or np.sum(np.logical_not(res)) < min_per_exp:
            continue
        valid_exps.add(exp)
        valid_prims.update([v for v in [v_a, v_b] if v is not None])
    valid_prims = set([p for p in valid_prims if np.sum([p in exp[1:] for exp in valid_exps]) > 1])
    valid_exps = set([exp for exp in valid_exps if all([p in valid_prims for p in exp[1:] if p is not None])])
    logger.info("Number of valid primitives and expressions: {}, {}".format(len(valid_prims), len(valid_exps)))

    if len(valid_exps) > num_exps:
        samp_idx = np.random.randint(len(valid_exps), size=num_exps)
        valid_exps = list(valid_exps)
        valid_exps = set([valid_exps[idx] for idx in samp_idx])
        valid_prims = set([p for p in itertools.chain.from_iterable(valid_exps) if p not in operations and p is not None])
        valid_prims = set([p for p in valid_prims if np.sum([p in exp[1:] for exp in valid_exps]) > 1])
        valid_exps = set([exp for exp in valid_exps if all([p in valid_prims for p in exp[1:] if p is not None])])
        logger.info("Number of sampled primitives and expressions: {}, {}".format(len(valid_prims), len(valid_exps)))


    # Compute expressions split - 75%, 25% and empty intersection
    # All primitives must have at least one expression in training exps and test exps
    num_test = np.floor(len(valid_exps) * 0.25)
    train_exps, test_exps = valid_exps.copy(), set()
    while len(test_exps) < num_test:

        prims = [p for p in valid_prims if all([p not in exp[1:] for exp in test_exps]) or len(test_exps) == 0]
        while True:
            exps = [exp for exp in train_exps if any([p in prims for p in exp[1:] if p is not None])]
            exps = [exp for exp in exps if all(
                [len([exp2 for exp2 in train_exps if p in exp2[1:] and exp2 != exp]) > 0 for p in exp[1:] if
                    p is not None])]
            if len(exps) > 0:
                break
            else:
                prims = valid_prims

        samp_exp = exps[np.random.randint(len(exps))]
        test_exps.add(samp_exp)
        train_exps.remove(samp_exp)
        if len(test_exps) % 50 == 0:
            logger.info("Collected {}/{} test expressions.".format(len(test_exps), num_test))
    assert len(test_exps) == num_test, "Was not possible to correctly partition the dataset expressions"
    train_prims = [p for p in itertools.chain.from_iterable(train_exps) if p is not None and p not in operations]
    assert len(valid_prims - set(train_prims)) == 0, "There are primitives without training expressions"
    test_prims = [p for p in itertools.chain.from_iterable(test_exps) if p is not None and p not in operations]
    #assert len(valid_prims - set(test_prims)) == 0, "There are primitives without test expressions"
    logger.info(len(valid_prims - set(test_prims)))
    logger.info("Total train, test expressions: {}, {}".format(len(train_exps), len(test_exps)))
    logger.info("Intersection expressions: {}".format(len(train_exps.intersection(test_exps))))

    # Compute images splits
    test_imgs = set()
    for exp in test_exps:
        op, v_a, v_b = exp[0], dataset.labels[:, exp[1]], dataset.labels[:, exp[2]] if exp[2] else None
        ids = set(np.where(dbac_expression.eval_op(op, v_a, v_b))[0])
        samp = min_pos_exp_test - len(test_imgs.intersection(ids))
        if samp > 0:
            ids = np.random.choice(list(ids - test_imgs), size=samp, replace=False)
            test_imgs.update(ids)

    val_imgs = set()
    for exp in train_exps:
        op, v_a, v_b = exp[0], dataset.labels[:, exp[1]], dataset.labels[:, exp[2]] if exp[2] else None
        ids = set(np.where(dbac_expression.eval_op(op, v_a, v_b))[0]) - test_imgs
        samp = min_pos_exp_test - len(val_imgs.intersection(ids))
        if samp > 0:
            ids = np.random.choice(list(ids - val_imgs), size=samp, replace=False)
            val_imgs.update(ids)

    train_imgs = set()
    for exp in train_exps:
        op, v_a, v_b = exp[0], dataset.labels[:, exp[1]], dataset.labels[:, exp[2]] if exp[2] else None
        ids = set(np.where(dbac_expression.eval_op(op, v_a, v_b))[0]) - test_imgs - val_imgs
        train_imgs.update(ids)
    logger.info(
        "Number of train, val and test images: {}, {}, {}".format(len(train_imgs), len(val_imgs), len(test_imgs)))
    logger.info("Intesections of train and val, train and test, val and test images: {}, {}, {}"
                .format(len(train_imgs.intersection(val_imgs)), len(train_imgs.intersection(test_imgs)),
                        len(val_imgs.intersection(test_imgs))))
    train_imgs, val_imgs, test_imgs = list(train_imgs), list(val_imgs), list(test_imgs)

    # compute statistics
    info_exp, info_prim = [], []
    for imgs, exps, prims in zip([train_imgs, val_imgs, test_imgs], [train_exps, train_exps, test_exps],
                                 [valid_prims] * 3):
        count_prim = np.zeros(len(prims))
        for p, prim in enumerate(prims):
            count_prim[p] = np.sum(dataset.labels[imgs, prim])
        mean, std, min, max = np.mean(count_prim), np.std(count_prim), np.min(count_prim), np.max(count_prim)
        info_prim.append((mean, std, min, max))

        count_exp = np.zeros(len(exps))
        for e, exp in enumerate(exps):
            op, v_a, v_b = exp[0], dataset.labels[imgs, exp[1]], dataset.labels[imgs, exp[2]] if exp[2] else None
            count_exp[e] = np.sum(dbac_expression.eval_op(op, v_a, v_b))
        mean, std, min, max = np.mean(count_exp), np.std(count_exp), np.min(count_exp), np.max(count_exp)
        info_exp.append((mean, std, min, max))
    logger.info("Primitives sample density (mean, std, min, max) train, val, test: {}".format(info_prim))
    logger.info("Expression sample density (mean, std, min, max) train, val, test: {}".format(info_exp))

    # Post processing and Fill dictionary and output - workaround to get rid of the np.int types
    splits_dic = {'train_exps': sorted(np.array(train_exps).tolist()),
                  'test_exps': sorted(np.array(test_exps).tolist()),
                  'train_imgs': sorted(np.array(train_imgs).tolist()), 'val_imgs': sorted(np.array(val_imgs).tolist()),
                  'test_imgs': sorted(np.array(test_imgs).tolist()),
                  'valid_prims': sorted(np.array(valid_prims).tolist())}

    # save to json and return
    with open(output_path, "w") as outfile:
        json.dump(splits_dic, outfile, indent=4, sort_keys=True)
        logger.info("Split saved to {}.".format(output_path))
    return splits_dic


def _compute_combs(db_name, db_path, split_file, complexity, output_file, form='D', allow_not=False,
                   min_pos_comb=[100, 10, 10],
                   qtd_combs=[3e3, 1e3, 1e3]):
    # load data
    logger.info("Loading dataset and split...")
    db = dbac_data.IDataset.factory(db_name, db_path)
    db.load_split(split_file)
    train_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    train_exps = db.expressions[db.expressions_split == dbac_data.DB_EXP_SPLITS.index('train')]
    val_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('val')]
    test_labels = db.labels[db.images_split == dbac_data.DB_IMAGE_SPLITS.index('test')]
    test_exps = db.expressions[db.expressions_split == dbac_data.DB_EXP_SPLITS.index('test')]
    # compute combinations
    comb_dict = {}
    train_combs, test_combs = [], []
    data_iter = zip(['train', 'test'], [train_combs, test_combs], min_pos_comb, qtd_combs,
                    [train_exps, test_exps], [train_labels, test_labels])
    for name, combs, min_pos, qtd, exps, labels in data_iter:
        logger.info("Computing combinations for {}".format(name))
        for idx, exp in enumerate(dbac_expression.exp_comb_gen(exps, complexity, form=form, allow_not=allow_not)):
            # compute expression positive samples
            var_dic = {p: labels[:, int(p)] for p in dbac_expression.get_vars(exp)}
            exp_labels = dbac_expression.eval_exp(exp, var_dic)
            num_pos = exp_labels.sum()
            num_neg = np.logical_not(exp_labels).sum()
            # filter out expressions with few samples
            if num_pos > min_pos and num_neg > min_pos and not any(
                    [dbac_expression.exp_tree_eq(exp, comb) for comb in combs]):
                if name == 'train':
                    var_dic = {p: val_labels[:, int(p)] for p in dbac_expression.get_vars(exp)}
                    exp_labels = dbac_expression.eval_exp(exp, var_dic)
                    num_pos = exp_labels.sum()
                    num_neg = np.logical_not(exp_labels).sum()
                    if num_pos <= min_pos_comb[1] or num_neg <= min_pos_comb[1]:
                        continue
                combs.append(copy.deepcopy(exp))
                # maximum number of combinations sampled
            if len(combs) > qtd:
                break
            if (idx + 1) % 100 == 0:
                logger.info('{} - Collected {} from {} expressions'.format(name, len(combs), (idx + 1)))
        comb_dict["{}_combs".format(name)] = [dbac_expression.exp2list_parse(comb) for comb in combs[:qtd]]

    # save to json file
    with open(output_file, "w") as outfile:
        json.dump(comb_dict, outfile, indent=4, sort_keys=True)
    logger.info("Combination saved to {}.".format(output_file))
    return comb_dict


def _compute_statistics(db_name, db_path, split_file, comb_file=None, plot=False):
    # Read data
    logger.info("Loading dataset and split...")
    dataset = dbac_data.IDataset.factory(db_name, db_path)
    dataset.load_split(split_file, comb_file)

    # extract information
    train_labels = dataset.labels[dataset.images_split == dbac_data.DB_IMAGE_SPLITS.index('train')]
    val_labels = dataset.labels[dataset.images_split == dbac_data.DB_IMAGE_SPLITS.index('val')]
    test_labels = dataset.labels[dataset.images_split == dbac_data.DB_IMAGE_SPLITS.index('test')]
    logger.info("Number of images (train, val, test): {}, {}, {}".format(len(train_labels), len(val_labels), len(test_labels)))

    train_exps = dataset.expressions[dataset.expressions_split == dbac_data.DB_EXP_SPLITS.index('train')]
    test_exps = dataset.expressions[dataset.expressions_split == dbac_data.DB_EXP_SPLITS.index('test')]
    logger.info("Number of expressions (train, test): {}, {}".format(len(train_exps), len(test_exps)))

    valid_prims = np.where(dataset.valid_primitives)[0]
    logger.info("Number of primitives: {}".format(len(valid_prims)))

    if comb_file:
        train_combs = dataset.combinations[dataset.combinations_split == dbac_data.DB_COMB_SPLITS.index('train')]
        test_combs = dataset.combinations[dataset.combinations_split == dbac_data.DB_COMB_SPLITS.index('test')]
        logger.info("Number of combinations (train, test): {}, {}".format(len(train_combs), len(test_combs)))
    else:
        train_combs, test_combs = None, None

    # compute detailed statistics
    info_exp, info_prim, info_comb = [], [], []
    prim_box, exp_box, comb_box = [], [], []
    for labels, combs, exps, prims in zip([train_labels, val_labels, test_labels],
                                          [train_combs, train_combs, test_combs], [train_exps, train_exps, test_exps],
                                          [valid_prims] * 3):
        count_prim = np.zeros(len(prims))
        for p, prim in enumerate(prims):
            count_prim[p] = np.sum(labels[:, prim])
        mean, std, min, max = np.mean(count_prim), np.std(count_prim), np.min(count_prim), np.max(count_prim)
        info_prim.append((mean, std, min, max))
        prim_box.append(count_prim)

        count_exp = np.zeros(len(exps))
        for e, exp in enumerate(exps):
            op, v_a, v_b = exp[0], labels[:, int(exp[1])], labels[:, int(exp[2])] if exp[2] is not None else None
            count_exp[e] = np.sum(dbac_expression.eval_op(op, v_a, v_b))
        mean, std, min, max = np.mean(count_exp), np.std(count_exp), np.min(count_exp), np.max(count_exp)
        info_exp.append((mean, std, min, max))
        exp_box.append(count_exp)

        if comb_file:
            count_comb = np.zeros(len(combs))
            for c, comb in enumerate(combs):
                comb_tree = dbac_expression.list2exp_parse(comb)
                var_dic = {p: labels[:, int(p)] for p in dbac_expression.get_vars(comb_tree)}
                count_comb[c] = dbac_expression.eval_exp(comb_tree, var_dic).sum()
            mean, std, min, max = np.mean(count_comb), np.std(count_comb), np.min(count_comb), np.max(count_comb)
            info_comb.append((mean, std, min, max))
            comb_box.append(count_comb)

    logger.info("Primitives sample density (mean, std, min, max) train, val, test: {}".format(info_prim))
    logger.info("Expression sample density (mean, std, min, max) train, val, test: {}".format(info_exp))
    logger.info("Compositions sample density (mean, std, min, max) train, val, test: {}".format(info_comb))
    if plot:
        import matplotlib.pyplot as plt
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.boxplot(prim_box, labels=['Train', 'Val', 'Test'], showmeans=True)
        ax1.set_title('Primitives', fontsize=16)
        ax1.set_ylabel('# Positive Images', fontsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax2.boxplot(exp_box, labels=['Train', 'Val', 'Test'], showmeans=True)
        ax2.set_title('Expressions', fontsize=16)
        ax2.tick_params(axis='x', labelsize=12)
        if comb_file:
            ax3.boxplot(comb_box, labels=['Train', 'Val', 'Test'], showmeans=True)
            ax3.set_title('Compositions', fontsize=16)
            ax3.tick_params(axis='x', labelsize=12)
        f.tight_layout()
        plt.show()


def _merge_split(split_files, output_file):

    # read split files
    split_dics = []
    for split_file in split_files:
        logger.info("Reading split file from: {}".format(split_file))
        with open(split_file, 'r') as file:
            split_dics.append(json.load(file))
    logger.info("{} split files read".format(len(split_dics)))

    # merge files
    merge_split_dic = split_dics[0]
    for split_dic in split_dics:
        for key in merge_split_dic.keys():
            merge_split_dic[key] += split_dic[key]
            if key in ['train_imgs', 'val_imgs', 'test_imgs', 'valid_prims']:
                merge_split_dic[key] = np.unique(merge_split_dic[key]).tolist()

    # remove intersections
    merge_split_dic['train_imgs'] = list((set(merge_split_dic['train_imgs']) - set(merge_split_dic['test_imgs'])) - set(merge_split_dic['val_imgs']))
    merge_split_dic['val_imgs'] = list(set(merge_split_dic['val_imgs']) - set(merge_split_dic['test_imgs']))
    logger.info("Image intersections: {}, {}, {}".format(
        len(set(merge_split_dic['train_imgs']).intersection(set(merge_split_dic['val_imgs']))),
        len(set(merge_split_dic['train_imgs']).intersection(set(merge_split_dic['test_imgs']))),
        len(set(merge_split_dic['val_imgs']).intersection(set(merge_split_dic['test_imgs'])))))

    # save to json file
    with open(output_file, "w") as outfile:
        json.dump(merge_split_dic, outfile, indent=4, sort_keys=True)
    logger.info("Merged split file saved to {}.".format(output_file))


if __name__ == '__main__':
    import argparse
    import os
    from datetime import datetime
    import time

    # Parse arguments
    parser = argparse.ArgumentParser(description="Dataset split Generation Script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('db_name', type=str, help="Dataset name", choices=dbac_data.DB_NAMES)
    parser.add_argument('db_path', type=str, help="Dataset directory path")
    subparsers = parser.add_subparsers(title='commands', dest='cmd_name', help='additional help')
    # Parse arguments for compute splits
    parser_splits = subparsers.add_parser('split', help='Compute dataset single expressions split.',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_splits.add_argument('output_path', type=str, help="Path to output dataset partition as .json file")
    parser_splits.add_argument('-ops', type=str, nargs='*', default=['NOT', 'AND', 'OR'], help="Operators")
    parser_splits.add_argument('-min_per_prim', type=int, default=10,
                               help='Minimum number of positive samples per primitive')
    parser_splits.add_argument('-min_per_exp', type=int, default=10,
                               help='Minimum number of positive samples per expression')
    parser_splits.add_argument('-num_exps', default=4000, type=int, help='Number of expression to sample.')
    parser_splits.add_argument('-min_pos_exp_test', type=int, default=10,
                               help='Minimum number of positive test samples per expression')
    parser_splits.add_argument('-nogroup', default=False, action='store_true',
                               help='Create expressions with grouped format.')
    # Parameters for compute combinations
    parser_comb = subparsers.add_parser('comb', help='Compute combined expressions from single expressions.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_comb.add_argument('split_file', type=str, help='Path to split file to use as single terms.')
    parser_comb.add_argument('complexity', type=int, help='Number of terms in the combined expressions.')
    parser_comb.add_argument('comb_file', type=str, help='Path to output the combination file.json')
    parser_comb.add_argument('form', type=str, choices=['D', 'C'], help='Normal form.')
    parser_comb.add_argument('-allow_not', default=False, action='store_true', help='Allow primitives negation.')
    parser_comb.add_argument('-min_pos_comb', type=int, nargs=2, default=[100, 10],
                             help='Minimum number of positives samples per combination. [train, test]')
    parser_comb.add_argument('-qtd_combs', type=int, nargs=2, default=[3000, 1000],
                             help='Number of sampled combinations. [train, test]')
    # Parameters for merge partition files
    parser_merge = subparsers.add_parser('merge', help='Merge split files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_merge.add_argument('merged_split_file', type=str, help='Path to save merged split file.')
    parser_merge.add_argument('split_files', type=str, nargs='*', help='List of split file to merge.')
    # Parameters for statistics
    parser_plot = subparsers.add_parser('stat', help='Plot split distributions.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_plot.add_argument('split_file', type=str, help='Path to split file to compute split distributions.')
    parser_plot.add_argument('-comb_file', type=str, help='Path to compositions file to compute distributions.')
    parser_plot.add_argument('-plot', default=False, action='store_true', help='Generate plot.')
    args = parser.parse_args()

    if args.cmd_name == 'split':
        # set up log system
        dbac_util.init_logging('_'.join(
            [os.path.splitext(args.output_path)[0], '{}.split.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))]))
        logger.info(args)
        # Compute dataset splits
        logger.info("Computing split for db {} ...".format(args.db_name))
        time.sleep(10)
        _compute_splits(args.db_name, args.db_path, args.ops, args.output_path, args.min_per_prim, args.min_per_exp,
                        args.num_exps, args.min_pos_exp_test, args.nogroup)
    elif args.cmd_name == 'comb':
        # set up log system
        dbac_util.init_logging('_'.join(
            [os.path.splitext(args.split_file)[0], '{}.comb.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))]))
        logger.info(args)
        logger.info("Computing combinations for {} and split file {}...".format(args.db_name, args.split_file))
        _compute_combs(args.db_name, args.db_path, args.split_file, args.complexity, args.comb_file, args.form,
                       args.allow_not,
                       args.min_pos_comb, args.qtd_combs)
    elif args.cmd_name == 'merge':
        dbac_util.init_logging('_'.join(
            [os.path.splitext(args.merged_split_file)[0], '{}.merge.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))]))
        logger.info(args)
        logger.info("Merging split files...")
        time.sleep(10)
        _merge_split(args.split_files, args.merged_split_file)
    elif args.cmd_name == 'stat':
        # set up log system
        dbac_util.init_logging('_'.join(
            [os.path.splitext(args.split_file)[0], '{}.stat.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))]))
        logger.info(args)
        logger.info("Computing statistics for precomputed split {}.".format(args.split_file))
        time.sleep(10)
        _compute_statistics(args.db_name, args.db_path, args.split_file, args.comb_file, plot=args.plot)
    else:
        raise ValueError('Not well formatted command line arguments. Parsed arguments {}'.format(args))
