import sys
sys.path.insert(0, '/home/rfsc/Projects/dbac/src')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import dbac_compute_metrics
from dbac_lib import dbac_expression
from sklearn import metrics as skmetrics
from scipy.interpolate import interp1d
from anytree import RenderTree


def _sample_pos_neg(result_file, samp_size=10):
    # read file
    res = np.load(result_file).item()
    pred = res['test_pred']
    gt = res['test_gt']
    imgs = res['test_imgs']
    exps = res['test_exps']

    for i in range(gt.shape[0]):
        print("Expression: {}".format(RenderTree(dbac_expression.list2exp_parse(exps[i]))))

        # find equal error rate threhold
        fpr, tpr, ths = skmetrics.roc_curve(gt[i], pred[i])
        eer = dbac_compute_metrics._compute_eer(gt[i].reshape(-1, 1), pred[i].reshape(-1, 1))[0]
        ths = interp1d(fpr, ths)(0.20)
        ths_pred = pred[i] >= ths

        # compute metrics
        tn, fp, fn, tp = skmetrics.confusion_matrix(gt[i], ths_pred).ravel()
        print("EER={}, THS={}, TPR={}, FPR={}, tn={}, fp={}, fn={}, tp={}".format(eer, ths, tp/(tp+fn), fp/(tn+fp), tn, fp, fn, tp))

        # sampling images for tp, fp, fn and tn
        tp_img_ids = np.where(ths_pred * gt[i])[0]
        fp_img_ids = np.where(ths_pred * np.logical_not(gt[i]))[0]
        fn_img_ids = np.where(np.logical_not(ths_pred) * gt[i])[0]
        tn_img_ids = np.where(np.logical_not(ths_pred) * np.logical_not(gt[i]))[0]
        assert (len(tp_img_ids), len(fp_img_ids), len(fn_img_ids), len(tn_img_ids)) == (tp, fp, fn, tn)
        img_samples = [[] if len(img_ids) == 0 else np.random.choice(imgs[img_ids], samp_size, replace=True) for img_ids in [tp_img_ids, fp_img_ids, fn_img_ids, tn_img_ids]]
        print("TP imgs: {}, FP imgs: {}, FN imgs: {}, Tn imgs:{}".format(*img_samples))

        # plot images
        f, grid = plt.subplots(4, samp_size)
        for p, samps in enumerate(img_samples):
            for q, samp in enumerate(samps):
                grid[p, q].imshow(mpimg.imread(samp), interpolation="bicubic")
                grid[p, q].xaxis.set_ticks([])
                grid[p, q].yaxis.set_ticks([])
        plt.show()


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=str)
    args = parser.parse_args()
    _sample_pos_neg(args.result_file)
