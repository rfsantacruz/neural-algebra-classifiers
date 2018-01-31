import numpy as np
from sklearn import metrics
from sklearn import calibration
import pandas as pd
import os
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


def _compute_eer(y_true, y_score, average=None):
    def _find_eer_intersection(false_pos_rate, true_pos_rate):
        interp_func = interp1d(false_pos_rate, true_pos_rate)
        line = lambda x: -x + 1.
        cost = lambda x: np.absolute(interp_func(x) - line(x))
        eer = minimize_scalar(cost, bounds=(0.0, 1.0), method='Bounded')
        return eer.x

    eer = None
    if average is None or average == 'macro':
        eer = []
        for cls_idx in range(y_true.shape[1]):
            fpr, tpr, _ = metrics.roc_curve(y_true[:, cls_idx], y_score[:, cls_idx], pos_label=1, drop_intermediate=False)
            eer.append(_find_eer_intersection(fpr, tpr))
        eer = np.mean(eer) if average == 'macro' else np.array(eer)

    elif average == 'micro':
        fpr, tpr, _ = metrics.roc_curve(y_true.ravel(), y_score.ravel(), pos_label=1, drop_intermediate=False)
        eer = _find_eer_intersection(fpr, tpr)
    else:
        raise ValueError('Not supported averaged value. Try {}.'.format(['micro', 'macro', 'None']))
    return eer


def _read_result_file(test_file_path):
    # read data
    test_dict = np.load(test_file_path).item()
    train_gt, train_pred, train_exps = test_dict.get('train_gt', None), test_dict.get('train_pred', None), test_dict.get('train_exps', None)
    val_gt, val_pred, val_exps = test_dict.get('val_gt', None), test_dict.get('val_pred', None), test_dict.get('val_exps', None)
    test_gt, test_pred, test_exps = test_dict.get('test_gt', None), test_dict.get('test_pred', None), test_dict.get('test_exps', None)
    return train_exps, train_gt, train_pred, val_exps, val_gt, val_pred, test_exps, test_gt, test_pred


def _compute_metrics(test_files):
    # compute metrics for each file
    print("{} files has be found!".format(len(test_files)))
    for test_file in test_files:
        # read data
        train_exps, train_gt, train_pred, val_exps, val_gt, val_pred, test_exps, test_gt, test_pred = \
            _read_result_file(test_file)
        # compute metrics
        print("Processing file {}.".format(test_file))
        for key, exps, groud_truth, predicted in zip(['train', 'val', 'test'], [train_exps, val_exps, test_exps], [train_gt.T, val_gt.T, test_gt.T], [train_pred.T, val_pred.T, test_pred.T]):
            if (groud_truth is not None) and (predicted is not None) and (exps is not None):
                aps = metrics.average_precision_score(groud_truth, predicted, average=None)
                ap_macro = np.mean(aps)
                ap_micro = metrics.average_precision_score(groud_truth, predicted, average='micro')
                aucs = metrics.roc_auc_score(groud_truth, predicted, average=None)
                aucs_macro = np.mean(aucs)
                aucs_micro = metrics.roc_auc_score(groud_truth, predicted, average='micro')
                eers = _compute_eer(groud_truth, predicted, average=None)
                eers_macro = np.mean(eers)
                eers_micro = _compute_eer(groud_truth, predicted, average='micro')
                supports = np.sum(groud_truth, axis=0)
                df = pd.DataFrame(np.vstack([aps, aucs, eers, supports]).T, index=exps, columns=['Ap', 'AUC', 'EER', 'Support'])
                df.loc['macro'] = [ap_macro, aucs_macro, eers_macro, np.mean(supports)]
                df.loc['micro'] = [ap_micro, aucs_micro, eers_micro, np.std(supports)]
                save_path = '_'.join([os.path.splitext(test_file)[0], "{}.metrics.csv".format(key)])
                df.to_csv(save_path)
                print("> Result {} save to {}:\n".format(key, save_path))
                print(df)


def _compute_curves(test_files):
    # compute plots for each file
    print("{} files has be found!".format(len(test_files)))
    for test_file in test_files:
        # read data
        train_exps, train_gt, train_pred, val_exps, val_gt, val_pred, test_exps, test_gt, test_pred = \
            _read_result_file(test_file)
        # compute metrics
        print("Processing file {}.".format(test_file))
        for key, exps, groud_truth, predicted in zip(['train', 'val', 'test'], [train_exps, val_exps, test_exps],
                                                     [train_gt.T, val_gt.T, test_gt.T],
                                                     [train_pred.T, val_pred.T, test_pred.T]):
            if (groud_truth is not None) and (predicted is not None) and (exps is not None):
                plots_dic = {'exps': [], 'pr_curve': [], 'roc_curve': [], 'calib_curve': [], 'micro_pr_curve': None, 'micro_roc_curve': None}
                for e_idx, e in enumerate(exps):
                    plots_dic['exps'].append(e)
                    prec, rec, pr_thr = metrics.precision_recall_curve(groud_truth[:, e_idx], predicted[:, e_idx], pos_label=1)
                    plots_dic['pr_curve'].append((rec, prec, pr_thr))
                    fpr, tpr, roc_thr = metrics.roc_curve(groud_truth[:, e_idx], predicted[:, e_idx], pos_label=1)
                    plots_dic['roc_curve'].append((fpr, tpr, pr_thr))
                    normalize = np.max(predicted[:, e_idx]) > 1.0 or np.min(predicted[:, e_idx]) < 0.0
                    pos_freqs, mean_preds = calibration.calibration_curve(groud_truth[:, e_idx], predicted[:, e_idx], normalize=normalize ,n_bins=10)
                    plots_dic['calib_curve'].append((mean_preds, pos_freqs))
                prec, rec, pr_thr = metrics.precision_recall_curve(groud_truth.ravel(), predicted.ravel(), pos_label=1)
                plots_dic['micro_pr_curve'] = (rec, prec, pr_thr)
                fpr, tpr, roc_thr = metrics.roc_curve(groud_truth.ravel(), predicted.ravel(), pos_label=1)
                plots_dic['micro_roc_curve'] = (fpr, tpr, pr_thr)
                save_path = '_'.join([os.path.splitext(test_file)[0], "{}.curves.npy".format(key)])
                np.save(save_path, plots_dic)
                print("> Result {} save to {}:\n".format(key, save_path))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Script for computing metrics for Neural Algebra of Classifiers models", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='commands', dest='cmd_name', help='additional help')
    parser_metrics = subparsers.add_parser('metrics', help='Compute numeric metrics',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_metrics.add_argument('test_files', type=str, nargs='*', help='Paths to test files.')
    parser_plots = subparsers.add_parser('curves', help='Compute graphic metrics',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_plots.add_argument('test_files', type=str, nargs='*', help='Paths to test files.')
    args = parser.parse_args()

    if args.cmd_name == 'metrics':
        _compute_metrics(args.test_files)

    elif args.cmd_name == 'curves':
        _compute_curves(args.test_files)

    else:
        raise ValueError('Command not well formatted.')