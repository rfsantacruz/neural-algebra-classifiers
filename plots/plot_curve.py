import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics


def _collect_curves(curves_files, curve_name, legends=None):
    print('{} Files found. files {}.'.format(len(curves_files), curves_files))
    # collect points
    curves = []
    for file in curves_files:
        data = np.load(file).item()
        curve = data.get(curve_name, None)
        if curve:
            curves.append(curve)
    print("{} curves found.".format(len(curves)))
    if legends:
        assert len(legends) == len(curves_files), "Legends must have the same number of curves"
    else:
        legends = [str(i) for i in range(len(curves))]
    print("Curve file legend map: {}".format(list(zip(curves_files, legends))))
    return curves, legends


def _plot_calib_curve(curves_files, legends=None):
    data_points, legends = _collect_curves(curves_files, 'calib_curve', legends)
    for curve_list, legend in zip(data_points, legends):
        for idx, (mean_values, true_freq) in enumerate(curve_list):
            plt.plot(mean_values, true_freq, 's-', label="{} - {}".format(legend, idx))
            plt.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated")

            params = {
                'axes.labelsize': 8,
                'font.size': 8,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.usetex': False,
                'figure.figsize': [4.5, 4.5]
            }
            plt.rcParams.update(params)
            plt.title('Calibration plots  (reliability curve)')
            plt.ylabel("Fraction of positives")
            plt.xlabel("Mean predicted value")
            plt.grid(linestyle='dashed', which='major', c='darkgray')
            plt.grid(linestyle='dotted', which='minor', c='lightgray')
            plt.ylim([-0.05, 1.05])
            plt.yticks(np.arange(0, 1.05, 0.2))
            plt.xlim([0.0, 1.0])
            plt.xticks(np.arange(0, 1.05, 0.1))
            plt.tight_layout()
            plt.show()


def _plot_pr(curves_files, legends=None):
    data_points, legends = _collect_curves(curves_files, 'micro_pr_curve', legends)
    cmap = plt.cm.get_cmap('Set1')
    for i, ((rec, prec, _), label) in enumerate(zip(data_points, legends)):
        auc = skmetrics.auc(rec, prec)
        plt.plot(rec, prec, c=cmap(float(i) / len(data_points)), label="{} (mAP = {:.3f})".format(label, auc))

    params = {
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5]
    }
    plt.rcParams.update(params)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(linestyle='dashed', which='major', c='darkgray')
    plt.grid(linestyle='dotted', which='minor', c='lightgray')
    plt.xlim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.05, 0.2))
    plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0, 1.05, 0.2))
    plt.legend(loc="upper right")
    plt.title("Precision-Recall Curves")
    plt.tight_layout()
    plt.show()


def _plot_roc(curves_files, legends=None):
    data_points, legends = _collect_curves(curves_files, 'micro_roc_curve', legends)
    cmap = plt.cm.get_cmap('Set1')
    for i, ((fpr, tpr, _), label) in enumerate(zip(data_points, legends)):
        auc = skmetrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, c=cmap(float(i)/len(data_points)), label="{} (AUC = {:.3f})".format(label, auc))
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    params = {
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5]
    }
    plt.rcParams.update(params)
    plt.grid(linestyle='dashed', which='major', c='darkgray')
    plt.grid(linestyle='dotted', which='minor', c='lightgray')
    plt.xlim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.05, 0.2))
    plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0, 1.05, 0.2))
    plt.minorticks_on()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('curves_files', type=str, nargs='*')
    parser.add_argument('-legends', default=None, type=str, nargs='*')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-roc', action='store_true', default=False)
    action.add_argument('-pr', action='store_true', default=False)
    action.add_argument('-calib', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    if args.roc:
        _plot_roc(args.curves_files, args.legends)
    elif args.pr:
        _plot_pr(args.curves_files, args.legends)
    elif args.calib:
        _plot_calib_curve(args.curves_files, args.legends)
    else:
        raise ValueError("Choose a valid graph.")
