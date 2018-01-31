import re
import numpy as np
import matplotlib.pyplot as plt


def _plot_primitives_boxplots(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    # read Primitives
    p = [re.findall('P=\d+', line) for line in lines]
    p = [v[0] for v in p if v]
    p = np.array([float(v.split('=')[1]) for v in p])
    p = [p[:int(len(p) / 2)], p[int(len(p) / 2):]]

    # read metrics
    metrics_cls, metrics_att = [], []
    for re_str in ['AP=\d+\.\d+', 'FScore=\d+\.\d+']:
        m = [re.findall(re_str, line) for line in lines]
        m = [v[0] for v in m if v]
        m = np.array([float(v.split('=')[1]) for v in m])
        m = [m[:int(len(m) / 2)], m[int(len(m) / 2):]]
        metrics_cls.append([m[0][p[0] < 200], m[1][p[1] < 200]])
        metrics_att.append([m[0][p[0] >= 200], m[1][p[1] >= 200]])

    # plot boxplots
    for ylabel, cls, att in zip(['Average Precision (Ap)', 'FScore'], metrics_cls, metrics_att):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.boxplot(cls, labels=['Train', 'Test'], showmeans=True)
        ax1.set_title('Objects', fontsize=16)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax2.boxplot(att, labels=['Train', 'Test'], showmeans=True)
        ax2.set_title('Attributes', fontsize=16)
        ax2.tick_params(axis='x', labelsize=12)
        f.tight_layout()
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for computing metrics for Neural Algebra of Classifiers models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_file', type=str, help='Paths to log file.')
    args = parser.parse_args()
    _plot_primitives_boxplots(args.log_file)
