import numpy as np
import matplotlib as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval


def get_pr_levels(ce: COCOeval):
    all_precision = ce.eval['precision']

    # pr = all_precision[:, :, 0, 0, 2]  # data for IoU@.50:.05:.95
    pr_50 = all_precision[0, :, 0, 0, 2]  # data for IoU@0.5
    pr_75 = all_precision[5, :, 0, 0, 2]  # data for IoU@0.75

    return pr_50, pr_75


def plot_curve_mpl(pr_50, pr_75):
    x = np.arange(0, 1.01, 0.01)
    plt.plot(x, pr_75, label='[{:.3f}] C75'.format(np.mean(pr_75)))
    plt.plot(x, pr_50, label='[{:.3f}] C50'.format(np.mean(pr_50)))

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # plt.title('Person')
    plt.xlabel('recall')
    plt.ylabel('precision')

    plt.legend()


def plot_pr_curve_tensorboard(p50, p75, writer=None, write_averages=False):
    if writer is None:
        writer = SummaryWriter()

    for x, y in zip(np.linspace(0.0,1.0,num=101), p50):
        writer.add_scalar('pr_curve/AP.5', y, x)

    for x, y in zip(np.linspace(0.0,1.0,num=101), p75):
        writer.add_scalar('pr_curve/AP.75', y, x)

    if write_averages:
        writer.add_scalar('metrics/AP.5', np.mean(p50), 0)
        writer.add_scalar('metrics/AP.5', np.mean(p75), 0)


def plot_pr_curve_altair(p50, p75):
    pr_dict = {
        '[{:.3f}] C75'.format(np.mean(p75)): p75,
        '[{:.3f}] C50'.format(np.mean(p50)): p50,
        'recall': np.arange(0, 1.01, 0.01)
    }
    df = pd.DataFrame.from_dict(pr_dict)
    list(df.columns[:2])
    df = df.melt(id_vars='recall', value_vars=list(df.columns[:2]), var_name='level', value_name='precision')

    options = {
        'x': 'recall',
        'y': 'precision',
        'color': 'level'
    }

    return df, options