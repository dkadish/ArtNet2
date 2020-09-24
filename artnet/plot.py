import numpy as np
import matplotlib as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def get_pr_levels(ce):
    all_precision = ce.eval['precision']

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


def plot_pr_curve_tensorboard(p50, p75, writer=None):
    if writer is None:
        writer = SummaryWriter()

    pr_dict = {
        '[{:.3f}] C75'.format(np.mean(p75)): p75,
        '[{:.3f}] C50'.format(np.mean(p50)): p50
    }

    data = zip(*[list(v) for v in pr_dict.values()])

    for i, v in enumerate(data):
        d = dict(zip(pr_dict.keys(), v))
        writer.add_scalars('pr_curve', d, i)


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