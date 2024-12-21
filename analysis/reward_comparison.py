import argparse
import copy
import os
import warnings
import time
import json
import numpy as np
import itertools
import scienceplots
import matplotlib.pyplot as plt


plt.style.use(['science','ieee', 'no-latex'])
plt.rcParams.update({'font.size': 12, 'hatch.linewidth': 0.25, 'hatch.color': 'gray', 'font.serif': 'DejaVu Sans',})
scriptDir = os.path.dirname(os.path.realpath(__file__))





def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatches = ['xx', '..', '++', '//']
    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []
    max_y = 0

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            y_avg = np.mean(y)
            bar = ax.bar(x + x_offset, y_avg, yerr=np.std(y) if hasattr(y, "__len__") else None, ecolor="firebrick", capsize=3, hatch=hatches[i % len(hatches)], width=bar_width * single_width, fill=False)
            if max_y > y_avg: max_y = y_avg

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


if __name__ == '__main__':
    gpu_8_ratio_04_015_015 = {
        "data": {
            "DRL":[-0.02559, -0.02052, -0.02727],
            "Rule-based":[-0.02819, -0.02221, -0.02733],
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_8_ratio_04_015_015"
    }

    gpu_8_ratio_06_02_02 = {
        "data": {
            "DRL":[-0.02654, -0.02620, -0.02967],
            "Rule-based":[-0.02640, -0.04117, -0.03297],
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_8_ratio_06_02_02"
    }


    gpu_8_ratio_08_03_03 = {
        "data": {
            "DRL":[-0.05678, -0.05223, -0.09085],
            "Rule-based":[-0.06719, -0.09829, -0.12566]
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_8_ratio_08_03_03"
    }

    gpu_16_ratio_08_03_03 = {
        "data": {
            "DRL":[-0.02638, -0.01977, -0.02909],
            "Rule-based":[-0.02924, -0.02134, -0.02962]
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_16_ratio_08_03_03"
    }

    gpu_32_ratio_08_03_03 = {
        "data": {
            "DRL":[-0.01842, -0.01416, -0.01792],
            "Rule-based":[-0.01930, -0.01602, -0.01851]
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_32_ratio_08_03_03"
    }

    gpu_64_ratio_3_1_1 = {
        "data": {
            "DRL":[-0.02633, -0.02454, -0.02525],
            "Rule-based":[-0.02429, -0.03031, -0.02930]
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_64_ratio_3_1_1"
    }

    gpu_128_ratio_6_2_2 = {
        "data": {
            "DRL":[-0.02654, -0.02620, -0.02967],
            "Rule-based":[-0.02640, -0.04117, -0.03297]
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "gpu_128_ratio_6_2_2"
    }

    real_testbed = {
        "data": {
            "DRL":[-0.0341, -0.0210, -0.0287],
            "Rule-based":[-0.0353, -0.0308, -0.0291]
        },
        # "ylim": (0, -0.08),
        "xticks" : ["Chatbot", "IC", "TTS"],
        "xlabel": "AI service",
        "ylabel": "Reward",
        "name": "real_testbed"
    }


    for data in [gpu_8_ratio_04_015_015, gpu_8_ratio_06_02_02, gpu_8_ratio_08_03_03, gpu_16_ratio_08_03_03, gpu_32_ratio_08_03_03, gpu_64_ratio_3_1_1, gpu_128_ratio_6_2_2, real_testbed]:
        print(f'-----{data["name"]}-----')
        # for key in data["data"]:
        #     print(f'{key}: {np.average(data["data"][key])}')

        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        ax.invert_yaxis()
        # participants
        if data.get("xticks"):
            plt.xticks(range(len(data["xticks"])), data["xticks"])
        if data.get('ylim'):
            plt.ylim(*data.get('ylim'))
        # else:
        #     min_neg_y = min(min(data['data']['DRL']), min(data['data']['Rule-based']))
        #     plt.ylim(0, 1.5*min_neg_y)
        # plt.xlabel(data["xlabel"])
        plt.ylabel(data["ylabel"])

        bar_plot(ax, data["data"], total_width=.8, single_width=.9)
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        # ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        ax.legend_.set_bbox_to_anchor((1, 1))
        plt.savefig(f'{scriptDir}/results/{data["name"]}.pdf', dpi=300, format='pdf')
        plt.clf()
