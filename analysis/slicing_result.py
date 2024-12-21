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
    pytorch_slo_violation_rate = {
        "data": {
            "No slice":[0.00124, 0.0242, 0.08628],
            "Dedicated slice":[0.000817, 0.00085639, 0.001034],
        },
        # "ylim": (0, -0.08),
        "xticks" : ["10", "20", "40"],
        "xlabel": "Iperf3 flows",
        "ylabel": "SLO violation rate",
        "name": "pytorch_slo_violation_rate"
    }

    pytorch_avg_processing_time = {
        "data": {
            "No slice":[0.147251, 0.21734, 0.330698],
            "Dedicated slice":[0.1074834, 0.1107978, 0.1142865],
        },
        # "ylim": (0, -0.08),
        "xticks" : ["10", "20", "40"],
        "xlabel": "Iperf3 flows",
        "ylabel": "Avg processing time (s)",
        "name": "pytorch_avg_processing_time"
    }

    coqui_avg_processing_time = {
        "data": {
            "No slice":[0.3719, 0.375869, 0.388238],
            "Dedicated slice":[0.368226, 0.367450, 0.3655692],
        },
        # "ylim": (0, -0.08),
        "xticks" : ["10", "20", "40"],
        "xlabel": "Iperf3 flows",
        "ylabel": "Avg processing time (s)",
        "name": "coqui_avg_processing_time"
    }


    for data in [pytorch_slo_violation_rate, pytorch_avg_processing_time, coqui_avg_processing_time]:
        print(f'-----{data["name"]}-----')
        # for key in data["data"]:
        #     print(f'{key}: {np.average(data["data"][key])}')

        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        # ax.invert_yaxis()
        # participants
        if data.get("xticks"):
            plt.xticks(range(len(data["xticks"])), data["xticks"])
        if data.get('ylim'):
            plt.ylim(*data.get('ylim'))
        # else:
        #     max_y = max(max(data['data']['No slice']), max(data['data']['Dedicated slice']))
        #     plt.ylim(0, 1.5*max_y)
        plt.xlabel(data["xlabel"])
        plt.ylabel(data["ylabel"])

        bar_plot(ax, data["data"], total_width=.8, single_width=.9)
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        # ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        ax.legend_.set_bbox_to_anchor((1, 1))
        plt.savefig(f'{scriptDir}/results/{data["name"]}.pdf', dpi=300, format='pdf')
        plt.clf()
