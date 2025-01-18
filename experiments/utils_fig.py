from typing import Callable, List, Text, Tuple


COLORS = [
    "#bc272d",  # red
    "#50ad9f",  # green
    "#0000a2",  # blue
    "#e9c716",  # yellow
]


def matplotlib_default():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.edgecolor"] = "None"
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["legend.borderpad"] = 0.1


def turn_off_spines(which=['top', 'right']):
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.spines[which].set_visible(False)


def plot_subset_selection(
        points: List[Tuple[List, List, Text]],
        filename=None,
        areas: List[Tuple[List, List, List]] = [],
        fn_extra: Callable = lambda _: None,
        colors: List[str] = COLORS,
):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import contextlib
    import os

    # either it's accuracy or clusters
    IS_CLUSTERS = any(
        y > 1
        for _, points_y, _ in points
        for y in points_y
    )

    matplotlib_default()
    plt.figure(figsize=(4, 2))

    if len(points) == 1:
        colors = ["black"]

    for (points_x, points_y, label), color in zip(points, colors):
        plt.plot(
            range(len(points_x)),
            points_y,
            marker="o",
            markersize=5,
            color=color,
            label=label,
            clip_on=True,
            # clip_on=False if min(points_y) > 0.65 else True,
            linewidth=2,
        )
    for (points_x, points_y_low, points_y_high), color in zip(areas, colors):
        plt.fill_between(
            range(len(points_x)),
            y1=points_y_low,
            y2=points_y_high,
            color=color,
            alpha=0.3,
            linewidth=0,
        )

    if IS_CLUSTERS:
        plt.ylabel("Cluster count" + " " * 5, labelpad=13)
    else:
        plt.ylabel("Pairwise accuracy" + " " * 5, labelpad=-1)
    plt.xlabel("Proportion of original data", labelpad=-1)

    plt.xticks(
        list(range(len(points_x)))[::3],
        [f"{x:.0%}" for x in points_x][::3],
    )

    ax = plt.gca()
    ax.spines[['top', 'right']].set_visible(False)
    # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))
    if not IS_CLUSTERS:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))

    fn_extra(plt.gca())

    plt.legend(
        loc="lower right",
        handletextpad=0.2,
        handlelength=1,
        labelspacing=0.2,
        facecolor="#ccc",
        scatteryoffsets=[0.5] * len(points),
    )

    if IS_CLUSTERS:
        plt.ylim(1.2, 5.0)
    else:
        plt.ylim(0.85, 0.97)
        # plt.ylim(0.75, 1.0)
    plt.tight_layout(pad=0.1)

    if filename:
        # temporarily change to the root directory
        with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/.."):
            os.makedirs("figures_pdf", exist_ok=True)
            os.makedirs("figures_svg", exist_ok=True)

            # save in files compatible with both LaTeX and Typst
            plt.savefig(f"figures_pdf/{filename}_{'clu' if IS_CLUSTERS else 'acc'}.pdf")
            plt.savefig(f"figures_svg/{filename}_{'clu' if IS_CLUSTERS else 'acc'}.svg")
    plt.show()
