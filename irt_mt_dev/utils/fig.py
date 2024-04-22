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


def plot_subsetacc(points, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    matplotlib_default()
    plt.figure(figsize=(3, 2))

    if len(points) == 1:
        colors = ["black"]
    else:
        colors = COLORS

    for (points_x, points_y, label), color in zip(points, colors):
        plt.plot(
            points_x,
            points_y,
            marker="o",
            markersize=5,
            color=color,
            label=label,
            clip_on=False if min(points_y) > 0.65 else True,
            linewidth=2,
        )

    plt.ylabel("Sys. rank accuracy" + " " * 5, labelpad=-5)
    plt.xlabel("Proportion of original data", labelpad=-2)

    ax = plt.gca()
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.legend(
        loc="lower right",
        handletextpad=0.2,
        handlelength=1,
        labelspacing=0.2,
        facecolor="#ccc",
        fancybox=False,
        edgecolor=None,
        borderpad=0.1,
        scatteryoffsets=[0.5]*len(points),
        fontsize=9
    )

    plt.ylim(0.7, 1)
    plt.tight_layout(pad=0.1)
    if filename:
        plt.savefig(f"figures/{filename}.svg")
    plt.show()
