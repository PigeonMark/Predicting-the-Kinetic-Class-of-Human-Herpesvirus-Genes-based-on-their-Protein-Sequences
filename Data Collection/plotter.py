import matplotlib.pyplot as plt
import numpy as np
from dataCollection import combine_counts


def plot_combined_counts(combined_counts):
    fig, ax = plt.subplots()

    bar_width = 0.25
    opacity = 0.4

    counts_list = list(combined_counts.items())

    counts_list_1 = counts_list[:5]

    ie_list = [phases.get('immediate-early', 0) for kw, phases in counts_list_1]
    early_list = [phases.get('early', 0) for kw, phases in counts_list_1]
    late_list = [phases.get('late', 0) for kw, phases in counts_list_1]

    index = np.arange(len(counts_list_1))

    bars_ie = ax.bar(index, ie_list, bar_width, alpha=opacity, color='b', label='IE')
    bars_early = ax.bar(index + bar_width, early_list, bar_width, alpha=opacity, color='g', label='Early')
    bars_late = ax.bar(index + 2*bar_width, late_list, bar_width, alpha=opacity, color='r', label='Late')

    ax.set_xticks(index + bar_width)
    ax.set_xticklabels([kw for kw, phases in counts_list_1])

    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    i_file = "hsv-1_comm_use.I-N_20191021-173402.p_keywords_10298.csv_10.p"

    combined_counts = combine_counts(i_file)

    plot_combined_counts(combined_counts)

