import json
from Util import ReviewDBReader
import matplotlib.pyplot as plt
import pandas as pd
from operator import add

REVIEW_STATUSES = ["CORRECT", "MODIFIED", "UNCERTAIN"]
color_dict = {'green': '#5cb85c', 'blue': '#5bc0de', 'orange': '#f0ad4e', 'red': '#d9534f'}
color_status_dict = {'CORRECT': color_dict['green'], 'UNCERTAIN': color_dict['orange'], 'MODIFIED': color_dict['red'],
                     'REVIEW_LATER': color_dict['blue']}
color_phase_dict = {'immediate-early': color_dict['blue'], 'early': color_dict['green'], 'late': color_dict['orange'],
                    'latent': color_dict['red']}

virus_display_name = {"HSV_1": "HSV-1", "HSV_2": "HSV-2", "VZV": "VZV", "EBV": "EBV", "HCMV": "HCMV",
                      "HHV_6A": "HHV-6A", "HHV_6B": "HHV-6B", "HHV_7": "HHV-7", "KSHV": "KSHV"}


class DataPlotter:
    def __init__(self, config_filepath):
        self.review_db_reader = None  # type: ReviewDBReader
        self.phases = None
        self.viruses = None
        self.output_directory_totals = None
        self.output_directory_per_virus = None
        self.homology_filter_data = None

        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.output_directory_totals = config['output_directory_totals']
            self.output_directory_per_virus = config['output_directory_per_virus']

        with open(config['general_config']) as general_config_file:
            general_config = json.load(general_config_file)
            self.phases = general_config['phases']
            self.viruses = general_config['viruses']

        self.review_db_reader = ReviewDBReader(config['review_db_reader_config'])
        self.homology_filter_data = pd.read_csv(config['homology_filter_data'])

    def plot_total_phase(self):
        result_dict = {}
        for phase in self.phases:
            result_dict[phase] = self.homology_filter_data[self.homology_filter_data['label'] == phase]

        bars = [len(result_dict[p]) for p in self.phases]
        plt.figure(figsize=(5, 4))
        plt.bar(range(len(result_dict)), bars,
                tick_label=[p for p in self.phases], color=[color_phase_dict[p] for p in self.phases])
        # plt.title('Total number of genes per kinetic class')
        plt.ylabel('Number of genes')

        for i, v in enumerate(bars):
            plt.text(i, v - 4, str(v), color='white', ha='center', va='top', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_directory_totals}kinetic_class_total.png', dpi=300)
        plt.close()

    def plot_total_status(self):
        result_dict = {}
        for status in REVIEW_STATUSES:
            result_dict[status] = self.review_db_reader.get_by('review_status', status)

        bars = [len(result_dict[s]) for s in REVIEW_STATUSES]
        plt.figure(figsize=(5, 4))
        plt.bar(range(len(result_dict)), bars, tick_label=[s for s in REVIEW_STATUSES],
                color=[color_status_dict[s] for s in REVIEW_STATUSES])
        # plt.title('Total number of genes per review status')
        plt.ylabel('Number of genes')

        for i, v in enumerate(bars):
            plt.text(i, v - 4, str(v), color='white', ha='center', va='top', fontweight="bold")
        plt.tight_layout()
        plt.savefig(f'{self.output_directory_totals}review_status_total.png', dpi=300)
        plt.close()

    def plot_per_virus_phase(self):

        bar_data = {}

        for phase in self.phases:
            data = self.homology_filter_data[self.homology_filter_data['label'] == phase]
            virus_dict = {v: 0 for v in self.viruses}
            for i, d in data.iterrows():
                virus_dict[d['virus']] += 1

            bar_data[phase] = [virus_dict[v] for v in self.viruses]
        plt.figure(figsize=(5.5, 4))
        bars = []
        phases = list(self.phases.keys())
        for i, phase in enumerate(phases):
            if len(bars) > 0:
                bottom = [0 for _ in self.viruses]
                for p in phases[:i]:
                    bottom = list(map(add, bottom, bar_data[p]))
                bars.append(plt.bar(range(len(self.viruses)), bar_data[phase], bottom=bottom, label=phase,
                                    color=color_phase_dict[phase]))
                for j, v in enumerate(bar_data[phase]):
                    if v == 3:
                        plt.text(j, v + bottom[j] - 0.5, str(v), color='white', ha='center', va='top',
                                 fontweight='bold')
                    elif 3 < v:
                        plt.text(j, v + bottom[j] - 1, str(v), color='white', ha='center', va='top', fontweight='bold')
            else:
                bars.append(
                    plt.bar(range(len(self.viruses)), bar_data[phase], label=phase, color=color_phase_dict[phase]))
                for j, v in enumerate(bar_data[phase]):
                    if v == 3:
                        plt.text(j, v - 0.5, str(v), color='white', ha='center', va='top', fontweight='bold')
                    elif 3 < v:
                        plt.text(j, v - 1, str(v), color='white', ha='center', va='top', fontweight='bold')

        plt.xticks(range(len(self.viruses)), [virus_display_name[v] for v in self.viruses], fontsize=8)
        plt.legend()
        # plt.title('Number of genes per kinetic class for each virus')
        plt.ylabel('Number of genes')
        plt.tight_layout()
        plt.savefig(f'{self.output_directory_per_virus}kinetic_class_per_virus.png', dpi=300)
        plt.close()

    def plot_per_virus_status(self):

        bar_data = {}

        for status in REVIEW_STATUSES:
            data = self.review_db_reader.get_by('review_status', status)
            virus_dict = {v: 0 for v in self.viruses}
            for d in data:
                virus_dict[d.virus] += 1

            bar_data[status] = [virus_dict[v] for v in self.viruses]

        plt.figure(figsize=(5.5, 4))

        bars = []
        for i, status in enumerate(REVIEW_STATUSES):
            if len(bars) > 0:
                bottom = [0 for _ in self.viruses]
                for p in REVIEW_STATUSES[:i]:
                    bottom = list(map(add, bottom, bar_data[p]))
                bars.append(plt.bar(range(len(self.viruses)), bar_data[status], bottom=bottom, label=status,
                                    color=color_status_dict[status]))
                for j, v in enumerate(bar_data[status]):
                    if v > 7:
                        plt.text(j, v + bottom[j] - 2.5, str(v), color='white', ha='center', va='top',
                                 fontweight="bold")
            else:
                bars.append(
                    plt.bar(range(len(self.viruses)), bar_data[status], label=status, color=color_status_dict[status]))
                for j, v in enumerate(bar_data[status]):
                    if v > 7:
                        plt.text(j, v - 2.5, str(v), color='white', ha='center', va='top', fontweight="bold")

        plt.xticks(range(len(self.viruses)), [virus_display_name[v] for v in self.viruses], fontsize=8)
        plt.legend()
        # plt.title('Number of genes per review status for each virus')
        plt.ylabel('Number of genes')
        plt.tight_layout()
        plt.savefig(f'{self.output_directory_per_virus}review_status_per_virus.png', dpi=300)
        plt.close()

    def plot(self):
        self.plot_total_phase()
        self.plot_total_status()
        self.plot_per_virus_phase()
        self.plot_per_virus_status()
