import json
from Util import ReviewDBReader
import matplotlib.pyplot as plt
from operator import add

REVIEW_STATUSES = ["CORRECT", "MODIFIED", "UNCERTAIN", "REVIEW_LATER"]
color_dict = {'green': '#5cb85c', 'blue': '#5bc0de', 'orange': '#f0ad4e', 'red': '#d9534f'}
color_status_dict = {'CORRECT': color_dict['green'], 'UNCERTAIN': color_dict['orange'], 'MODIFIED': color_dict['red'],
                     'REVIEW_LATER': color_dict['blue']}
color_phase_dict = {'immediate-early': color_dict['blue'], 'early': color_dict['green'], 'late': color_dict['orange'],
                    'latent': color_dict['red']}


class DataPlotter:
    def __init__(self, config_filepath):
        self.review_db_reader = None  # type: ReviewDBReader
        self.phases = None
        self.viruses = None
        self.output_directory_totals = None
        self.output_directory_per_virus = None

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

    def plot_total_phase(self):
        result_dict = {}
        for phase in self.phases:
            result_dict[phase] = self.review_db_reader.get_by('reviewed_phase', phase)

        bars = [len(result_dict[p]) for p in self.phases]

        plt.bar(range(len(result_dict)), bars,
                tick_label=[p for p in self.phases], color=[color_phase_dict[p] for p in self.phases])
        plt.title('Number of genes per phase (total over all viruses)')
        plt.ylabel('Number of genes')

        for i, v in enumerate(bars):
            plt.text(i, v - 4, str(v), color='white', ha='center', va='top')

        plt.savefig(f'{self.output_directory_totals}phase.png', dpi=300)
        plt.clf()

    def plot_total_status(self):
        result_dict = {}
        for status in REVIEW_STATUSES:
            result_dict[status] = self.review_db_reader.get_by('review_status', status)

        bars = [len(result_dict[s]) for s in REVIEW_STATUSES]

        plt.bar(range(len(result_dict)), bars, tick_label=[s for s in REVIEW_STATUSES],
                color=[color_status_dict[s] for s in REVIEW_STATUSES])
        plt.title('Number of genes per review status (total over all viruses)')
        plt.ylabel('Number of genes')

        for i, v in enumerate(bars):
            plt.text(i, v - 4, str(v), color='white', ha='center', va='top')

        plt.savefig(f'{self.output_directory_totals}review_status.png', dpi=300)
        plt.clf()

    def plot_per_virus_phase(self):

        bar_data = {}

        for phase in self.phases:
            data = self.review_db_reader.get_by('reviewed_phase', phase)
            virus_dict = {v: 0 for v in self.viruses}
            for d in data:
                virus_dict[d.virus] += 1

            bar_data[phase] = [virus_dict[v] for v in self.viruses]

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
                    if v > 2:
                        plt.text(j, v + bottom[j] - 0.5, str(v), color='white', ha='center', va='top', fontsize=8)
            else:
                bars.append(
                    plt.bar(range(len(self.viruses)), bar_data[phase], label=phase, color=color_phase_dict[phase]))
                for j, v in enumerate(bar_data[phase]):
                    if v > 2:
                        plt.text(j, v - 0.5, str(v), color='white', ha='center', va='top', fontsize=8)

        plt.xticks(range(len(self.viruses)), [v for v in self.viruses], fontsize=8)
        plt.legend()
        plt.title('Number of genes per phase per virus')
        plt.ylabel('Number of genes')
        plt.savefig(f'{self.output_directory_per_virus}phase.png', dpi=300)
        plt.clf()

    def plot_per_virus_status(self):

        bar_data = {}

        for status in REVIEW_STATUSES:
            data = self.review_db_reader.get_by('review_status', status)
            virus_dict = {v: 0 for v in self.viruses}
            for d in data:
                virus_dict[d.virus] += 1

            bar_data[status] = [virus_dict[v] for v in self.viruses]

        bars = []
        for i, status in enumerate(REVIEW_STATUSES):
            if len(bars) > 0:
                bottom = [0 for _ in self.viruses]
                for p in REVIEW_STATUSES[:i]:
                    bottom = list(map(add, bottom, bar_data[p]))
                bars.append(plt.bar(range(len(self.viruses)), bar_data[status], bottom=bottom, label=status,
                                    color=color_status_dict[status]))
                for j, v in enumerate(bar_data[status]):
                    if v > 6:
                        plt.text(j, v + bottom[j] - 2.5, str(v), color='white', ha='center', va='top', fontsize=8)
            else:
                bars.append(
                    plt.bar(range(len(self.viruses)), bar_data[status], label=status, color=color_status_dict[status]))
                for j, v in enumerate(bar_data[status]):
                    if v > 6:
                        plt.text(j, v - 2.5, str(v), color='white', ha='center', va='top', fontsize=8)

        plt.xticks(range(len(self.viruses)), [v for v in self.viruses], fontsize=8)
        plt.legend()
        plt.title('Number of genes per review status per virus')
        plt.ylabel('Number of genes')
        plt.savefig(f'{self.output_directory_per_virus}review_status.png', dpi=300)
        plt.clf()

    def plot(self):
        self.plot_total_phase()
        self.plot_total_status()
        self.plot_per_virus_phase()
        self.plot_per_virus_status()
