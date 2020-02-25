import csv
import json
import pickle

import networkx as nx

from Util import add_to_near_occ_dict
from Keywords import KeywordBuilder


class Combiner:
    def __init__(self, config_filepath):
        self.viruses = None
        self.keywords = {}
        self.index = {}
        self.combined_counts = {}
        self.sorted_combined_counts = {}
        self.paper_counts = {}
        self.normalized = {}
        self.cutted_index = {}
        self.phases = None
        self.min_score = 2.5
        self.min_diff = 0.05
        self.config = None
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)
            with open(self.config['general_config']) as general_config_file:
                general_config = json.load(general_config_file)
                self.viruses = general_config["viruses"]
                self.phases = general_config["phases"]

            for virus in self.viruses:
                self.index[virus] = pickle.load(open(self.config['input_files'][virus], 'rb'))
                self.keywords[virus] = KeywordBuilder(self.config['keywords_config']).get_keywords(virus)
            if 'min_score' in self.config:
                self.min_score = self.config['min_score']
            if 'min_diff' in self.config:
                self.min_diff = self.config['min_diff']

    def combine_counts_all_papers(self, virus_name):
        """
        Combine the counts of all the papers
        :param virus_name:  The virus to combine_all_viruses the counting results for
        :return:            A dictionary containing the combined counts:
                            {'kw1': {'phase1': score1, 'phase2': score2, ...}, 'kw2': ...}
        """
        index = self.index[virus_name]

        self.combined_counts[virus_name] = {}
        self.paper_counts[virus_name] = {}
        for paper, proteins in index.items():
            for protein, phases in proteins.items():
                for phase, count in phases.items():
                    add_to_near_occ_dict(count, protein, phase, self.combined_counts[virus_name])
                    add_to_near_occ_dict(1, protein, phase, self.paper_counts[virus_name])

    def combine_counts_alternate_names(self, virus_name):
        all_keys, name_to_headers, header_row = self.keywords[virus_name]
        g = nx.DiGraph()

        for kw in self.combined_counts[virus_name].keys():
            for hdr in name_to_headers[kw]:
                if kw != hdr:
                    g.add_edge(kw, hdr)

        connectedComponents = list(g.subgraph(c) for c in nx.connected_components(nx.Graph(g)))
        cc_dict = {}
        for i, connectedComponent in enumerate(connectedComponents):
            component_name = ""
            for node in sorted(connectedComponent.nodes()):
                if len(component_name) == 0:
                    component_name += node
                else:
                    component_name += "_" + node
            cc_dict[component_name] = list(connectedComponent.nodes())

        combined_counts = {}
        combined_paper_counts = {}
        for group_name, kws in cc_dict.items():
            for kw in kws:
                if kw in self.combined_counts[virus_name]:
                    for phase, count in self.combined_counts[virus_name][kw].items():
                        add_to_near_occ_dict(count, group_name, phase, combined_counts)
                        add_to_near_occ_dict(self.paper_counts[virus_name][kw][phase], group_name, phase,
                                             combined_paper_counts)

        self.combined_counts[virus_name] = combined_counts
        self.paper_counts[virus_name] = combined_paper_counts

    def sort_by_highest_value(self, virus_name):
        self.sorted_combined_counts[virus_name] = sorted(self.combined_counts[virus_name].items(),
                                                         key=lambda keyword: max(
                                                             [keyword[1].get(phase, 0) for phase in self.phases]),
                                                         reverse=True)

    def normalize(self, virus_name):
        self.normalized[virus_name] = []
        for kw, ph in self.sorted_combined_counts[virus_name]:
            total = sum([ph.get(phase, 0) for phase in self.phases])
            self.normalized[virus_name].append((kw, {phase: ph.get(phase, 0) / float(total) for phase in self.phases}))

    def to_csv(self, virus_name):
        with open(self.config['output_csv_files'][virus_name], "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ['Keyword'] + [out for phase in self.phases for out in [phase, '#', '    ']] + [phase + " (%)" for phase
                                                                                                in self.phases])
            for i, (kw, p) in enumerate(self.sorted_combined_counts[virus_name]):
                csv_writer.writerow(
                    [kw] + [out for phase in self.phases for out in
                            [f"{p.get(phase, 0):.2f}", f"{self.paper_counts[virus_name][kw].get(phase, 0)}", '']] + [
                        f"{self.normalized[virus_name][i][1].get(phase, 0) * 100:.2f}" for phase in self.phases])

    def cut(self, virus_name):
        self.cutted_index[virus_name] = []
        for idx, (kw, phases) in enumerate(self.sorted_combined_counts[virus_name]):
            scores = [score for phase, score in phases.items()]
            sorted_ns = sorted([score for phase, score in self.normalized[virus_name][idx][1].items()], reverse=True)
            if max(scores) > self.min_score and (sorted_ns[0] - sorted_ns[1]) > self.min_diff:
                self.cutted_index[virus_name].append((kw, phases))

    def save(self, virus_name):
        pickle.dump((self.combined_counts[virus_name], self.sorted_combined_counts[virus_name],
                     self.normalized[virus_name], self.paper_counts[virus_name], self.cutted_index[virus_name]),
                    open(self.config["output_raw_files"][virus_name], 'wb'))

    def combine_all_viruses(self):
        for virus in self.viruses:
            self.combine_counts_all_papers(virus)
            self.combine_counts_alternate_names(virus)
            self.sort_by_highest_value(virus)
            self.normalize(virus)
            self.cut(virus)
            self.to_csv(virus)
            self.save(virus)
