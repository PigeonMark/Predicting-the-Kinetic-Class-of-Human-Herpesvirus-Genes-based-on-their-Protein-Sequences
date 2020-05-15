import json
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo
from time import time

from Util import data_from_protein, print_data_row

color_dict = {'green': '#5cb85c', 'blue': '#5bc0de', 'orange': '#f0ad4e', 'red': '#d9534f'}


class HomologyFilter:
    def __init__(self, config_filepath):
        self.data = None  # type: pd.DataFrame
        self.filtered_data = None  # type: pd.DataFrame
        self.identities = {}
        self.identity_file = None
        self.plot_directory = None
        self.csv_directory = None
        self.threshold = None
        self.inspection_threshold = None
        self.matrix = MatrixInfo.blosum62
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.data = pd.read_csv(config['input_csv_file'])
            self.identity_file = config['output_identity_file']
            self.plot_directory = config['output_plot_directory']
            self.csv_directory = config['output_csv_directory']
            self.threshold = config['threshold']
            self.inspection_threshold = config['inspection_threshold']

    def pairwise_align(self, req1, req2):
        alignment = pairwise2.align.globalds(req1['sequence'], req2['sequence'], self.matrix, -11, -1,
                                             one_alignment_only=True)
        seq1, seq2, score, begin, end = alignment[0]
        gaps = 0
        mismatches = 0
        matches = 0
        for i, a in enumerate(seq1):
            b = seq2[i]
            if a == '-' or b == '-':
                gaps += 1
            elif a != b:
                mismatches += 1
            else:
                matches += 1
        identity = float(matches) / (mismatches + gaps + matches)
        return identity

    def save_identity(self):
        pickle.dump(self.identities, open(self.identity_file, 'wb'))

    def load_identity(self):
        self.identities = pickle.load(open(self.identity_file, 'rb'))

    def __i_hist(self, inp, bins, title, filename):
        y, x, patch = plt.hist(inp, bins, color=color_dict['blue'])
        for i, v in enumerate(y):
            plt.text((x[i] + x[i + 1]) / 2., v, str(int(v)), color='black', ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.xticks(bins)
        plt.ylabel('Number of sequence pairs')
        plt.xlabel('Identity Score')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{self.plot_directory}{filename}.png', dpi=300)
        plt.clf()

    def identity_histogram(self):
        inp = [v2 for v in self.identities.values() for v2 in v.values()]
        bins = np.arange(0, 1.1, 0.1)
        self.__i_hist(inp, bins, 'Histogram of identity scores', 'identities_histogram')

        inp2 = [v for v in inp if v >= 0.3]
        bins2 = bins[3:]
        self.__i_hist(inp2, bins2, 'Histogram of identity scores (>30%)', 'identities_30_histogram')

    def calculate_identities(self):
        done = 0
        tot = int((len(self.data.index) * (len(self.data.index) - 1)) / 2)
        t = time()
        for i, geneA in self.data.iterrows():
            for j, geneB in self.data.iterrows():
                if j > i:
                    identity = self.pairwise_align(self.data.loc[i], self.data.loc[j])
                    if geneA['protein'] in self.identities:
                        self.identities[geneA['protein']][geneB['protein']] = identity
                    else:
                        self.identities[geneA['protein']] = {geneB['protein']: identity}

                    done += 1
                    if done % 100 == 0:
                        print(f"Done {done}/{tot} alignments in {(time() - t):.2f}s")

        self.save_identity()

    def get_homology_pairs(self, threshold=None):
        if threshold is None:
            threshold = self.threshold

        ret_list = []
        for geneA, genes in self.identities.items():
            for geneB, identity in genes.items():
                if identity >= threshold:
                    ret_list.append((geneA, geneB, identity))
        sorted_ret_list = sorted(ret_list, key=lambda value: value[2], reverse=True)
        return sorted_ret_list

    def print_homology(self, homology):
        print_data_row(data_from_protein(self.data, homology[0]))
        print_data_row(data_from_protein(self.data, homology[1]))
        print(f"Identity Score: {100 * homology[2]:.2f}%")
        print()

    def print_homologies_overview(self, threshold=None):
        homologies = self.get_homology_pairs(threshold)
        for hom in homologies:
            self.print_homology(hom)

    def inspect_homologies(self, threshold=None):
        if threshold is None:
            threshold = self.inspection_threshold

        identity_pairs = self.get_homology_pairs(threshold)
        for hom in identity_pairs:
            data_a = data_from_protein(self.data, hom[0])
            data_b = data_from_protein(self.data, hom[1])
            if data_a['label'] != data_b['label']:
                self.print_homology(hom)
                json_homology = {"proteins": [
                    {"virus": data_a['virus'], "name": data_a['protein'], "gene": "", "phase": data_a['label']},
                    {"virus": data_b['virus'], "name": data_b['protein'], "gene": "", "phase": data_b['label']}
                ], "reason": ""}
                print(json.dumps(json_homology))
                print()

    def next_index(self):
        return 0 if pd.isnull(self.filtered_data.index.max()) else self.filtered_data.index.max() + 1

    def filter_data(self):
        homology_pairs = self.get_homology_pairs()
        homologue_proteins = []
        for (geneA, geneB, identity) in homology_pairs:
            homologue_proteins.append(geneA)
            homologue_proteins.append(geneB)

        self.filtered_data = pd.DataFrame(columns=self.data.columns[1:])

        # Add all proteins not occurring in a homology pair
        for index, row in self.data.iterrows():
            if row['protein'] not in homologue_proteins:
                self.filtered_data.loc[self.next_index()] = row

        if len(homologue_proteins) != len(set(homologue_proteins)):
            print("ATTENTION: a protein occurred in 2 homologue pairs, filtered data might contain a duplicate or "
                  "homologue!")

        # Selectively add proteins occurring in homology pairs
        random.seed(0)
        for (geneA, geneB, identity) in homology_pairs:
            data_a = data_from_protein(self.data, geneA)
            data_b = data_from_protein(self.data, geneB)
            # add a random protein if label is the same
            if data_a['label'] == data_b['label']:
                choice = [data_a, data_b][random.randrange(2)]
                self.filtered_data.loc[self.next_index()] = choice
            # add both if label is different
            else:
                self.filtered_data.loc[self.next_index()] = data_a
                self.filtered_data.loc[self.next_index()] = data_b

    def save_filtered_data(self):
        self.filtered_data.to_csv(f'{self.csv_directory}filtered_features_{str(int(self.threshold * 100))}.csv')

    def filter(self):
        # self.calculate_identities()
        self.load_identity()
        self.identity_histogram()
        self.inspect_homologies()
        self.filter_data()
        self.save_filtered_data()
