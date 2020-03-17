import json
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo

from Util import data_from_protein, print_data_row


class HomologyFilter:
    def __init__(self, config_filepath):
        self.data = None  # type: pd.DataFrame
        self.identities = {}
        self.identity_file = None
        self.plot_directory = None
        self.matrix = MatrixInfo.blosum62
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.data = pd.read_csv(config['input_csv_file'])
            self.identity_file = config['output_identity_file']
            self.plot_directory = config['output_plot_directory']

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
        y, x, patch = plt.hist(inp, bins)
        for i, v in enumerate(y):
            plt.text((x[i] + x[i + 1]) / 2., v, str(int(v)), color='black', ha='center', va='bottom')
        plt.xticks(bins)
        plt.ylabel('Number of pairs')
        plt.xlabel('Identity Score')
        plt.title(title)
        plt.savefig(f'{self.plot_directory}{filename}.png', dpi=300)
        plt.clf()

    def identity_histogram(self):
        inp = [v2 for v in self.identities.values() for v2 in v.values()]
        bins = np.arange(0, 1.1, 0.1)
        self.__i_hist(inp, bins, 'Identity Scores for gene pairs', 'identities_histogram')

        inp2 = [v for v in inp if v >= 0.3]
        bins2 = bins[3:]
        self.__i_hist(inp2, bins2, 'Identity Scores (>30%) for gene pairs', 'identities_30_histogram')

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

    def get_pairs_identity_above(self, threshold):
        ret_list = []
        for geneA, genes in self.identities.items():
            for geneB, identity in genes.items():
                if identity >= threshold:
                    ret_list.append((geneA, geneB, identity))
        sorted_ret_list = sorted(ret_list, key=lambda value: value[2], reverse=True)
        return sorted_ret_list

    def print_homologs_overview(self):
        homologs = self.get_pairs_identity_above(0.4)
        for hom in homologs:
            print_data_row(data_from_protein(self.data, hom[0]))
            print_data_row(data_from_protein(self.data, hom[1]))
            print(f"Identity Score: {100 * hom[2]:.2f}%")
            print()

    def filter(self):
        # self.calculate_identities()
        self.load_identity()
        self.identity_histogram()
        self.print_homologs_overview()
