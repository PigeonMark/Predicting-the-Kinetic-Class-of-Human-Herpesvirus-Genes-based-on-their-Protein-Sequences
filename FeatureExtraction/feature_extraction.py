"""
Adapted from https://github.com/bittremieux/TCR-Classifier/blob/master/tcr_classifier_v2.ipynb
"""
import json
import operator
import pickle

import numpy as np
import pandas as pd
from pyteomics import electrochem, mass, parser

from Keywords import KeywordBuilder
from ProteinCollecting import ProteinCollector


class FeatureExtraction:
    def __init__(self, config_filepath):
        self.physchem_properties = {}
        self.dataframe = {}
        self.keywords = {}
        self.viruses = None
        self.cutted_index = {}
        self.protein_collector = None
        self.features = {}
        self.output_csv_directory = None
        self.general_config = None
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.protein_collector = ProteinCollector(config['protein_collector_config'])
            self.output_csv_directory = config['output_csv_directory']
            self.physchem_properties = {'basicity': config['basicity'],
                                        'hydrophobicity': config['hydrophobicity'],
                                        'helicity': config['helicity'],
                                        'mutation stability': config['mutation_stability']}
            with open(config['general_config']) as general_config_file:
                self.general_config = json.load(general_config_file)
                self.viruses = self.general_config["viruses"]

            for virus in self.viruses:
                self.keywords[virus] = KeywordBuilder(config['keywords_config']).get_keywords(virus)
                self.cutted_index[virus] = \
                pickle.load(open(f"{config['input_directory']}{virus}_{self.general_config['distance']}.p", 'rb'))[4]

    def create_dataframe(self, virus_name):
        all_keys, name_to_headers, header_row = self.keywords[virus_name]

        self.dataframe[virus_name] = pd.DataFrame(columns=['protein_group', 'protein', 'sequence', 'label'])
        for i, (kws, phases) in enumerate(self.cutted_index[virus_name]):
            kw_lst = kws.split('_')
            max_evidence = 6
            max_uniprot_id = ''
            max_sequence = ''
            for kw in kw_lst:
                if kw in header_row:
                    sequence, evidence_level = self.protein_collector.read_protein_sequence(kw)
                    if evidence_level < max_evidence:
                        max_evidence = evidence_level
                        max_uniprot_id = kw
                        max_sequence = sequence

            self.dataframe[virus_name].loc[i] = [kws, max_uniprot_id, str(max_sequence),
                                                 max(phases.items(), key=operator.itemgetter(1))[0]]

    def compute_features(self, virus_name):
        """
        Creates feature vector representations for each TCR beta sequence in a pandas `DataFrame`.
        Each row/TCR beta is expected to be made up of a V-gene, J-gene and CDR3 sequence.

        Sequences are turned into feature vectors based on the present V- and J gene as well
        as physicochemical properties of the CDR3 sequence.

        Args:
            - data: The pandas `DataFrame` containing TCR beta sequences.

        Returns:
            A pandas `DataFrame` in which rows contain feature information on a TCR beta sequence.
        """

        features_list = [self.dataframe[virus_name]['protein_group'], self.dataframe[virus_name]['protein']]

        # non-positional features (i.e. over the whole sequence)

        # sequence length
        features_list.append(
            self.dataframe[virus_name]['sequence'].apply(lambda sequence: parser.length(sequence)).to_frame()
                .rename(columns={'sequence': 'length'}))

        # number of occurences of each amino acid
        aa_counts = pd.DataFrame.from_records(
            [parser.amino_acid_composition(sequence) for sequence in self.dataframe[virus_name]['sequence']]).fillna(0)
        aa_counts.columns = ['{} count_all_viruses'.format(column) for column in aa_counts.columns]
        features_list.append(aa_counts)

        # average physico-chemical properties
        for prop_name, prop_lookup in self.physchem_properties.items():
            features_list.append(self.dataframe[virus_name]['sequence'].apply(
                lambda sequence: np.mean(list(prop_lookup[aa] for aa in sequence)))
                                 .to_frame().rename(columns={'sequence': 'average {}'.format(prop_name)}))

        # peptide mass
        features_list.append(self.dataframe[virus_name]['sequence'].apply(
            lambda sequence: mass.fast_mass(sequence)).to_frame().rename(columns={'sequence': 'mass'}))

        # pI
        features_list.append(self.dataframe[virus_name]['sequence'].apply(
            lambda sequence: electrochem.pI(sequence)).to_frame().rename(columns={'sequence': 'pI'}))

        features_list.append(self.dataframe[virus_name]['label'])

        self.features[virus_name] = pd.concat(features_list, axis=1)

    def save_features(self, virus_name):
        self.features[virus_name].to_csv(
            f"{self.output_csv_directory}{virus_name}_{self.general_config['distance']}.csv")

    def extract(self):
        for virus in self.viruses:
            self.create_dataframe(virus)
            self.compute_features(virus)
            self.save_features(virus)
