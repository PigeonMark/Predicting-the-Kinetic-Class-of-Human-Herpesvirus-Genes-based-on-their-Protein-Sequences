"""
Adapted from https://github.com/bittremieux/TCR-Classifier/blob/master/tcr_classifier_v2.ipynb
"""
import json
import itertools
import numpy as np
import pandas as pd
from pyteomics import electrochem, mass, parser


class FeatureExtraction:
    def __init__(self, config_filepath):
        self.physchem_properties = {}
        self.data_frame = None  # type: pd.DataFrame
        self.output_csv_directory = None
        self.feature_possibilities = None
        self.aa_categories = {}
        self.aa_to_category = {}
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.output_csv_directory = config['output_csv_directory']
            self.physchem_properties = {'basicity': config['basicity'],
                                        'hydrophobicity': config['hydrophobicity'],
                                        'helicity': config['helicity'],
                                        'mutation stability': config['mutation_stability']}

            self.data_frame = pd.read_csv(config['input_csv_file'], index_col=0)
            self.feature_possibilities = config['feature_possibilities']
            self.aa_categories = config['aa_categories']
            for cat, aas in self.aa_categories.items():
                for aa in aas:
                    self.aa_to_category[aa] = cat

    def add_length(self):
        self.data_frame['length'] = self.data_frame['sequence'].apply(lambda sequence: parser.length(sequence))

    def add_aa_counts(self):
        aa_counts = pd.DataFrame.from_records(
            [parser.amino_acid_composition(sequence) for sequence in self.data_frame['sequence']]) \
            .fillna(0, downcast='infer')
        aa_counts.columns = ['{} count'.format(column) for column in aa_counts.columns]
        self.data_frame = pd.concat([self.data_frame, aa_counts], axis=1)

    def add_relative_counts(self):
        record_list = []
        for sequence in self.data_frame['sequence']:
            record_list.append(
                {k: v / float(parser.length(sequence)) for k, v in parser.amino_acid_composition(sequence).items()})
        aa_counts = pd.DataFrame.from_records(record_list).fillna(0, downcast='infer')
        aa_counts.columns = ['{} relative_count'.format(column) for column in aa_counts.columns]
        self.data_frame = pd.concat([self.data_frame, aa_counts], axis=1)

    def add_physchem_properties(self):
        def physchem_properties_function(sequence):
            if prop_name == 'mutation stability':
                return np.mean(list(prop_lookup[aa] for aa in sequence.replace('X', '')))
            else:
                return np.mean(list(prop_lookup[aa] for aa in sequence))

        # average physico-chemical properties
        for prop_name, prop_lookup in self.physchem_properties.items():
            self.data_frame[f'average {prop_name}'] = self.data_frame['sequence'].apply(physchem_properties_function)

    def add_mass(self):
        self.data_frame['mass'] = self.data_frame['sequence'].apply(
            lambda sequence: mass.fast_mass(sequence.replace('X', '')))

    def add_p_i(self):
        self.data_frame['pI'] = self.data_frame['sequence'].apply(lambda sequence: electrochem.pI(sequence))

    def add_windowed(self, size):
        categories_string_data = self.data_frame['sequence'].apply(
            lambda sequence: ''.join([self.aa_to_category[aa] for aa in sequence.replace('X', '')]))

        category_n_tuples = [''.join(prod) for prod in itertools.product(self.aa_categories.keys(), repeat=size)]
        records = []
        for cat_string in categories_string_data:
            record_dict = {}
            for comb in category_n_tuples:
                record_dict[comb] = comb in cat_string

            records.append(record_dict)
        self.data_frame = pd.concat([self.data_frame, pd.DataFrame.from_records(records)], axis=1)

    def save(self, name):
        self.data_frame.to_csv(
            f"{self.output_csv_directory}features_{name}.csv")

    def extract(self, name):
        features = self.feature_possibilities[name]
        if 'length' in features:
            self.add_length()
        if 'counts' in features:
            self.add_aa_counts()
        if 'relative-counts' in features:
            self.add_relative_counts()
        if 'physchem' in features:
            self.add_physchem_properties()
        if 'mass' in features:
            self.add_mass()
        if 'pI' in features:
            self.add_p_i()

        for feature in features:
            if feature[2:] == 'windowed':
                self.add_windowed(int(feature[0]))

        # Move label to last position in dataframe
        self.data_frame = self.data_frame[[c for c in self.data_frame.columns if c != 'label'] + ['label']]

        print(self.data_frame)

        self.save(name)
