"""
Adapted from https://github.com/bittremieux/TCR-Classifier/blob/master/tcr_classifier_v2.ipynb
"""
import json
import itertools
import pickle

import numpy as np
import pandas as pd
from pyteomics import electrochem, mass, parser
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from Util import compose_filename, compose_configuration


class FeatureExtraction:
    def __init__(self, config_filepath):
        self.physchem_properties = {}
        self.data_frame = None  # type: pd.DataFrame
        self.output_csv_directory = None
        self.output_pca_directory = None
        self.feature_possibilities = None
        self.filter_phase = False
        self.standardization = False
        self.aa_categories = {}
        self.aa_to_category = {}
        self.skip_features = None
        self.pca_features = False
        self.n_pca = None
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.output_csv_directory = config['output_csv_directory']
            self.output_pca_directory = config['output_pca_directory']
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

            self.filter_phase = config['filter_latent']
            self.standardization = config['standardization']
            self.skip_features = config['skip-features']

            self.pca_features = config['pca_features']
            self.n_pca = config.get('n-pca', None)

    def filter_original_viruses(self):
        self.data_frame = self.data_frame[
            self.data_frame.virus.isin(["HSV_1", "HSV_2", "VZV", "EBV", "HCMV"])].reset_index(drop=True)

    def filter_original_phases(self):
        self.data_frame = self.data_frame[
            self.data_frame.label.isin(["immediate-early", "early", "late"])].reset_index(drop=True)

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

    def standardize(self):
        columns = [col for col in self.data_frame.columns if col not in self.skip_features]
        self.data_frame[columns] = scale(self.data_frame[columns])

    def apply_pca(self, n):
        columns_to_pca = [col for col in self.data_frame.columns if col not in self.skip_features]
        columns_to_keep = [col for col in self.data_frame.columns if col in self.skip_features]

        pca = PCA(n_components=n)
        pca.fit(self.data_frame[columns_to_pca])

        pickle.dump(pca, open(f"{self.output_pca_directory}{str(n)}-pca.p", 'wb'))

        new_features = pca.transform(self.data_frame[columns_to_pca])
        nf_df = pd.DataFrame(new_features, columns=[f'comp_{i}' for i in range(new_features.shape[1])])

        self.data_frame = pd.concat([self.data_frame[columns_to_keep], nf_df], axis=1)

    def save(self, name, n_pca):
        filename = compose_filename(self.output_csv_directory, self.filter_phase, self.standardization, n_pca,
                                    'features', name, 'csv')
        self.data_frame.to_csv(filename)

    def _extract(self, name, n_pca):
        print(compose_configuration('Extracting features', self.filter_phase, self.standardization, n_pca, name))

        if self.filter_phase:
            self.filter_original_phases()

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

        if self.standardization:
            self.standardize()

    def finish(self, name, n_pca):
        # Move label to last position in dataframe
        self.data_frame = self.data_frame[[c for c in self.data_frame.columns if c != 'label'] + ['label']]
        self.save(name, n_pca)

    def extract(self, name):
        if self.pca_features:
            for n in self.n_pca:
                self._extract(name, n)
                self.apply_pca(n)
                self.finish(name, n)
        else:
            self._extract(name, 'no-pca')
            self.finish(name, 'no-pca')
