"""
Adapted from https://github.com/bittremieux/TCR-Classifier/blob/master/tcr_classifier_v2.ipynb
"""
import json

import numpy as np
import pandas as pd
from pyteomics import electrochem, mass, parser
from Util import ReviewDBReader, get_uniprot_id
from Keywords import KeywordBuilder
from ProteinCollecting import ProteinCollector


class FeatureExtraction:
    def __init__(self, config_filepath):
        self.physchem_properties = {}
        self.dataframe = None
        self.keywords = {}
        self.viruses = None
        self.review_db_reader = None  # type: ReviewDBReader
        self.protein_collector = None  # type: ProteinCollector
        self.features = None
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

            self.review_db_reader = ReviewDBReader(config['review_db_reader_config'])

            for virus in self.viruses:
                self.keywords[virus] = KeywordBuilder(config['keywords_config']).get_keywords(virus)

    def create_df(self):
        self.dataframe = pd.DataFrame(columns=['virus', 'protein_group', 'protein', 'sequence', 'label'])
        i = 0
        for review in self.review_db_reader.get_all():
            if review.review_status in ['CORRECT', 'MODIFIED']:
                uniprot_id = get_uniprot_id(review.names, self.protein_collector, self.keywords[review.virus])
                seq, evidence = self.protein_collector.read_protein_sequence(uniprot_id)
                self.dataframe.loc[i] = [review.virus, review.names, uniprot_id, str(seq), review.reviewed_phase]
                i += 1

    # noinspection PyListCreation
    def compute_features(self):
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

        features_list = [self.dataframe['virus'], self.dataframe['protein_group'], self.dataframe['protein'],
                         self.dataframe['sequence']]

        # non-positional features (i.e. over the whole sequence)

        # sequence length
        features_list.append(
            self.dataframe['sequence'].apply(lambda sequence: parser.length(sequence)).to_frame()
                .rename(columns={'sequence': 'length'}))

        # number of occurences of each amino acid
        aa_counts = pd.DataFrame.from_records(
            [parser.amino_acid_composition(sequence) for sequence in self.dataframe['sequence']]).fillna(0)
        aa_counts.columns = ['{} count'.format(column) for column in aa_counts.columns]
        features_list.append(aa_counts)

        def physchem_properties_function(sequence):
            if prop_name == 'mutation stability':
                return np.mean(list(prop_lookup[aa] for aa in sequence.replace('X', '')))
            else:
                return np.mean(list(prop_lookup[aa] for aa in sequence))

        # average physico-chemical properties
        for prop_name, prop_lookup in self.physchem_properties.items():
            features_list.append(self.dataframe['sequence'].apply(physchem_properties_function)
                                 .to_frame().rename(columns={'sequence': 'average {}'.format(prop_name)}))

        # peptide mass
        features_list.append(self.dataframe['sequence'].apply(
            lambda sequence: mass.fast_mass(sequence.replace('X', ''))).to_frame().rename(columns={'sequence': 'mass'}))

        # pI
        features_list.append(self.dataframe['sequence'].apply(
            lambda sequence: electrochem.pI(sequence)).to_frame().rename(columns={'sequence': 'pI'}))

        features_list.append(self.dataframe['label'])

        self.features = pd.concat(features_list, axis=1)

    def save_features(self):
        self.features.to_csv(
            f"{self.output_csv_directory}features_{self.general_config['distance']}.csv")

    def extract(self):
        self.create_df()
        # print(self.dataframe)
        self.compute_features()
        self.save_features()
