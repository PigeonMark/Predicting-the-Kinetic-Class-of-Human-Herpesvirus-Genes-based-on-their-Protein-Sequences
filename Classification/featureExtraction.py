"""
Adapted from https://github.com/bittremieux/TCR-Classifier/blob/master/tcr_classifier_v2.ipynb
"""

import numpy as np
import pandas as pd
from pyteomics import electrochem, mass, parser

from Classification import proteinQuerying
from DataCollection.dataCollection import combine_counts_all_papers, combine_counts_alternate_names
from helper import sort_by_highest_value, normalize_combined_counts_tuple_list, cut_score_data
from DataCollection.input_data import get_viruses_data

# physico-chemical amino acid properties
basicity = {'A': 206.4, 'B': 210.7, 'C': 206.2, 'D': 208.6, 'E': 215.6, 'F': 212.1, 'G': 202.7,
            'H': 223.7, 'I': 210.8, 'K': 221.8, 'L': 209.6, 'M': 213.3, 'N': 212.8, 'P': 214.4,
            'Q': 214.2, 'R': 237.0, 'S': 207.6, 'T': 211.7, 'V': 208.7, 'W': 216.1, 'X': 210.2,
            'Y': 213.1, 'Z': 214.9}

hydrophobicity = {'A': 0.16, 'B': -3.14, 'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31,
                  'H': -4.63, 'I': 4.41, 'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92,
                  'Q': -2.76, 'R': -2.77, 'S': -2.85, 'T': -1.08, 'V': 3.02, 'W': 4.88, 'X': 4.59,
                  'Y': 2.00, 'Z': -2.13}

helicity = {'A': 1.24, 'B': 0.92, 'C': 0.79, 'D': 0.89, 'E': 0.85, 'F': 1.26, 'G': 1.15, 'H': 0.97,
            'I': 1.29, 'K': 0.88, 'L': 1.28, 'M': 1.22, 'N': 0.94, 'P': 0.57, 'Q': 0.96, 'R': 0.95,
            'S': 1.00, 'T': 1.09, 'V': 1.27, 'W': 1.07, 'X': 1.29, 'Y': 1.11, 'Z': 0.91}

mutation_stability = {'A': 13, 'C': 52, 'D': 11, 'E': 12, 'F': 32, 'G': 27, 'H': 15, 'I': 10,
                      'K': 24, 'L': 34, 'M': 6, 'N': 6, 'P': 20, 'Q': 10, 'R': 17, 'S': 10,
                      'T': 11, 'V': 17, 'W': 55, 'Y': 31}

physchem_properties = {'basicity': basicity, 'hydrophobicity': hydrophobicity,
                       'helicity': helicity, 'mutation stability': mutation_stability}


def compute_features(data):
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

    features_list = [data['protein_group'], data['protein']]

    # non-positional features (i.e. over the whole sequence)

    # sequence length
    features_list.append(data['sequence'].apply(lambda sequence: parser.length(sequence)).to_frame()
                         .rename(columns={'sequence': 'length'}))

    # number of occurences of each amino acid
    aa_counts = pd.DataFrame.from_records(
        [parser.amino_acid_composition(sequence) for sequence in data['sequence']]).fillna(0)
    aa_counts.columns = ['{} count'.format(column) for column in aa_counts.columns]
    features_list.append(aa_counts)

    # average physico-chemical properties
    for prop_name, prop_lookup in physchem_properties.items():
        features_list.append(data['sequence'].apply(
            lambda sequence: np.mean(list(prop_lookup[aa] for aa in sequence)))
                             .to_frame().rename(columns={'sequence': 'average {}'.format(prop_name)}))

    # peptide mass
    features_list.append(data['sequence'].apply(
        lambda sequence: mass.fast_mass(sequence)).to_frame().rename(columns={'sequence': 'mass'}))

    # pI
    features_list.append(data['sequence'].apply(
        lambda sequence: electrochem.pI(sequence)).to_frame().rename(columns={'sequence': 'pI'}))

    # # positional features (i.e. localized at a specific amino acid position)
    # pos_aa, pos_basicity, pos_hydro, pos_helicity, pos_mutation, pos_pI = [[] for _ in range(6)]
    # for sequence in data['sequence']:
    #     length = parser.length(sequence)
    #     start_pos = -1 * (length // 2)
    #     pos_range = list(range(start_pos, start_pos + length)) if length % 2 == 1 else \
    #         list(range(start_pos, 0)) + list(range(1, start_pos + length + 1))
    #
    #     pos_aa.append({'pos_{}_{}'.format(pos, aa): 1 for pos, aa in zip(pos_range, sequence)})
    #     pos_basicity.append({'pos_{}_basicity'.format(pos): basicity[aa]
    #                          for pos, aa in zip(pos_range, sequence)})
    #     pos_hydro.append({'pos_{}_hydrophobicity'.format(pos): hydrophobicity[aa]
    #                       for pos, aa in zip(pos_range, sequence)})
    #     pos_helicity.append({'pos_{}_helicity'.format(pos): helicity[aa]
    #                          for pos, aa in zip(pos_range, sequence)})
    #     pos_mutation.append({'pos_{}_mutation_stability'.format(pos): mutation_stability[aa]
    #                          for pos, aa in zip(pos_range, sequence)})
    #
    #     pos_pI.append({'pos_{}_pI'.format(pos): electrochem.pI(aa)
    #                    for pos, aa in zip(pos_range, sequence)})
    #
    # features_list.append(pd.DataFrame.from_records(pos_aa).fillna(0))
    # features_list.append(pd.DataFrame.from_records(pos_basicity).fillna(0))
    # features_list.append(pd.DataFrame.from_records(pos_hydro).fillna(0))
    # features_list.append(pd.DataFrame.from_records(pos_helicity).fillna(0))
    # features_list.append(pd.DataFrame.from_records(pos_mutation).fillna(0))
    # features_list.append(pd.DataFrame.from_records(pos_pI).fillna(0))

    features_list.append(data['label'])

    return pd.concat(features_list, axis=1)


def save_features(features: pd.DataFrame, output_file):
    features.to_csv(f'Classification/Output/features/{output_file}.csv')


def main():
    viruses_data = get_viruses_data()

    for virus in viruses_data:
        combined_counts, paper_counts = combine_counts_all_papers(virus["counted_file"])
        combined_counts_an, paper_counts_an = combine_counts_alternate_names(combined_counts,
                                                                                            paper_counts,
                                                                                            virus["keywords_file"])
        sorted_combined_counts_an = sort_by_highest_value(combined_counts_an)
        normalized_combined_counts_an = normalize_combined_counts_tuple_list(sorted_combined_counts_an)

        cutted_index = cut_score_data(sorted_combined_counts_an, normalized_combined_counts_an)

        df, sequence_dict = proteinQuerying.read_protein_sequences_batch(cutted_index, virus["keywords_file"])

        features = compute_features(df)
        save_features(features, f'{virus["name"]}_features')


if __name__ == "__main__":
    main()
