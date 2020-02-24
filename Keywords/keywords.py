import ast
import csv
import json
import os
import pickle


def build_keywords(keywords_file, output_file):
    """
    Reads a csv file containing the keywords of a certain tax-id.

    :param keywords_file:   The filepath of the csv file containing the keywords
    :param output_file:     The filepath of output file
    :return:    A list of all keywords
                A dictionary mapping each keyword to its 'header'
                A list of the header row
    """

    # Open and read the csv file, make it a list of rows
    keywords_csv = open(keywords_file, "r", newline='')
    csv_reader = csv.reader(keywords_csv)
    csv_list = list(csv_reader)

    # Extract each row and throw away the first column
    header_row = csv_list[0][1:]
    genes_row = csv_list[1][1:]
    names_row = csv_list[3][1:]
    proteins_row = csv_list[5][1:]
    uniprot_ac_row = csv_list[6][1:]

    # Make lowercase of each keyword in the first row
    for i, head in enumerate(header_row):
        header_row[i] = head.lower()

    # For each other row, unpack the list items and make them lowercase
    for i, gene in enumerate(genes_row):
        genes_row[i] = ast.literal_eval(gene)
        names_row[i] = ast.literal_eval(names_row[i])
        proteins_row[i] = ast.literal_eval(proteins_row[i])
        uniprot_ac_row[i] = ast.literal_eval(uniprot_ac_row[i])
        for j, elem in enumerate(genes_row[i]):
            genes_row[i][j] = elem.lower()
        for j, elem in enumerate(names_row[i]):
            names_row[i][j] = elem.lower()
        for j, elem in enumerate(proteins_row[i]):
            proteins_row[i][j] = elem.lower()
        for j, elem in enumerate(uniprot_ac_row[i]):
            uniprot_ac_row[i][j] = elem.lower()

    # Make a dictionary {'header' : ['kw1', 'kw2', ...]}
    # and remove all keywords that are just digits or keywords with length <= 2
    header_to_all_names = {}
    for i, header in enumerate(header_row):
        new_genes = []
        for gene in genes_row[i]:
            if not gene.isdigit() and not len(gene) <= 2:
                new_genes.append(gene)
        header_to_all_names[header] = set(
            [header] + uniprot_ac_row[i] + new_genes + names_row[i])  # + proteins_row[i]

    # Make a dictionary {'keyword': ['header1', 'header2', ...]
    # and a simple list of ALL KEYWORDS
    name_to_headers = {}
    all_keys = set()
    for key, value in header_to_all_names.items():
        for val in value:
            if val in name_to_headers:
                name_to_headers[val] += [key]
            else:
                name_to_headers[val] = [key]
            all_keys.add(val)

    keywords_csv.close()

    pickle.dump((all_keys, name_to_headers, header_row), open(output_file, 'wb'))

    return all_keys, name_to_headers, header_row


class KeywordBuilder:
    def __init__(self, config_filepath):
        self.keyword_files = None
        self.output_directory = None
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.keyword_files = config["keyword_files"]
            self.output_directory = config["output_directory"]

    def get_keywords(self, virus_name):
        keywords_filename = os.path.basename(self.keyword_files[virus_name])
        pickle_file = os.path.join(self.output_directory, keywords_filename) + '.p'
        if os.path.isfile(pickle_file):
            return pickle.load(open(pickle_file, 'rb'))

        else:
            return build_keywords(self.keyword_files[virus_name], pickle_file)
