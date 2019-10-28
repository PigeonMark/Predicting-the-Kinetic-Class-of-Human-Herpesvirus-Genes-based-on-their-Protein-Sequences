import csv
import ast
import pickle
from paperSelection import PUNCTUATION, open_xml_paper
from helper import *

ALL_PHASES = ['immediate-early', 'ie', 'early', 'early-late', 'late-early', 'late']


def build_keywords(keywords_file):
    """
    Reads a csv file containing the keywords of a certain tax-id.

    :param keywords_file: The filename of the csv file containing the keywords
    :return:    A list of all keywords
                A dictionary mapping each keyword to its 'header' TODO: map to a better keyword
    """

    # Open and read the csv file, make it a list of rows
    keywords_csv = open("Data/keywords/" + keywords_file, "r", newline='')
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
        header_to_all_names[header] = set([header] + uniprot_ac_row[i] + new_genes + names_row[i])  # + proteins_row[i]

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
    return all_keys, name_to_headers


def add_to_near_occ_dict(to_add, keyword, phase, near_occ_dic):
    """
    A helper function that adds a keyword-phase combination to the near-occurrences dictionary in the right way
    :param keyword:         The keyword (protein/gene/...)
    :param phase:           The phase
    :param near_occ_dic:    The near-occurrences dictionary
    :return:                Nothing, it directly changes the dictionary
    """

    if keyword in near_occ_dic:
        if phase in near_occ_dic[keyword]:
            near_occ_dic[keyword][phase] += to_add
        else:
            near_occ_dic[keyword][phase] = to_add
    else:
        near_occ_dic[keyword] = {phase: to_add}


def count_near_occ_by_distance(word, kw_i, distance, content, near_occ_dict):
    # For each phase
    for phase in ALL_PHASES:

        for dis in range(1, distance + 1):
            i1 = kw_i - dis
            i2 = kw_i + dis

            to_add = 1 / float(dis)

            if 0 <= i1 < len(content) and content[i1] == phase:
                # Special case for immediate early
                if phase == 'early' and content[i1 - 1] == "immediate":
                    add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)

                elif phase == 'ie':
                    add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)

                # Normal case for all other phases
                else:
                    add_to_near_occ_dict(to_add, word, phase, near_occ_dict)

            if 0 <= i2 < len(content) and content[i2] == phase:
                # Special case for immediate early
                if phase == 'early' and content[i2 - 1] == "immediate":
                    add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)

                elif phase == 'ie':
                    add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)

                # Normal case for all other phases
                else:
                    add_to_near_occ_dict(to_add, word, phase, near_occ_dict)


def count_near_occurrences(papers_directory, keywords_file, distance):
    """
    A function that iterates over a list of papers and counts the distances between the keywords and the phases
    :param lower_d:             The lower bound for the distances to take into account
    :param upper_d:             The upper bound for the distances to take into account
    :param papers_list_file:    The filename of the pickle file with the list of papers
    :param keywords_file:       The filename of the csv file with the papers
    :return:                    A dictionary containing the counts of the near occurrences of all keywords and phases
                                for each paper
    """

    all_keys, name_to_headers = build_keywords(keywords_file)

    index = {}
    file_count = 0
    total_file_count = len([_ for _ in os.listdir(papers_directory)])
    print(f'Counting near occurrences in {total_file_count} files')
    # For each in file in the papers_list
    for filename in os.listdir(papers_directory):

        # Open file and make a lowercase list without punctuation and whitespace
        file = open_xml_paper(os.path.join(papers_directory, filename))
        content = file.lower().translate(str.maketrans('\n\t', '  ', PUNCTUATION)).split()

        near_occ_dict = {}

        # For each word of the paper
        for kw_i, word in enumerate(content):
            # If the word is a keyword
            if word in all_keys:
                count_near_occ_by_distance(word, kw_i, distance, content, near_occ_dict)

        if not len(near_occ_dict) == 0:
            index[filename] = near_occ_dict

        file_count += 1
        if file_count % 1000 == 0:
            print(f'{file_count} files done ({100*file_count/float(total_file_count):.2f}%)')

    sorted_i = sort_by_highest_total(index)
    papers_directory_name = os.path.basename(os.path.normpath(papers_directory))
    pickle.dump((index, sorted_i),
                open("Output/countingResults/%s_%s_%i.p" % (papers_directory_name, keywords_file, distance), "wb"))

    return index, sorted_i, "Output/countingResults/%s_%s_%i.p" % (papers_directory_name, keywords_file, distance)


def combine_counts(index_file):
    index, sorted_index = pickle.load(open(index_file, "rb"))

    combined_counts = {}
    for paper, proteins in index.items():
        for protein, phases in proteins.items():
            for phase, count in phases.items():
                add_to_near_occ_dict(count, protein, phase, combined_counts)
    return combined_counts


if __name__ == "__main__":
    hsv1_data = {
        "keywords_file": "keywords_10298.csv",
        "papers_directory": "Output/selectedPapers/hsv1_all_20191025-174132/",
        "counted_file": "Output/countingResults/hsv1_all_20191025-174132_keywords_10298.csv_10.p"
    }

    hsv2_data = {
        "keywords_file": "keywords_10310.csv",
        "papers_directory": "Output/selectedPapers/hsv2_all_20191025-174101/",
        "counted_file": "Output/countingResults/hsv2_all_20191025-174101_keywords_10310.csv_10.p"
    }

    vzv_data = {
        "keywords_file": "keywords_10335.csv",
        "papers_directory": "Output/selectedPapers/vzv_all_20191025-174034/",
        "counted_file": "Output/countingResults/vzv_all_20191025-174034_keywords_10335.csv_10.p"
    }

    ebv_data = {
        "keywords_file": "keywords_10376.csv",
        "papers_directory": "Output/selectedPapers/ebv_all_20191025-173954/",
        "counted_file": "Output/countingResults/ebv_all_20191025-173954_keywords_10376.csv_10.p"
    }

    hcmv_data = {
        "keywords_file": "keywords_10359.csv",
        "papers_directory": "Output/selectedPapers/hcmv_all_20191025-173918/",
        "counted_file": "Output/countingResults/hcmv_all_20191025-173918_keywords_10359.csv_10.p"
    }

    viruses_data = [hsv1_data, hsv2_data, vzv_data, ebv_data, hcmv_data]

    for virus in viruses_data:

        near_occ_index, sorted_index, i_file = count_near_occurrences(virus["papers_directory"],
                                                                      virus["keywords_file"], 10)

        print(virus["papers_directory"], i_file)

    for virus in viruses_data:

        combined_counts = combine_counts(virus["counted_file"])
        print(virus["papers_directory"])
        print(len(combined_counts))
        sorted_combined_counts = sort_by_highest_value(combined_counts)
        print_combined_counts_tuple_list(sorted_combined_counts)
        print()

    # combined_counts = combine_counts(hsv1_data["counted_file"])
    # sorted_combined_counts = sort_by_highest_value(combined_counts)
    # print_combined_counts_tuple_list(sorted_combined_counts)
    # print_combined_counts_tuple_list(normalize_combined_counts_tuple_list(sorted_combined_counts))
