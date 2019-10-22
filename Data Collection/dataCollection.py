import csv
import ast
import pickle
from paperSelection import PUNCTUATION, open_xml_paper
from helper import sort_by_highest_total, print_sorted_occ_dict


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


def add_to_near_occ_dict(keyword, phase, near_occ_dic):
    """
    A helper function that adds a keyword-phase combination to the near-occurrences dictionary in the right way
    :param keyword: The keyword (protein/gene/...)
    :param phase:   The phase
    :param near_occ_dic:    The near-occurrences dictionary
    :return:        Nothing, it directly changes the dictionary
    """

    if keyword in near_occ_dic:
        if phase in near_occ_dic[keyword]:
            near_occ_dic[keyword][phase] += 1
        else:
            near_occ_dic[keyword][phase] = 1
    else:
        near_occ_dic[keyword] = {phase: 1}


def count_near_occurrences(papers_list_file, keywords_file, distance):
    """
    A function that iterates over a list of papers and counts the distances between the keywords and the phases
    :param papers_list_file:    The filename of the pickle file with the list of papers
    :param keywords_file:       The filename of the csv file with the papers
    :param distance:            The distance between the keywords and the phases to take into account
    :return:                    A dictionary containing the counts of the near occurrences of all keywords and phases
                                for each paper
    """

    all_keys, name_to_headers = build_keywords(keywords_file)
    papers_list = pickle.load(open("Data/selectedPapers/" + papers_list_file, "rb"))
    all_phases = ['immediate-early', 'ie', 'early', 'early-late', 'late-early', 'late']

    index = {}

    # For each in file in the papers_list
    for filename in papers_list:

        # Open file and make a lowercase list without punctuation and whitespace
        file = open_xml_paper(filename)
        content = file.lower().translate(str.maketrans('\n\t', '  ', PUNCTUATION)).split()

        near_occ_dict = {}

        # For each word of the paper
        for i, word in enumerate(content):
            # If the word is a keyword
            if word in all_keys:
                # Calculate the sublist that represents the 'near-occurrences zone'
                start_i = i - distance if i - distance >= 0 else 0
                end_i = i + distance + 1
                sublist = content[start_i:end_i]
                # For each phase
                for phase in all_phases:
                    # For each word of the sublist
                    for j, w in enumerate(sublist):
                        # If this word of the sublist is the phase
                        if w == phase:

                            # Special case for immediate early
                            if phase == 'early' and sublist[j - 1] == "immediate":
                                add_to_near_occ_dict(word, "immediate early", near_occ_dict)

                            # Normal case for all other phases
                            else:
                                add_to_near_occ_dict(word, phase, near_occ_dict)

        if not len(near_occ_dict) == 0:
            index[filename] = near_occ_dict

    sorted_i = sort_by_highest_total(index)
    pickle.dump((index, sorted_i), open("Data/countingResults/%s_%s.p" % (papers_list_file, keywords_file), "wb"))

    return index, sorted_i


if __name__ == "__main__":
    hsv1_keywords_file = "keywords_10298.csv"
    papers_list_f = "hsv-1_comm_use.I-N_20191021-173402.p"

    near_occ_index, sorted_index = count_near_occurrences(papers_list_f, hsv1_keywords_file, 10)

    print_sorted_occ_dict(sorted_index, near_occ_index)
