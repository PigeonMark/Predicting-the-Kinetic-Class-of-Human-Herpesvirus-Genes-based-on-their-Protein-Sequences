import csv
import json
import ast
import os
import nltk
import pickle


def build_keywords(keywords_file):
    """
    Reads a csv file containing the keywords of a certain tax-id.

    :param keywords_file: The filename of the csv file containing the keywords
    :return:    A list of all keywords
                A dictionary mapping each keyword to its 'header'
    """

    # Open and read the csv file, make it a list of rows
    keywords_csv = open("DataCollection/Data/keywords/" + keywords_file, "r", newline='')
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
    return all_keys, name_to_headers, header_row


class DataCollector:
    def __init__(self, config_filepath):
        self.keywords = {}
        self.index = {}
        self.debug_index = {}
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            for virus in config:
                self.keywords[virus["name"]] = build_keywords(virus["keywords_file"])

    def add_to_near_occ_dict(self, to_add, keyword, phase, near_occ_dic):
        """
        A helper function that adds a keyword-phase combination to the near-occurrences dictionary in the right way
        :param to_add:          The number to add
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

    def add_to_debug_dict(self, to_add, keyword, phase, debug_info_dict):
        if keyword in debug_info_dict:
            if phase in debug_info_dict[keyword]:
                debug_info_dict[keyword][phase].append(to_add)
            else:
                debug_info_dict[keyword][phase] = [to_add]
        else:
            debug_info_dict[keyword] = {phase: [to_add]}

    def count_near_occ_by_distance(self, word, kw_i, distance, content, near_occ_dict, debug_info_dict):
        """
        Searches for phases within distance of a found keyword in a paper, a phase at distance x of a keyword will get 1/x
        added to its score for that keyword
        :param word:            The keyword found
        :param kw_i:            The index of the keyword found
        :param distance:        The maximum distance to search for phases
        :param content:         The content text
        :param near_occ_dict:   The dictionary to add the scores to
        :return:                Nothing, function changes the near_occ_dict
        """

        main_phases, alternate_names, all_phases = get_phases_data()

        # For each phase
        for phase in all_phases:
            # For each distance
            for dis in range(1, distance + 1):
                # Get the indices of the text of distance 'dis'
                i1 = kw_i - dis
                i2 = kw_i + dis

                debug_range_min = max(0, i1 - 3)
                debug_range_max = min(len(content) - 1, i2 + 3)

                # Calculate the possible score to add
                to_add = 1 / float(dis)

                # For the indices 'dis' before and 'dis' after the keyword
                for i in [i1, i2]:
                    # Check that the indices are still in the content
                    # Check if the word on distance 'dis' of the keyword is a phase
                    if 0 <= i < len(content) and content[i] == phase:
                        # Special case for immediate early
                        if phase == 'early' and content[i - 1] == "immediate":
                            self.add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)
                            self.add_to_debug_dict(content[debug_range_min: debug_range_max], word, "immediate-early",
                                                   debug_info_dict)

                        # Map 'ie' to 'immediate-early'
                        elif phase == 'ie':
                            self.add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)
                            self.add_to_debug_dict(content[debug_range_min: debug_range_max], word, "immediate-early",
                                                   debug_info_dict)

                        # Normal case for all other phases
                        else:
                            self.add_to_near_occ_dict(to_add, word, phase, near_occ_dict)
                            self.add_to_debug_dict(content[debug_range_min: debug_range_max], word, phase,
                                                   debug_info_dict)

    def count_near_occurrences(self, papers_directory, keywords_file, distance):
        """
        A function that iterates over a list of papers and counts the distances between the keywords and the phases
        :param distance:            The upper bound for the distances to take into account
        :param papers_directory:    The directory containing all the selected papers
        :param keywords_file:       The filename of the csv file containing the keywords
        :return:                    A dictionary containing the counts of the near occurrences of all keywords and phases
                                    for each paper, also a sorted index and the location of the pickle file where the index
                                    is stored
        """

        all_keys, name_to_headers, header_row = build_keywords(keywords_file)

        file_count = 0
        total_file_count = len([_ for _ in os.listdir(papers_directory)])
        print(f'Counting near occurrences in {total_file_count} files')
        # For each in file in the papers_list
        for filename in os.listdir(papers_directory):

            # Open file and make a lowercase list without punctuation and whitespace
            file = open_xml_paper(os.path.join(papers_directory, filename))

            stop_words = set(nltk.corpus.stopwords.words('english'))
            words = file.lower().translate(str.maketrans('\n\t' + PUNCTUATION, ' ' * (len(PUNCTUATION) + 2))).split()
            content = [word for word in words if word not in stop_words and not word.isdigit()]

            near_occ_dict = {}
            debug_info_dict = {}
            # For each word of the paper
            for kw_i, word in enumerate(content):
                # If the word is a keyword
                if word in all_keys:
                    # Check for phases within distance
                    self.count_near_occ_by_distance(word, kw_i, distance, content, near_occ_dict, debug_info_dict)

            if not len(near_occ_dict) == 0:
                self.index[filename] = near_occ_dict
                self.debug_index[filename] = debug_info_dict

            file_count += 1
            if file_count % 1000 == 0:
                print(f'{file_count} files done ({100 * file_count / float(total_file_count):.2f}%)')

        sorted_i = sort_by_highest_total(self.index)
        papers_directory_name = os.path.basename(os.path.normpath(papers_directory))
        pickle.dump((self.index, sorted_i),
                    open("DataCollection/Output/countingResults/%s_%s_%i.p" % (
                        papers_directory_name, keywords_file, distance), "wb"))
        pickle.dump(self.debug_index,
                    open("DataCollection/Output/debug_info/%s_%s_%i.p" % (
                        papers_directory_name, keywords_file, distance), "wb"))

        return


if __name__ == '__main__':
    dataCollector = DataCollector("DataCollection/config/dataCollection_config.json")
