import ast
import networkx as nx
import nltk

from helper import *
from DataCollection.input_data import get_viruses_data, get_phases_data, PUNCTUATION
from DataCollection.paperSelection import open_xml_paper
# from Classification import proteinQuerying
from Util.util import filename_from_path


def build_keywords(keywords_file, from_classification=False):
    """
    Reads a csv file containing the keywords of a certain tax-id.

    :param keywords_file: The filename of the csv file containing the keywords
    :return:    A list of all keywords
                A dictionary mapping each keyword to its 'header' TODO: map to a better keyword
    """

    # Open and read the csv file, make it a list of rows
    if from_classification:
        keywords_csv = open("../DataCollection/Data/keywords/" + keywords_file, "r", newline='')
    else:
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
    return all_keys, name_to_headers, header_row


def add_to_near_occ_dict(to_add, keyword, phase, near_occ_dic):
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


def add_to_debug_dict(to_add, keyword, phase, debug_info_dict):
    if keyword in debug_info_dict:
        if phase in debug_info_dict[keyword]:
            debug_info_dict[keyword][phase].append(to_add)
        else:
            debug_info_dict[keyword][phase] = [to_add]
    else:
        debug_info_dict[keyword] = {phase: [to_add]}


def count_near_occ_by_distance(word, kw_i, distance, content, near_occ_dict, debug_info_dict):
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
                        add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)
                        add_to_debug_dict(content[debug_range_min: debug_range_max], word, "immediate-early",
                                          debug_info_dict)

                    # Map 'ie' to 'immediate-early'
                    elif phase == 'ie':
                        add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)
                        add_to_debug_dict(content[debug_range_min: debug_range_max], word, "immediate-early",
                                          debug_info_dict)

                    # Normal case for all other phases
                    else:
                        add_to_near_occ_dict(to_add, word, phase, near_occ_dict)
                        add_to_debug_dict(content[debug_range_min: debug_range_max], word, phase,
                                          debug_info_dict)


def count_near_occurrences(papers_directory, file_list, keywords_file, distance, virus_name):
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

    index = {}
    debug_index = {}
    file_count = 0
    total_file_count = len(file_list)
    print(f'Counting near occurrences in {total_file_count} files')
    # For each in file in the papers_list
    for filename in file_list:

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
                count_near_occ_by_distance(word, kw_i, distance, content, near_occ_dict, debug_info_dict)

        if not len(near_occ_dict) == 0:
            index[filename] = near_occ_dict
            debug_index[filename] = debug_info_dict

        file_count += 1
        if file_count % 1000 == 0:
            print(f'{file_count} files done ({100 * file_count / float(total_file_count):.2f}%)')

    sorted_i = sort_by_highest_total(index)
    # papers_directory_name = os.path.basename(os.path.normpath(papers_directory))
    pickle.dump((index, sorted_i),
                open("Output/countingResults/%s_%i.p" % (virus_name, distance), "wb"))
    pickle.dump(debug_index,
                open("Output/debug_info/%s_%i.p" % (virus_name, distance), "wb"))

    return index, sorted_i, "Output/countingResults/%s_%i.p" % (virus_name, distance)


def combine_counts_all_papers(index_file):
    """
    Combine the counts of all the papers
    :param index_file:  The file containing the counting results per paper
    :return:            A dictionary containing the combined counts:
                        {'kw1': {'phase1': score1, 'phase2': score2, ...}, 'kw2': ...}
    """
    index, sorted_index = pickle.load(open(index_file, "rb"))

    combined_counts = {}
    paper_counts = {}
    for paper, proteins in index.items():
        for protein, phases in proteins.items():
            for phase, count in phases.items():
                add_to_near_occ_dict(count, protein, phase, combined_counts)
                add_to_near_occ_dict(1, protein, phase, paper_counts)
    return combined_counts, paper_counts


def combine_counts_alternate_names(index, paper_counts, keywords_file, from_classification=False):
    all_keys, name_to_headers, header_row = build_keywords(keywords_file, from_classification)
    G = nx.DiGraph()

    for kw in index.keys():
        for hdr in name_to_headers[kw]:
            if kw != hdr:
                G.add_edge(kw, hdr)

    connectedComponents = list(nx.connected_component_subgraphs(nx.Graph(G)))
    cc_dict = {}
    for i, connectedComponent in enumerate(connectedComponents):
        component_name = ""
        for node in sorted(connectedComponent.nodes()):
            if len(component_name) == 0:
                component_name += node
            else:
                component_name += "_" + node
        cc_dict[component_name] = list(connectedComponent.nodes())

    combined_counts = {}
    combined_paper_counts = {}
    for group_name, kws in cc_dict.items():
        for kw in kws:
            if kw in index:
                for phase, count in index[kw].items():
                    add_to_near_occ_dict(count, group_name, phase, combined_counts)
                    add_to_near_occ_dict(paper_counts[kw][phase], group_name, phase, combined_paper_counts)

    return combined_counts, combined_paper_counts


def main():
    viruses_data = get_viruses_data()

    papers_list = filename_from_path(pickle.load(open("Output/selected.p", "rb")))
    virus_name_converter = {'HSV 1': 'HSV_1', 'HSV 2': 'HSV_2', 'Varicella zoster virus': 'VZV',
                            'Epstein-Barr virus': 'EBV', 'Human cytomegalovirus': 'HCMV'}

    # for virus in viruses_data:
    #     near_occ_index, sorted_index, i_file = __count_near_occurrences("Output/selected_papers",
    #                                                                   papers_list[virus_name_converter[virus['name']]],
    #                                                                   virus["keywords_file"], 10,
    #                                                                   virus_name_converter[virus['name']])

    for virus in viruses_data:
        counted_file = f"Output/countingResults/{virus_name_converter[virus['name']]}_10.p"
        combined_counts, paper_counts = combine_counts_all_papers(counted_file)
        combined_counts_an, paper_counts_an = combine_counts_alternate_names(combined_counts, paper_counts,
                                                                             virus["keywords_file"])
        print(virus["name"])
        print(len(combined_counts))
        print(len(combined_counts_an))

        sorted_combined_counts_an = sort_by_highest_value(combined_counts_an)
        # print_combined_counts_tuple_list(sorted_combined_counts)
        print()
        normalized_combined_counts_an = normalize_combined_counts_tuple_list(sorted_combined_counts_an)
        print_combined_counts_to_csv(sorted_combined_counts_an, normalized_combined_counts_an, paper_counts_an,
                                     counted_file)

        # proteinQuerying.get_protein_sequences_batch(combined_counts_an, virus["keywords_file"])


if __name__ == "__main__":
    main()
