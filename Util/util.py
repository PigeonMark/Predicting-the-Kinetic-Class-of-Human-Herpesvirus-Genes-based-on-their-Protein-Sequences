import re
import os


def open_xml_paper(filename):
    """
    Open xml file and strip xml tags of the form <tag>...</tag>
    :param filename:    The xml file to open
    :return:            The string of the stripped xml file
    """
    with open(filename, "r") as f:
        paper = f.read()
        matches = re.search(r"<back>.*</back>", paper, re.MULTILINE | re.DOTALL)
        if matches is not None:
            paper = paper[:matches.start()] + paper[matches.end():]

        no_tags = re.sub('<[^<]+>', " ", paper)
        return no_tags


def filename_from_path(file_list_dict):
    new = {}
    for virus, file_list in file_list_dict.items():
        new_list = []
        for file in file_list:
            new_list.append(os.path.basename(file))
        new[virus] = new_list
    return new


def add_to_debug_dict(to_add, keyword, phase, debug_info_dict):
    if keyword in debug_info_dict:
        if phase in debug_info_dict[keyword]:
            debug_info_dict[keyword][phase].append(to_add)
        else:
            debug_info_dict[keyword][phase] = [to_add]
    else:
        debug_info_dict[keyword] = {phase: [to_add]}


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


def get_separate_keywords(gene):
    return gene.split('_')


def get_uniprot_id(gene, protein_collector, keywords):
    all_keys, name_to_headers, header_row = keywords
    max_evidence = 6
    max_uniprot_id = ''
    for kw in get_separate_keywords(gene):
        if kw in header_row:
            sequence, evidence_level = protein_collector.read_protein_sequence(kw)
            if evidence_level < max_evidence:
                max_evidence = evidence_level
                max_uniprot_id = kw
    return max_uniprot_id


def data_from_protein(data, protein):
    for i, row in data.iterrows():
        if row['protein'] == protein:
            return row


def print_data_row(row):
    to_print = ""
    to_print += row['virus']
    to_print += (6 - len(to_print)) * ' '
    to_print += '  ' + row['protein_group']
    to_print += (53 - len(to_print)) * ' '
    to_print += '  ' + row['protein']
    to_print += (59 - len(to_print)) * ' '
    to_print += '  ' + row['label']
    to_print += (78 - len(to_print)) * ' '
    if len(row['sequence']) > 100:
        to_print += '  ' + row['sequence'][:50] + '...'
    else:
        to_print += '  ' + row['sequence']
    to_print += (133 - len(to_print)) * ' '
    to_print += '  ' + f"http://localhost:5000/index/{row['virus']}/{row['protein_group']}"

    print(to_print)
