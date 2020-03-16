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
