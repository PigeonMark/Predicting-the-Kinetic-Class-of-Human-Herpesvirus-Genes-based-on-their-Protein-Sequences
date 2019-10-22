import csv
import ast
import pickle
from datetime import datetime
from paperSelection import PUNCTUATION, open_xml_paper
from helper import print_index, print_score_dict, print_near_occ_dict, sort_by_highest_total, print_sorted_occ_dict


def build_keywords(keywords_file):
    keywords_csv = open(keywords_file, "r", newline='')
    csv_reader = csv.reader(keywords_csv)

    csv_list = list(csv_reader)

    header_row = csv_list[0][1:]
    genes_row = csv_list[1][1:]
    names_row = csv_list[3][1:]
    proteins_row = csv_list[5][1:]
    uniprot_ac_row = csv_list[6][1:]

    for i, head in enumerate(header_row):
        header_row[i] = head.lower()

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

    header_to_all_names = {}
    for i, header in enumerate(header_row):
        new_genes = []
        for gene in genes_row[i]:
            if not gene.isdigit() and not len(gene) <= 2:
                new_genes.append(gene)
        header_to_all_names[header] = set([header] + uniprot_ac_row[i] + new_genes + names_row[i])  # + proteins_row[i]

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


def build_index(papers_list, keywords_file):
    all_keys, name_to_headers = build_keywords(keywords_file)
    all_phases = ['immediate-early', 'ie', 'early', 'early-late', 'late-early', 'late']

    index = {}

    file_cnt = 0
    for filename in papers_list:
        file = open_xml_paper(filename)
        content = file.lower().translate(str.maketrans('\n\t', '  ', PUNCTUATION)).split()

        kw_dict = {}
        phase_dict = {}
        for i, word in enumerate(content):
            if word in all_keys:
                if word in kw_dict:
                    kw_dict[word] += [i]
                else:
                    kw_dict[word] = [i]
            if word in all_phases:
                if word in phase_dict:
                    phase_dict[word] += [i]
                else:
                    phase_dict[word] = [i]

        if len(kw_dict) > 0 and len(phase_dict) > 0:
            index[filename] = (kw_dict, phase_dict)

        file_cnt += 1
        if file_cnt % 1000 == 0:
            print("%s files done" % file_cnt)

    print_index(index)
    pickle.dump(index, open("index_comm_use.I-N_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"), "wb"))
    return index


def add_to_near_occ_dict(prot, phase, dict):
    if prot in dict:
        if phase in dict[prot]:
            dict[prot][phase] += 1
        else:
            dict[prot][phase] = 1
    else:
        dict[prot] = {phase: 1}


def count_near_occurrences(papers_list, keywords_file, distance):
    all_keys, name_to_headers = build_keywords(keywords_file)
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
                            if phase == 'early' and sublist[j-1] == "immediate":
                                add_to_near_occ_dict(word, "immediate early", near_occ_dict)

                            # Normal case for all other phases
                            else:
                                add_to_near_occ_dict(word, phase, near_occ_dict)

        if not len(near_occ_dict) == 0:
            index[filename] = near_occ_dict

    return index


def calculate_distances_from_index(index):
    full_score_dict = {}
    for filename, dicts in index.items():
        kw_dict, phase_dict = dicts
        score_dict = {}
        for kw, kw_indices in kw_dict.items():
            for phase, phase_indices in phase_dict.items():
                kw_phase_score = 0
                for kw_indice in kw_indices:
                    for phase_indice in phase_indices:
                        if phase_indice != kw_indice:
                            kw_phase_score += 1 / float(abs(phase_indice - kw_indice))
                kw_phase_score /= (len(kw_indices) * len(phase_indices))
                if kw in score_dict:
                    score_dict[kw][phase] = (kw_phase_score, len(kw_indices) * len(phase_indices))
                else:
                    score_dict[kw] = {phase: (kw_phase_score, len(kw_indices) * len(phase_indices))}
        full_score_dict[filename] = score_dict
    return full_score_dict


def test_on_known_papers():
    papers_list = ["Data/known_herpes_files/paper1.txt", "Data/known_herpes_files/paper2.txt",
                   "Data/known_herpes_files/paper2.txt"]
    index = build_index(papers_list)
    score_dict = calculate_distances_from_index(index)
    print_score_dict(score_dict)


if __name__ == "__main__":
    hsv1_kewords_file = "Data/keywords_10298.csv"

    # all_keys, name_to_headers = build_keywords(hsv1_kewords_file)
    # print(len(all_keys))
    #
    # index = build_index(pickle.load(open("hsv-1_comm_use.I-N_20191021-173402.p", "rb")), hsv1_kewords_file)

    # print_index(pickle.load(open("index_comm_use.I-N_20191021-182253.p", "rb")), length_only=True)

    papers_list = ["Data/comm_use.I-N/J_Biol_Chem/PMC5602406.nxml"]
    # index = count_near_occurrences(pickle.load(open("hsv-1_comm_use.I-N_20191021-173402.p", "rb")), hsv1_kewords_file, 20)

    index = count_near_occurrences(papers_list, hsv1_kewords_file, 5)
    sorted = sort_by_highest_total(index)
    print_sorted_occ_dict(sorted, index)



    # print(test_near_occurrences())

    # score_dict = calculate_distances_from_index(pickle.load(open("index_20191016-120410.p", "rb")))
    # print_score_dict(score_dict)

    # test_on_known_papers()
