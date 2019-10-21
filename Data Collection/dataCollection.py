import csv
import ast
import pickle
from datetime import datetime
from paperSelection import select_papers_in_topic, PUNCTUATION, open_xml_paper, PAPER_KEYWORDS
from helper import print_index, print_score_dict


def build_keywords():
    keywords_csv = open("Data/keywords_10292.csv", "r", newline='')
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
    # print(header_to_all_names)
    # print(name_to_headers)
    return all_keys, name_to_headers


def build_index(papers_list):
    print(len(papers_list))

    all_keys, name_to_headers = build_keywords()
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
                            kw_phase_score += 1/float(abs(phase_indice-kw_indice))
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

    # select_papers_in_topic("Data/comm_use.I-N/", PAPER_KEYWORDS)

    all_keys, name_to_headers = build_keywords()
    print(name_to_headers)

    # build_index(pickle.load(open("herpespapers_comm_use.I-N_20191016-134537.p", "rb")))

    # print_index(pickle.load(open("index_20191016-120410.p", "rb")))
    # score_dict = calculate_distances_from_index(pickle.load(open("index_20191016-120410.p", "rb")))
    # print_score_dict(score_dict)

    # test_on_known_papers()
