from shutil import copy
import os
import pickle
import csv


def print_index(index, length_only=False):
    for filename, index_tuple in index.items():
        print(filename)
        kw_dict, phase_dict = index_tuple
        print("\tKeywords:")
        for kw, i_list in kw_dict.items():
            if length_only:
                print(f"\t\t{kw}: {len(i_list)}")
            else:
                print(f"\t\t{kw}: {i_list}")
        print("\n\tPhases:")
        for phase, i_list in phase_dict.items():
            if length_only:
                print(f"\t\t{phase}: {len(i_list)}")
            else:
                print(f"\t\t{phase}: {i_list}")
        print()


def print_score_dict(score_dict):
    for filename, f_score_dict in score_dict.items():
        print(filename)
        for gene, scores in f_score_dict.items():
            print(f"\t{gene}:")
            for phase, score in scores.items():
                print(f"\t\t{phase}: {score}")
            print()
        print()


def print_near_occ_dict(dictionary):
    for k, v in dictionary.items():
        print(f'{k}')
        for k2, v2 in v.items():
            print(f'\t{k2}: ')
            for k3, v3 in v2.items():
                print(f'\t\t{k3}: {v3}')


def print_sorted_occ_dict(sorted_dictionary, index):
    for fn, tot in sorted_dictionary:
        print(f'{fn}')
        for k2, v2 in index[fn].items():
            print(f'\t{k2}: ')
            for k3, v3 in v2.items():
                print(f'\t\t{k3}: {v3}')


def sort_by_highest_total(dictionary):
    tot_dict = {}
    for fn, keywords in dictionary.items():
        tot = 0
        for keyword, phases in keywords.items():
            for phase, count in phases.items():
                tot += count
        tot_dict[fn] = tot
    sorted_dict = sorted(tot_dict.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_dict


def sort_by_highest_value(dictionary):
    return sorted(dictionary.items(), key=lambda keyword: max(keyword[1].get('late', 0), keyword[1].get('early', 0),
                                                              keyword[1].get('immediate-early', 0),
                                                              keyword[1].get('early-late', 0)), reverse=True)


def print_combined_counts(combined_index):
    for kw, phases in combined_index.items():
        print(f'{kw}:')
        for phase, count in phases.items():
            print(f'\t{phase}: {count}')
        print()


def print_combined_counts_tuple_list(combined_index):
    for kw, phases in combined_index:
        print(f'{kw}:')
        for phase, count in phases.items():
            print(f'\t{phase}: {count}')
        print()


def normalize_combined_counts_tuple_list(combined_index):
    normalized_list = []
    for kw, phases in combined_index:
        total = sum([phases.get("late", 0), phases.get("early", 0), phases.get("immediate-early", 0),
                     phases.get("early-late", 0)])
        normalized_list.append((kw, {"immediate-early": phases.get("immediate-early", 0) / float(total),
                                     "early": phases.get("early", 0) / float(total),
                                     "early-late": phases.get("early-late", 0) / float(total),
                                     "late": phases.get("late", 0) / float(total)}))
    return normalized_list


def copy_selected_papers(paper_list_path, directory_name):
    os.mkdir("Output/selectedPapers/" + directory_name)
    paper_list = pickle.load(open(paper_list_path, "rb"))
    for paper in paper_list:
        copy(paper, "Output/selectedPapers/" + directory_name)


def print_combined_counts_to_csv(combined_index, ncc, paper_counts, file_n):
    with open(file_n[:-2] + '.csv', "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            ['Keyword', 'Immediate-early', '# papers', 'Early', '# papers', 'Early-late', '# papers', 'Late',
             '# papers', '', 'Immediate-early (%)', 'Early (%)', 'Early-late (%)', 'Late (%)'])
        for i, (kw, p) in enumerate(combined_index):
            csv_writer.writerow(
                [kw, f"{p.get('immediate-early', 0):.2f}", f"{paper_counts[kw].get('immediate-early', 0)}",
                 f"{p.get('early', 0):.2f}", f"{paper_counts[kw].get('early', 0)}",
                 f"{p.get('early-late', 0):.2f}", f"{paper_counts[kw].get('early-late', 0)}",
                 f"{p.get('late', 0):.2f}", f"{paper_counts[kw].get('late', 0)}", "",
                 f"{ncc[i][1].get('immediate-early', 0) * 100:.2f}", f"{ncc[i][1].get('early', 0) * 100:.2f}",
                 f"{ncc[i][1].get('early-late', 0) * 100:.2f}", f"{ncc[i][1].get('late', 0) * 100:.2f}"])


if __name__ == "__main__":
    file_names = ["ebv_all_20191025-173954", "hcmv_all_20191025-173918", "hsv1_all_20191025-174132",
                  "hsv2_all_20191025-174101", "hsv-1_comm_use.I-N_20191021-173402", "vzv_all_20191025-174034"]

    for file_name in file_names:
        copy_selected_papers("Output/selectedPapers/" + file_name + ".p", file_name)
