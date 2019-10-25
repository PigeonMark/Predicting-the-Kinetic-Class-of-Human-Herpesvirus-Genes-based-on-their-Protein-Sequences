

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


def print_near_occ_dict(dict):
    for k, v in dict.items():
        print(f'{k}')
        for k2, v2 in v.items():
            print(f'\t{k2}: ')
            for k3, v3 in v2.items():
                print(f'\t\t{k3}: {v3}')


def print_sorted_occ_dict(sorted, index):
    for fn, tot in sorted:
        print(f'{fn}')
        for k2, v2 in index[fn].items():
            print(f'\t{k2}: ')
            for k3, v3 in v2.items():
                print(f'\t\t{k3}: {v3}')


def sort_by_highest_total(dict):
    tot_dict = {}
    for fn, prots in dict.items():
        tot = 0
        for prot, phases in prots.items():
            for phase, count in phases.items():
                tot += count
        tot_dict[fn] = tot
    sorted_dict = sorted(tot_dict.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_dict


def sort_by_highest_value(dict):
    return sorted(dict.items(), key=lambda keyword: max(keyword[1].get('late', 0), keyword[1].get('early', 0), keyword[1].get('immediate-early', 0)), reverse=True)


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
