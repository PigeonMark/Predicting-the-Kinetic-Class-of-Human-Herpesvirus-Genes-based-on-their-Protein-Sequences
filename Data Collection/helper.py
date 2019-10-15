def print_index(index):
    for filename, index_tuple in index.items():
        print(filename)
        kw_dict, phase_dict = index_tuple
        print("\tKeywords:")
        for kw, i_list in kw_dict.items():
            print(f"\t\t{kw}: {i_list}")
        print("\n\tPhases:")
        for phase, i_list in phase_dict.items():
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
