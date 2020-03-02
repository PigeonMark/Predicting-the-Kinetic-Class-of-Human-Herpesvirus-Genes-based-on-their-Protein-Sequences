from Combine import Combiner
from Counting import Counter
from Review.config import Config


def get_separate_keywords(gene):
    return gene.split('_')


def debug_info(virus, gene):
    counter = Counter(Config.COUNTER_CONFIG)
    debug_index = counter.read_debug_index(virus)


    combined_debug_dict = {}
    for paper, debug_kws in debug_index.items():
        genes_found = [g for g in debug_kws if g in get_separate_keywords(gene)]
        if genes_found:
            for g in genes_found:
                for phase, debug_list in debug_kws[g].items():
                    if phase in combined_debug_dict:
                        if paper in combined_debug_dict[phase]:
                            combined_debug_dict[phase][paper] += debug_list
                        else:
                            combined_debug_dict[phase][paper] = debug_list
                    else:
                        combined_debug_dict[phase] = {paper: debug_list}

    # for phase, debug_list in combined_debug_dict.items():
    #     print(phase)
    #     for paper, l in debug_list.items():
    #         print(f"\t{paper}:")
    #         for li in l:
    #             print(f"\t\t{li}")
    return combined_debug_dict


class DebugInfo:
    def __init__(self, virus, gene):
        self.all_names = get_separate_keywords(gene)

        combiner = Combiner(Config.COMBINER_CONFIG)
        combined_counts, sorted_combined_counts, normalized, paper_counts, cutted_index = combiner.read_index(virus)
        self.scores = combined_counts[gene]

        self.percentages = None
        for genes, phases in normalized:
            if genes == gene:
                self.percentages = phases
                break

        self.uniprot_function = None
        self.context = debug_info(virus, gene)
