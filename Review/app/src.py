from collections import namedtuple

from DebugInfoCollector import DebugInfoCollector
from Review.app.models import Gene

CurrentGene = namedtuple('Current_Gene', 'virus gene debug_info')


def paper_url(paper_name):
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper_name[:-5]}"


def sorted_keys_phases(dict, score_dict):
    return sorted(dict.keys(), key=lambda phase: score_dict[phase], reverse=True)


class GeneRotator:
    debug_info_collector = DebugInfoCollector('config/debug_info_collector_config.json')
    debug_info = debug_info_collector.load_debug_info()
    paper_titles = debug_info_collector.load_paper_titles()

    current = None

    @staticmethod
    def get():
        return GeneRotator.current, GeneRotator.paper_titles

    @staticmethod
    def next():
        for virus, genes in GeneRotator.debug_info.items():
            for gene, gene_info in genes.items():
                if not Gene.exists(gene):
                    GeneRotator.current = CurrentGene(virus, gene, gene_info)
                    return

        GeneRotator.current = CurrentGene(None, None, None)
