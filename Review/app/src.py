import operator
import json
import re

import requests
from flask import Markup
from Combine import Combiner
from Counting import Counter
from Keywords import KeywordBuilder
from ProteinCollecting import ProteinCollector
from Review.config import Config
from Util import open_xml_paper


def get_separate_keywords(gene):
    return gene.split('_')


def paper_url(paper_name):
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper_name[:-5]}"


def paper_title(paper_name):
    r = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id={paper_name[3:-5]}&retmode=json&tool=Viral-protein-life-cycle-prediction&email=ceder.dens@student.uantwerpen.be")
    return_data = r.json()
    result_keys = list(return_data['result'].keys())
    result_keys.remove('uids')
    return return_data['result'][result_keys[0]]['title']


def to_unicode(s):
    return chr(int('0x' + s[2:], base=16))


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
    return combined_debug_dict


def find_list_in_paper(full_string, sub_list):
    poss = None
    for word in sub_list:
        find = [m.start() for m in re.finditer(word, full_string, re.IGNORECASE)]
        if poss is None:
            poss = [[s] for s in find]
        else:
            to_add = []
            for p_i, p in enumerate(poss):
                for s in find:
                    if s > p[-1]:
                        to_add.append((p_i, s))
                        break
            for p_i, s in to_add:
                poss[p_i].append(s)

    min_len = float('inf')
    min_p = []
    for p in poss:
        if len(p) == len(sub_list):
            if p[-1] - p[0] < min_len:
                min_p = p
                min_len = p[-1] - p[0]
    min_str_positions = (min_p[0], min_p[-1] + len(sub_list[-1]))

    return min_str_positions


class DebugInfo:
    def __init__(self, virus, gene):
        self.all_names = get_separate_keywords(gene)
        self.protein_name = None
        self.gene_name = []
        self.organism = None
        self.host_organism = []
        self.uniprot_accession = None
        self.uniprot_id = None
        self.scores = None
        self.percentages = None
        self.uniprot_info = {}
        self.phases = None
        self.keywords = None
        self.protein_collector = None
        self.virus = virus
        self.gene = gene
        self.context = debug_info(virus, gene)
        self.set_attributes()

    def winning_phase(self):
        return max(self.scores.items(), key=operator.itemgetter(1))[0]

    def context_text(self, context_list, paper):
        full_text = open_xml_paper(f'PaperSelection/Output/selected_papers/{paper}')
        min_str_positions = find_list_in_paper(full_text, context_list)
        text = full_text[min_str_positions[0]:min_str_positions[1]]

        to_bold = self.all_names + ['immediate early'] + list(self.phases.keys()) + [subphase for phase, subphases in
                                                                                     self.phases.items() for
                                                                                     subphase in subphases]

        for name in to_bold:
            sub_text = re.compile('(' + re.escape(name) + ')', re.IGNORECASE)
            text = re.sub(sub_text, r'<strong>\1</strong>', text)

        return '"...' + Markup(text) + '..."'

    def request_function(self, uniprot_id):
        r = requests.get(f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}")
        return_data = r.json()

        for name in return_data["organism"]["names"]:
            if name["type"] == "common":
                self.organism = name["value"]
                break

        for host in return_data["organismHost"]:
            for name in host["names"]:
                if name["type"] == "common":
                    self.host_organism.append(name["value"])
                    break

        for gene in return_data["gene"]:
            for name_type, value_dict_or_list in gene.items():
                if name_type == "name":
                    self.gene_name.append(value_dict_or_list["value"])
                elif name_type in ["orfName", "orfNames"]:
                    self.gene_name += [v["value"] for v in value_dict_or_list]

        self.protein_name = return_data["protein"]["recommendedName"]["fullName"]["value"]
        self.uniprot_accession = return_data["accession"]
        self.uniprot_id = return_data["id"]

        for comment in return_data["comments"]:
            if comment["type"] in ["FUNCTION"]:
                if comment["type"] in self.uniprot_info:
                    self.uniprot_info[comment["type"]].append(comment["text"])
                else:
                    self.uniprot_info[comment["type"]] = comment["text"]

    def set_attributes(self):
        combiner = Combiner(Config.COMBINER_CONFIG)
        combined_counts, sorted_combined_counts, normalized, paper_counts, cutted_index = combiner.read_index(
            self.virus)
        self.scores = combined_counts[self.gene]
        for genes, phases in normalized:
            if genes == self.gene:
                self.percentages = phases
                break

        with open(Config.GENERAL_CONFIG) as general_config_file:
            general_config = json.load(general_config_file)
            self.phases = general_config["phases"]

        self.keywords = KeywordBuilder(Config.KEYWORDS_CONFIG).get_keywords(self.virus)
        self.protein_collector = ProteinCollector(Config.PROTEIN_COLLECTOR_CONFIG)

        all_keys, name_to_headers, header_row = self.keywords
        max_evidence = 6
        max_uniprot_id = ''
        max_sequence = ''
        for kw in self.all_names:
            if kw in header_row:
                sequence, evidence_level = self.protein_collector.read_protein_sequence(kw)
                if evidence_level < max_evidence:
                    max_evidence = evidence_level
                    max_uniprot_id = kw
                    max_sequence = sequence
        self.request_function(max_uniprot_id)
