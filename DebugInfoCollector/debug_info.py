import json
import operator
import os
import re
import pickle
import requests

from flask import Markup
from Combine import Combiner
from Counting import Counter
from Keywords import KeywordBuilder
from ProteinCollecting import ProteinCollector
from Util import open_xml_paper
import xml.etree.ElementTree as ET


def combine_debug_info(virus, gene, counter):
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


def context_text(gene, phases, context_list, paper, paper_directory):
    full_text = open_xml_paper(f'{paper_directory}{paper}')
    min_str_positions = find_list_in_paper(full_text, context_list)
    text = full_text[min_str_positions[0]:min_str_positions[1]]

    to_bold = get_separate_keywords(gene) + \
              ['immediate early'] + list(phases.keys()) + \
              [subphase for phase, subphases in phases.items() for subphase in subphases]

    for name in to_bold:
        sub_text = re.compile('(' + re.escape(name) + ')', re.IGNORECASE)
        text = re.sub(sub_text, r'<strong>\1</strong>', text)

    return '"...' + Markup(text) + '..."'


class DebugInfo:
    def __init__(self):
        self.uniprot_accession = None
        self.uniprot_id = None
        self.protein_name = None
        self.gene_names = []
        self.organism = None
        self.host_organisms = []

        self.phases = None
        self.scores = None
        self.percentages = None
        self.uniprot_info = {}
        self.context = {}

    def __repr__(self):
        ret = ""
        ret += f"self.uniprot_accession: {self.uniprot_accession}\n"
        ret += f"self.uniprot_id: {self.uniprot_id}\n"
        ret += f"self.protein_name: {self.protein_name}\n"
        ret += f"self.gene_names: {self.gene_names}\n"
        ret += f"self.organism: {self.organism}\n"
        ret += f"self.host_organisms: {self.host_organisms}\n"

        ret += f"self.phases: {self.phases}\n"
        ret += f"self.scores: {self.scores}\n"
        ret += f"self.percentages: {self.percentages}\n"
        ret += f"self.uniprot_info: {self.uniprot_info}\n"
        ret += f"self.context: {self.context}\n"

        return ret

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
                    self.host_organisms.append(name["value"])
                    break

        for gene in return_data["gene"]:
            for name_type, value_dict_or_list in gene.items():
                if name_type == "name":
                    self.gene_names.append(value_dict_or_list["value"])
                elif name_type in ["orfName", "orfNames"]:
                    self.gene_names += [v["value"] for v in value_dict_or_list]

        self.protein_name = return_data["protein"]["recommendedName"]["fullName"]["value"]
        self.uniprot_accession = return_data["accession"]
        self.uniprot_id = return_data["id"]

        for comment in return_data["comments"]:
            if comment["type"] in ["FUNCTION"]:
                if comment["type"] in self.uniprot_info:
                    self.uniprot_info[comment["type"]].append(comment["text"])
                else:
                    self.uniprot_info[comment["type"]] = comment["text"]

    def collect(self, virus, gene, combined_counts, normalized_counts, keywords, protein_collector, counter, phases,
                paper_directory):
        self.phases = phases
        self.request_function(get_uniprot_id(gene, protein_collector, keywords))

        self.scores = combined_counts[gene]
        for genes, phss in normalized_counts:
            if genes == gene:
                self.percentages = phss
                break

        combined_debug_info = combine_debug_info(virus, gene, counter)
        for phase, papers in combined_debug_info.items():
            for paper, contexts in papers.items():
                for context in contexts:
                    if phase in self.context:
                        if paper in self.context[phase]:
                            self.context[phase][paper].append(
                                context_text(gene, phases, context, paper, paper_directory))
                        else:
                            self.context[phase][paper] = [context_text(gene, phases, context, paper, paper_directory)]
                    else:
                        self.context[phase] = {paper: [context_text(gene, phases, context, paper, paper_directory)]}

    def winning_phase(self):
        return max(self.scores.items(), key=operator.itemgetter(1))[0]


def paper_title(paper_directory, paper_name):
    tree = ET.parse(f"{paper_directory}{paper_name}")
    root = tree.getroot()
    for title_group in root.iter('title-group'):
        return title_group.find('article-title').text


class DebugInfoCollector:
    def __init__(self, config_filepath):
        self.config = None
        self.viruses = None
        self.phases = None
        self.combiner = None
        self.counter = None
        self.keywords = {}
        self.protein_collector = None
        self.__read_config(config_filepath)

        self.debug_info = {}
        self.title_dict = {}

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.config = config
            with open(config['general_config']) as general_config_file:
                general_config = json.load(general_config_file)
                self.viruses = general_config["viruses"]
                self.phases = general_config["phases"]

            self.combiner = Combiner(config["combiner_config"])
            self.counter = Counter(config["counter_config"])
            self.protein_collector = ProteinCollector(config["protein_collector_config"])

            for virus in self.viruses:
                self.keywords[virus] = KeywordBuilder(self.config['keywords_config']).get_keywords(virus)

    def save_debug_info(self):
        pickle.dump(self.debug_info, open(self.config['debug_info_output_file'], 'wb'))

    def save_paper_titles(self):
        pickle.dump(self.title_dict, open(self.config['title_output_file'], 'wb'))

    def load_debug_info(self):
        return pickle.load(open(self.config['debug_info_output_file'], 'rb'))

    def load_paper_titles(self):
        return pickle.load(open(self.config['title_output_file'], 'rb'))

    def collect(self):
        self.title_dict = self.load_paper_titles()
        for virus in self.viruses:
            combined_counts, sorted_combined_counts, normalized, paper_counts, cutted_index = self.combiner.read_index(
                virus)
            for gene, values in combined_counts.items():
                debug_i = DebugInfo()
                debug_i.collect(virus, gene, combined_counts, normalized, self.keywords[virus], self.protein_collector,
                                self.counter, self.phases, self.config['paper_selection_directory'])

                if virus in self.debug_info:
                    self.debug_info[virus][gene] = debug_i
                else:
                    self.debug_info[virus] = {gene: debug_i}

                break
            break

        num_papers = len(os.listdir(self.config['paper_selection_directory']))
        done = 0
        for paper in os.listdir(self.config['paper_selection_directory']):
            if paper not in self.title_dict:
                self.title_dict[paper] = paper_title(self.config['paper_selection_directory'], paper)
            done += 1
            if done % 1000 == 0:
                print(f"Done {done}/{num_papers} ({100*done/float(num_papers):.2f}%)")

        self.save_debug_info()
        self.save_paper_titles()
