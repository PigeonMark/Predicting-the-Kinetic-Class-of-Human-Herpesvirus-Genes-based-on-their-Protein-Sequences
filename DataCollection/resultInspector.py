import os
import pickle

import nltk

from DataCollection.input_data import get_viruses_data, PUNCTUATION
from helper import print_debug_gene
from DataCollection.paperSelection import open_xml_paper


def get_info_by_gene(virus_data, gene, print_paper_stripping_steps=False):
    index, sorted_index = pickle.load(open(virus_data["counted_file"], "rb"))
    debug_index = pickle.load(open(virus_data["debug_file"], "rb"))
    print(f'Debug info for gene: {gene}')
    for paper, kws in index.items():
        if gene in kws:
            print_debug_gene(paper, debug_index[paper][gene])
            if print_paper_stripping_steps:
                # Open file and make a lowercase list without punctuation and whitespace
                file = open_xml_paper(os.path.join(virus_data['papers_directory'], paper))
                print(file)
                stop_words = set(nltk.corpus.stopwords.words('english'))

                words = file.lower().translate(
                    str.maketrans('\n\t' + PUNCTUATION, ' ' * (len(PUNCTUATION) + 2))).split()
                print(words)
                content = [word for word in words if word not in stop_words and not word.isdigit()]
    print('----\n')


def main():
    viruses_data = get_viruses_data()
    get_info_by_gene(viruses_data[0], 'ul1')  # toeval
    get_info_by_gene(viruses_data[0], 'ul12')  # toeval
    get_info_by_gene(viruses_data[0], 'us4')  # toeval

    get_info_by_gene(viruses_data[0], 'ul17')  # toeval
    get_info_by_gene(viruses_data[0], 'us9')  # toeval

    get_info_by_gene(viruses_data[0], 'rl1')  # toeval

    get_info_by_gene(viruses_data[0], 'ul4')  # toeval

    get_info_by_gene(viruses_data[0], 'ul14')  # niet duidelijk, waarschijnlijk fout
    get_info_by_gene(viruses_data[0], 'us6')  # correct, ookal is score nipt
    get_info_by_gene(viruses_data[0], 'ul49')  # onduidelijk
    get_info_by_gene(viruses_data[0], 'ul41')  # toeval

    get_info_by_gene(viruses_data[0], 'ul6')  # correct
    get_info_by_gene(viruses_data[0], 'ul46')  # correct
    get_info_by_gene(viruses_data[0], 'ul26')  # correct
    get_info_by_gene(viruses_data[0], 'ul13')  # correct

    get_info_by_gene(viruses_data[0],
                     'ul56')  # Heel nipte score, slechts 1 paper zegt duidelijk dat hij late is, late heeft ook de hoogste score
    get_info_by_gene(viruses_data[0],
                     'ul15')  # Duidelijke score maar afkomstig uit slechts 1 paper, wel correct resultaat
    get_info_by_gene(viruses_data[0],
                     'ul16')  # Duidelijke score maar afkomstig uit slechts 1 paper, wel correct resultaat


if __name__ == "__main__":
    main()
