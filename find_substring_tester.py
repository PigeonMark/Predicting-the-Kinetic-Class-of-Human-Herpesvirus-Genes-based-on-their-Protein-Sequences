import nltk
import re

from Util import open_xml_paper


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
        if p[-1] - p[0] < min_len:
            min_p = p
            min_len = p[-1] - p[0]
    min_str = full_string[min_p[0]:min_p[-1] + len(sub_list[-1])]

    return min_str


if __name__ == "__main__":
    punc = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"
    full_string = "A happy quotation string contianing some !? quotation and some stopwords!. ALSO SoMe capitals or other weird stufkessssss... and string again also capitals or stopwords"

    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = full_string.lower().translate(
        str.maketrans('\n\t' + punc, ' ' * (len(punc) + 2))).split()
    content = [word for word in words if word not in stop_words and not word.isdigit()]
    full_string = open_xml_paper('PaperSelection/Output/selected_papers/PMC6220328.nxml')

    sub_list = ['analyses', 'investigate', 'role', 'us10', 'viral', 'replication', 'cycle', 'performed', 'multistep',
                'replication', 'analyses', 'bac-g', 'x00394', 'us10', 'us10frt', 'described', 'materials', 'methods',
                'defs', 'infected', 'corresponding', 'viruses', 'moi', 'early', 'stage', 'infection']

    find_list_in_paper(full_string, sub_list)
