import os
import re
import pickle
from datetime import datetime
import time

PUNCTUATION = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"  # adapted from string.punctuation (removed '-')
PAPER_KEYWORDS = [
    "herpesvirus",
    "herpesviruses",
    "herpes",
    "hsv"
]

HSV_1_KEYWORDS = [
    "hsv-1",
    "hsv1",
    "human alphaherpesvirus 1",
    "herpes simplex virus 1",
    "human herpesvirus 1",
    "human herpesvirus type 1",
    "herpes simplex virus 1 hsv-1",
    "herpes simplex virus hsv-1",
    "herpes simplex virus type 1 hsv-1",
    "herpes simplex virus type 1 hsv1",
    "herpes simplex virus type-1 hsv-1"
]

HSV_2_KEYWORDS = [
    "hsv-2",
    "hsv2",
    "hsv 2",
    "human alphaherpesvirus 2",
    "herpes simplex virus 2",
    "herpes simplex virus II",
    "human herpesvirus 2",
    "human herpesvirus type 2",
    "herpes simplex virus type 2",
    "herpes simplex virus (type 2)"
]

VZV_KEYWORDS = [
    "human alphaherpesvirus 3",
    "hhv3",
    "hhv-3",
    "hhv 3",
    "vzv",
    "human herpesvirus 3",
    "human herpes virus 3",
    "varicella zoster virus",
    "varicella-zoster virus"
]

EBV_KEYWORDS = [
    "human gammaherpesvirus 4",
    "epv",
    "ebv",
    "hhv4",
    "hhv-4",
    "hhv 4",
    "epstein-barr virus"
    "epstein barr virus",
    "human herpesvirus 4",
    "human herpesvirus type 4"
]

HCMV_KEYWORDS = [
    "human betaherpesvirus 5",
    "hhv5",
    "hhv-5"
    "hhv 5",
    "human herpesvirus 5",
    "human herpes virus 5",
    "human herpesvirus type 5",
    "human cytomegalovirus",
    "hcmv"
]

alpha = [
    'alpha',
    '&#x003b1;',
    '&#945;',
    '&alpha;'
]


def open_xml_paper(filename):
    with open(filename, "r") as f:
        no_tags = re.sub('<[^<]+>', " ", f.read())
        return no_tags


def select_papers_in_topic(directory_list, keywords, output_file, stop_early=False):
    papers_list = set()
    papers_done = 0
    t_start = time.time()

    for directory in directory_list:
        for subdir in os.listdir(directory):
            dir = os.path.join(directory, subdir)
            for filename in os.listdir(dir):

                papers_done += 1
                if papers_done % 1000 == 0:
                    print(f'{len(papers_list)} of {papers_done} selected in {time.time() - t_start} seconds')

                if stop_early:
                    if papers_done == 50000:
                        t_end = time.time()
                        pickle.dump(papers_list, open(output_file, "wb"))

                        print(f'Number of papers selected: {len(papers_list)}')
                        print(f'Ended in {t_end - t_start} seconds')

                        return

                file = open_xml_paper(os.path.join(dir, filename))
                content = " ".join(
                    file.lower().translate(str.maketrans(PUNCTUATION, ' ' * len(PUNCTUATION), '')).split())

                for word in keywords:
                    if word in content:
                        papers_list.add(os.path.join(dir, filename))
                        break

    t_end = time.time()
    pickle.dump(papers_list, open("Output/selectedPapers/" + output_file, "wb"))

    print(f'Number of papers selected: {len(papers_list)}')
    print(f'Ended in {t_end - t_start} seconds')


if __name__ == "__main__":
    directory_list = ["Data/comm_use.A-B/", "Data/comm_use.C-H/", "Data/comm_use.I-N/", "Data/comm_use.O-Z/",
                      "Data/non_comm_use.A-B/", "Data/non_comm_use.C-H/", "Data/non_comm_use.I-N/",
                      "Data/non_comm_use.O-Z/"]
    length = 0
    for directory in directory_list:
        for subdir in os.listdir(directory):
            dir = os.path.join(directory, subdir)
            length += len(os.listdir(dir))

    print(length)

    select_papers_in_topic(directory_list, HCMV_KEYWORDS,
                           "hcmv_all_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"))
