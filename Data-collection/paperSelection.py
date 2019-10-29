import os
import re
import pickle
import time

from datetime import datetime
from input_data import HSV_1_KEYWORDS, HSV_2_KEYWORDS, VZV_KEYWORDS, EBV_KEYWORDS, HCMV_KEYWORDS, PUNCTUATION


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
            directory2 = os.path.join(directory, subdir)
            for filename in os.listdir(directory2):

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

                file = open_xml_paper(os.path.join(directory2, filename))
                content = " ".join(
                    file.lower().translate(str.maketrans(PUNCTUATION, ' ' * len(PUNCTUATION), '')).split())

                for word in keywords:
                    if word in content:
                        papers_list.add(os.path.join(directory2, filename))
                        break

    t_end = time.time()
    pickle.dump(papers_list, open("Output/selectedPapers/" + output_file, "wb"))

    print(f'Number of papers selected: {len(papers_list)}')
    print(f'Ended in {t_end - t_start} seconds')


def select_hsv1():
    directory_list = ["Data/comm_use.A-B/", "Data/comm_use.C-H/", "Data/comm_use.I-N/", "Data/comm_use.O-Z/",
                      "Data/non_comm_use.A-B/", "Data/non_comm_use.C-H/", "Data/non_comm_use.I-N/",
                      "Data/non_comm_use.O-Z/"]

    select_papers_in_topic(directory_list, HSV_1_KEYWORDS,
                           "hsv1_all_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"))


def select_hsv2():
    directory_list = ["Data/comm_use.A-B/", "Data/comm_use.C-H/", "Data/comm_use.I-N/", "Data/comm_use.O-Z/",
                      "Data/non_comm_use.A-B/", "Data/non_comm_use.C-H/", "Data/non_comm_use.I-N/",
                      "Data/non_comm_use.O-Z/"]

    select_papers_in_topic(directory_list, HSV_2_KEYWORDS,
                           "hsv2_all_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"))


def select_vzv():
    directory_list = ["Data/comm_use.A-B/", "Data/comm_use.C-H/", "Data/comm_use.I-N/", "Data/comm_use.O-Z/",
                      "Data/non_comm_use.A-B/", "Data/non_comm_use.C-H/", "Data/non_comm_use.I-N/",
                      "Data/non_comm_use.O-Z/"]

    select_papers_in_topic(directory_list, VZV_KEYWORDS,
                           "vzv_all_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"))


def select_ebv():
    directory_list = ["Data/comm_use.A-B/", "Data/comm_use.C-H/", "Data/comm_use.I-N/", "Data/comm_use.O-Z/",
                      "Data/non_comm_use.A-B/", "Data/non_comm_use.C-H/", "Data/non_comm_use.I-N/",
                      "Data/non_comm_use.O-Z/"]

    select_papers_in_topic(directory_list, EBV_KEYWORDS,
                           "ebv_all_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"))


def select_hcmv():
    directory_list = ["Data/comm_use.A-B/", "Data/comm_use.C-H/", "Data/comm_use.I-N/", "Data/comm_use.O-Z/",
                      "Data/non_comm_use.A-B/", "Data/non_comm_use.C-H/", "Data/non_comm_use.I-N/",
                      "Data/non_comm_use.O-Z/"]

    select_papers_in_topic(directory_list, HCMV_KEYWORDS,
                           "hcmv_all_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"))


if __name__ == "__main__":
    select_hsv1()
    # select_hsv2()
    # select_vzv()
    # select_ebv()
    # select_hcmv()
