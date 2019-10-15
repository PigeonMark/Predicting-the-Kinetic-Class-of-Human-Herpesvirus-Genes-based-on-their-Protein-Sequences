import os
import re
import pickle
from datetime import datetime

PUNCTUATION = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"  # adapted from string.punctuation (removed '-')
PAPER_KEYWORDS = [
    "herpesvirus",
    "herpesviruses",
    "herpes",
    "hsv"
]


def xml_to_text(filename):
    with open(filename, "r") as f:
        notags = re.sub('<[^<]+>', " ", f.read())
        return notags


def select_papers(directory, keywords):

    papers_list = set()
    papers_done = 0
    for subdir in os.listdir(directory):
        dir = os.path.join(directory, subdir)
        for filename in os.listdir(dir):
            
            papers_done += 1
            if papers_done == 50000:
                pickle.dump(papers_list, open("herpespapers_50000_%s.p" % datetime.now().strftime("%Y%m%d-%H%M%S"), "wb"))
                exit(1)

            file = xml_to_text(os.path.join(dir, filename))
            content = file.lower().translate(str.maketrans('\n\t', '  ', PUNCTUATION)).split()

            for word in content:
                if word in keywords:
                    papers_list.add(os.path.join(dir, filename))
                    break
