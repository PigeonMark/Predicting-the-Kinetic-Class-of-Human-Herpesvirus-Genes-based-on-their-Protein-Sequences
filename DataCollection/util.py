import re
import os


def open_xml_paper(filename):
    """
    Open xml file and strip xml tags of the form <tag>...</tag>
    :param filename:    The xml file to open
    :return:            The string of the stripped xml file
    """
    with open(filename, "r") as f:
        paper = f.read()
        matches = re.search(r"<back>.*</back>", paper, re.MULTILINE | re.DOTALL)
        if matches is not None:
            paper = paper[:matches.start()] + paper[matches.end():]

        no_tags = re.sub('<[^<]+>', " ", paper)
        return no_tags


def filename_from_path(file_list_dict):
    new = {}
    for virus, file_list in file_list_dict.items():
        new_list = []
        for file in file_list:
            new_list.append(os.path.basename(file))
        new[virus] = new_list
    return new
