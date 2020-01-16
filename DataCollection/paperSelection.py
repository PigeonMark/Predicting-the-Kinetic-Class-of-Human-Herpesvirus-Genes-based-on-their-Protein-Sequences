import os
import re
import pickle
import time
import json

from shutil import copy
from input_data import PUNCTUATION


def print_status(done, t_start):
    done += 1
    if done % 100 == 0:
        print(f'{done} papers done in {time.time() - t_start} seconds')
    return done


class Selector:
    def __init__(self, config_filepath, test=False):
        self.directory_list = None
        self.viruses = None
        self.test = test
        self.selected = {}
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.directory_list = config["directory_list"]
            self.viruses = config["viruses"]

            for virus in self.viruses:
                self.selected[virus["name"]] = set()

    def check_stop_early(self, stop_early, done, t_start):
        if stop_early and done == 50000:
            t_end = time.time()
            # Write to pickle file
            pickle.dump(self.selected,
                        open(f'Output/selected_test.p', "wb"))

            for virus_name, selected_lst in self.selected.items():
                print(f'{virus_name}: {len(selected_lst)}')
            print(f'Ended in {t_end - t_start} seconds')

            return True
        return False

    def search_in_paper(self, filepath):
        # The real work: open the xml paper, set to lowercase and remove punctuation
        file = open_xml_paper(filepath)
        content = " ".join(
            file.lower().translate(str.maketrans(PUNCTUATION, ' ' * len(PUNCTUATION), '')).split())

        viruses_found = []
        for virus in self.viruses:
            for alt_name in virus["alternate_names"]:
                if alt_name in content:
                    viruses_found.append(virus["name"])
                    break
        return viruses_found

    def select(self, stop_early=False):
        done = 0
        t_start = time.time()
        # Iterate over all directories, subdirectories and papers
        for directory in self.directory_list:
            for subdir in os.listdir(directory):
                subdir_path = os.path.join(directory, subdir)
                for filename in os.listdir(subdir_path):
                    done = print_status(done, t_start)
                    if self.check_stop_early(stop_early, done, t_start):
                        return
                    filepath = os.path.join(subdir_path, filename)
                    viruses_found = self.search_in_paper(filepath)
                    for virus_name in viruses_found:
                        self.selected[virus_name].add(filepath)

        t_end = time.time()
        # Write to pickle file
        if self.test:
            pickle.dump(self.selected,
                        open(f'Output/selected_test.p', "wb"))
        else:
            pickle.dump(self.selected,
                        open(f'Output/selected.p', "wb"))

        print(f'Ended in {t_end - t_start} seconds')

    def select_from_pickle(self):
        if self.test:
            self.selected = pickle.load(open("Output/selected_test.p", "rb"))
        else:
            self.selected = pickle.load(open("Output/selected.p", "rb"))

    def selected_to_folder(self):
        for virus_name, papers in self.selected.items():
            for paper in papers:
                if self.test:
                    copy(paper, "Output/selected_papers_test/")
                else:
                    copy(paper, "Output/selected_papers/")


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


if __name__ == "__main__":
    test_selector = Selector("config/test_selection_config.json", test=True)
    test_selector.select(stop_early=True)
    # test_selector.select_from_pickle()
    test_selector.selected_to_folder()

    # selector = Selector("config/selection_config.json")
    # selector.select()
    # selector.selected_to_folder()
