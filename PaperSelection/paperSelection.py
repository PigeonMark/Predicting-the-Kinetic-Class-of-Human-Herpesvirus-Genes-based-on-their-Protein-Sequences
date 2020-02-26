import os
import pickle
import time
import json

from shutil import copy
from Util import open_xml_paper


def print_status(done, t_start):
    done += 1
    if done % 1000 == 0:
        print(f'{done} done (in {time.time() - t_start:.2f} seconds)')
    return done


class Selector:
    def __init__(self, config_filepath):
        self.input_directory_list = None
        self.output_file = None
        self.output_directory = None
        self.viruses = None
        self.alternate_names = None
        self.punctuation = None
        self.selected = {}
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            with open(config['general_config']) as general_config_file:
                general_config = json.load(general_config_file)
                self.viruses = general_config["viruses"]
                self.punctuation = general_config["punctuation"]

            self.output_file = config["output_file"]
            self.output_directory = config["output_directory"]
            self.input_directory_list = config["input_directory_list"]
            self.alternate_names = config["alternate_names"]

            for virus in self.viruses:
                self.selected[virus] = set()

    def check_stop_early(self, stop_early, done, t_start):
        if stop_early and done == 5000:
            t_end = time.time()
            # Write to pickle file
            pickle.dump(self.selected, open(self.output_file, "wb"))

            for virus_name, selected_lst in self.selected.items():
                print(f'{virus_name}: {len(selected_lst)}')
            print(f'Ended in {t_end - t_start} seconds')

            return True
        return False

    def search_in_paper(self, filepath):
        # The real work: open the xml paper, set to lowercase and remove punctuation
        file = open_xml_paper(filepath)
        content = " ".join(
            file.lower().translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation), '')).split())

        viruses_found = []

        for virus in self.viruses:
            for alt_name in self.alternate_names[virus]:
                if alt_name in content:
                    if content.count('bovine') <= content.count('human'):
                        viruses_found.append(virus)
                        break
        return viruses_found

    def select(self, stop_early=False):
        done = 0
        t_start = time.time()
        # Iterate over all directories, subdirectories and papers
        for directory in self.input_directory_list:
            total = sum([len(os.listdir(os.path.join(directory, subdir))) for subdir in os.listdir(directory)])
            print(f"Selecting papers from {directory} ({total} in total)")
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
            print(f"Selecting from {directory} done (in {time.time() - t_start:.2f} seconds)")

        t_end = time.time()
        # Write to pickle file
        pickle.dump(self.selected, open(self.output_file, "wb"))

        print(f'All done (in {t_end - t_start:.2f} seconds)')

    def select_from_pickle(self):
        self.selected = pickle.load(open(self.output_file, "rb"))

    def selected_to_folder(self):
        for virus_name, papers in self.selected.items():
            for paper in papers:
                copy(paper, self.output_directory)
