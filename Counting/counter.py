import os
import pickle
import json

import nltk

from Keywords import KeywordBuilder
from Util import open_xml_paper, filename_from_path, add_to_debug_dict, add_to_near_occ_dict


class Counter:
    def __init__(self, config_filepath):
        self.viruses = None
        self.distance = None
        self.punctuation = None
        self.keywords = {}
        self.phases = None
        self.config = None
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.config = config
            with open(config['general_config']) as general_config_file:
                general_config = json.load(general_config_file)
                self.viruses = general_config["viruses"]
                self.punctuation = general_config["punctuation"]
                self.phases = general_config["phases"]
                self.distance = general_config["distance"]

            for virus in self.viruses:
                self.keywords[virus] = KeywordBuilder(self.config['keywords_config']).get_keywords(virus)

    def __count_near_occ_by_distance(self, word, kw_i, content, near_occ_dict, debug_info_dict):
        """
        Searches for phases within distance of a found keyword in a paper, a phase at distance x of a keyword will
        get 1/x added to its score for that keyword
        :param word:            The keyword found
        :param kw_i:            The index of the keyword found
        :param content:         The content text
        :param near_occ_dict:   The dictionary to add the scores to
        :param debug_info_dict: The dictionary to
        :return:                Nothing, function changes the near_occ_dict
        """
        main_phases = list(self.phases.keys())
        all_phases = main_phases + [phase for k, val in self.phases.items() for phase in val]
        # For each phase
        for phase in all_phases:
            # For each distance
            for dis in range(1, self.distance + 1):
                # Get the indices of the text of distance 'dis'
                i1 = kw_i - dis
                i2 = kw_i + dis

                # Calculate the possible score to add
                to_add = 1 / float(dis)

                # For the indices 'dis' before and 'dis' after the keyword
                for i in [i1, i2]:
                    debug_range_min = max(0, min(i, kw_i) - self.config['debug_distance'])
                    debug_range_max = min(len(content) - 1, max(i, kw_i) + self.config['debug_distance'] + 1)
                    # Check that the indices are still in the content
                    # Check if the word on distance 'dis' of the keyword is a phase
                    if 0 <= i < len(content) and content[i] == phase:
                        # Special case for immediate early
                        if phase == 'early' and content[i - 1] == "immediate":
                            add_to_near_occ_dict(to_add, word, "immediate-early", near_occ_dict)
                            add_to_debug_dict(content[debug_range_min: debug_range_max], word, "immediate-early",
                                              debug_info_dict)

                        else:
                            phase_to_add = phase
                            if phase not in main_phases:
                                for p in main_phases:
                                    if phase in self.phases[p]:
                                        phase_to_add = p
                                        break
                            add_to_near_occ_dict(to_add, word, phase_to_add, near_occ_dict)
                            add_to_debug_dict(content[debug_range_min: debug_range_max], word, phase_to_add,
                                              debug_info_dict)

    def __count_near_occurrences(self, virus_name):
        """
        A function that iterates over a list of papers and counts the distances between the keywords and the phases
        :param virus_name:          The virus to count_all_viruses the occurrences for
        :return:                    A dictionary containing the counts of the near occurrences of all keywords and
                                    phases for each paper, also a sorted index and the location of the pickle file where
                                    the index is stored
        """

        all_keys, name_to_headers, header_row = self.keywords[virus_name]
        papers_list = filename_from_path(pickle.load(open(self.config['selected_papers_file'], 'rb')))[virus_name]

        file_count = 0
        total_file_count = len(papers_list)
        print(f'Counting near occurrences for {virus_name} in {total_file_count} files')
        index = {}
        debug_index = {}
        # For each file in the papers_list
        for filename in papers_list:

            # Open file and make a lowercase list without punctuation and whitespace
            file = open_xml_paper(os.path.join(self.config['selected_papers_directory'], filename))

            stop_words = set(nltk.corpus.stopwords.words('english'))
            words = file.lower().translate(
                str.maketrans('\n\t' + self.punctuation, ' ' * (len(self.punctuation) + 2))).split()
            content = [word for word in words if word not in stop_words and not word.isdigit()]

            near_occ_dict = {}
            debug_info_dict = {}
            # For each word of the paper
            for kw_i, word in enumerate(content):
                # If the word is a keyword
                if word in all_keys:
                    # Check for phases within distance
                    self.__count_near_occ_by_distance(word, kw_i, content, near_occ_dict, debug_info_dict)

            if not len(near_occ_dict) == 0:
                index[filename] = near_occ_dict
                debug_index[filename] = debug_info_dict

            file_count += 1
            if file_count % 1000 == 0:
                print(f'{file_count} files done ({100 * file_count / float(total_file_count):.2f}%)')

        pickle.dump(index,
                    open(self.config['output_count_directory'] + "%s_%i.p" % (virus_name, self.distance), "wb"))
        pickle.dump(debug_index,
                    open(self.config['output_debug_directory'] + "%s_%i.p" % (virus_name, self.distance), "wb"))

    def count_all_viruses(self):
        for virus in self.viruses:
            self.__count_near_occurrences(virus)

    def read_debug_index(self, virus_name):
        return pickle.load(open(self.config['output_debug_directory'] + "%s_%i.p" % (virus_name, self.distance), "rb"))
