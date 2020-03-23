import json

from Keywords import KeywordBuilder
from ProteinCollecting import ProteinCollector
from Util import ReviewDBReader, get_uniprot_id
import pandas as pd


class BaseData:
    def __init__(self, config_filepath):
        self.review_db_reader = None  # type: ReviewDBReader
        self.protein_collector = None  # type: ProteinCollector
        self.viruses = None
        self.keywords = {}
        self.output_directory = None
        self.data_frame = None  # type: pd.DataFrame

        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.review_db_reader = ReviewDBReader(config['review_db_reader_config'])
            self.protein_collector = ProteinCollector(config['protein_collector_config'])
            self.output_directory = config['output_directory']

            with open(config['general_config']) as general_config_file:
                general_config = json.load(general_config_file)
                self.viruses = general_config["viruses"]

            for virus in self.viruses:
                self.keywords[virus] = KeywordBuilder(config['keywords_config']).get_keywords(virus)

    def create_df(self):
        self.data_frame = pd.DataFrame(columns=['virus', 'protein_group', 'protein', 'sequence', 'label'])
        i = 0
        for review in self.review_db_reader.get_all():
            if review.review_status in ['CORRECT', 'MODIFIED']:
                uniprot_id = get_uniprot_id(review.names, self.protein_collector, self.keywords[review.virus])
                seq, evidence = self.protein_collector.read_protein_sequence(uniprot_id)
                self.data_frame.loc[i] = [review.virus, review.names, uniprot_id, str(seq), review.reviewed_phase]
                i += 1

    def save(self):
        self.data_frame.to_csv(f"{self.output_directory}base_data.csv")

    def create_data(self):
        self.create_df()
        self.save()
