import json
import os
import requests
from Bio import SeqIO


class ProteinCollector:
    def __init__(self, config_filepath):
        self.output_directory = None
        self.request_url = None
        self.__read_config(config_filepath)

    def __read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            self.output_directory = config['output_directory']
            self.request_url = config['request_url']

    def save_protein_sequence(self, uniprot_id):
        filename = os.path.join(self.output_directory, uniprot_id) + ".fasta"
        if not os.path.isfile(filename):
            done = False
            while not done:
                try:
                    r = requests.get(f"{self.request_url}{uniprot_id}.fasta")
                    f = open(filename, "w")
                    f.write(r.text)
                    f.close()
                    print(f"saved {uniprot_id}.fasta")
                    done = True
                except Exception as e:
                    print(f"Exception in save_protein_sequence with {uniprot_id}.fasta")
                    print(e)
                    print("trying again...")

    def read_protein_sequence(self, uniprot_id):
        self.save_protein_sequence(uniprot_id)

        record = SeqIO.read(os.path.join(self.output_directory, uniprot_id) + ".fasta", "fasta")  # type: SeqRecord
        evidence_level = 0
        for el in record.description.split():
            if el.startswith('PE='):
                evidence_level = int(el[3:])
        if evidence_level == 0:
            print(f'No evidence level found for {uniprot_id}')
        return record.seq, evidence_level
