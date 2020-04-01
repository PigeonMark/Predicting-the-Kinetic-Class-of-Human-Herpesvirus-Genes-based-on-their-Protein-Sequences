import json

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd
from Util import ReviewDBReader


def create_fasta(config_file):
    with open(config_file) as config_file:
        config = json.load(config_file)
        data = pd.read_csv(config["input_csv_file"])
    review_records = ReviewDBReader(config['review_db_reader_config']).get_all()
    records = {'immediate-early': [], 'early': [], 'late': [], 'latent': []}
    for i, row in data.iterrows():
        names = None
        for record in review_records:
            if row['protein'] in record.names:
                names = record.names
                break

        record = SeqRecord(Seq(row['sequence']), id=f"{row['virus']}_{names}", name=f"{row['virus']}_{row['protein']}", description=f"{row['label']} protein of {row['virus']}")
        records[row['label']].append(record)

    for phase, recs in records.items():
        SeqIO.write(recs, f"{config['output_directory']}{phase}.fasta", "fasta")

