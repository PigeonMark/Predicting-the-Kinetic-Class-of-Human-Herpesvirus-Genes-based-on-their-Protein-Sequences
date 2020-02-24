import os
import operator
import pandas as pd
import requests
from Bio import SeqIO, SeqRecord

from DataCollection.dataCollection import build_keywords


def get_protein_sequence(uniprot_id):
    filename = f"../Classification/Output/sequences/{uniprot_id}.fasta"
    if os.path.isfile(filename):
        # print(f'{uniprot_id}.fasta already existed')
        pass
    else:
        r = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta")
        f = open(filename, "w")
        f.write(r.text)
        f.close()
        print(f'{uniprot_id}.fasta created')


def get_protein_sequences_batch(index, keywords_file):
    all_keys, name_to_headers, header_row = build_keywords(keywords_file)

    for kws, _ in index.items():
        kw_lst = kws.split('_')
        for kw in kw_lst:
            if kw in header_row:
                get_protein_sequence(kw)


def read_protein_sequence(uniprot_id):
    record = SeqIO.read(f"Output/sequences/{uniprot_id}.fasta", "fasta")  # type: SeqRecord
    evidence_level = 0
    for el in record.description.split():
        if el.startswith('PE='):
            evidence_level = int(el[3:])
    if evidence_level == 0:
        print(f'No evidence level found for {uniprot_id}')
    return record.seq, evidence_level


def read_protein_sequences_batch(index, keywords_file, from_classification=False):
    all_keys, name_to_headers, header_row = build_keywords(keywords_file, from_classification)

    sequence_dict = {}
    df = pd.DataFrame(columns=['protein_group', 'protein', 'sequence', 'label'])
    for i, (kws, phases) in enumerate(index):
        kw_lst = kws.split('_')
        max_evidence = 6
        max_uniprot_id = ''
        max_sequence = ''
        for kw in kw_lst:
            if kw in header_row:
                sequence, evidence_level = read_protein_sequence(kw)
                if evidence_level < max_evidence:
                    max_evidence = evidence_level
                    max_uniprot_id = kw
                    max_sequence = sequence

        sequence_dict[kws] = (
            max_sequence, max_uniprot_id, max_evidence, max(phases.items(), key=operator.itemgetter(1))[0])
        df.loc[i] = [kws, max_uniprot_id, str(max_sequence), max(phases.items(), key=operator.itemgetter(1))[0]]

    return df, sequence_dict


if __name__ == "__main__":
    # save_protein_sequence('P10220')
    read_protein_sequence('P10220')
