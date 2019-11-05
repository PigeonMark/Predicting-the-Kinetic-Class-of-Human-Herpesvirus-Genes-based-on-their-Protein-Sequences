import requests
from Bio import SeqIO


def get_protein_sequence(uniprot_id):
    r = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta")
    f = open(f"Output/sequences/{uniprot_id}.fasta", "w")
    f.write(r.text)
    f.close()


def read_protein_sequence(uniprot_id):
    record = SeqIO.read(f"Output/sequences/{uniprot_id}.fasta", "fasta")
    print(record)


if __name__ == "__main__":
    # get_protein_sequence('P10220')
    read_protein_sequence('P10220')
