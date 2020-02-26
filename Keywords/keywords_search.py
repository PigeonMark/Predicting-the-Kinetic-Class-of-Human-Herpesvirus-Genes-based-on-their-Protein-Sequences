# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def fetch_search_queries(taxid):
    """Query the UniProt REST API for the names and keywords for all
    reviewed entries for a given taxonomic identifier.

    Retrieves accession numbers, names, protein, gene and keyword annotations.
    
    Parameters
    ----------
    taxid : str
        Taxonomic identifier.
    
    Returns
    -------
    dict
        A dictionary containing UniProt accession numbers as keys and
        dictionaries of different types of annotations as their values.
    set
        A set of all lineage names.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"querying UniProt REST API for reviewed entries for taxid:{taxid}")

    payload = {"query": f'taxonomy: "{taxid}" AND reviewed:yes', "format": "xml"}

    try:
        response = requests.get("http://www.uniprot.org/uniprot/", params=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        logger.error("Encountered error while accessing UniProt", exc_info=True)
        sys.exit(1)
    except requests.exceptions.HTTPError as err:
        print(err)
        logger.error("Encountered error while accessing UniProt", exc_info=True)
        sys.exit(1)

    tree = ET.fromstring(response.text)
    return tree


def extract_lineages(tree):
    ns = "{http://uniprot.org/uniprot}"

    # Store all lineage information except the top level
    lineages = set()
    for entry in tree.iter(tag=f'{ns}entry'):
        lineages.update([ele.text for ele in entry.iterfind(f'{ns}organism/{ns}lineage/{ns}taxon')])
    return lineages


def extract_gene_names(tree):
    ns = "{http://uniprot.org/uniprot}"

    results = {}
    # Iterate over all entries
    for entry in tree.iter(tag=f"{ns}entry"):
        main_ac = [ele.text for ele in entry.iter(f"{ns}accession")][0]
        results[main_ac] = {}

        # All accession elements
        results[main_ac]["uniprot_ac"] = [
            ele.text for ele in entry.iter(tag=f"{ns}accession") if ele.text.strip()
        ]
        # All root name elements
        results[main_ac]["names"] = [
            ele.text
            for ele in entry.iterfind(f"{ns}name")
            if ele.text.strip()
        ]
        # All names under protein element
        results[main_ac]["protein"] = [
            ele.text for ele in entry.iterfind(f".//{ns}protein//") if ele.text.strip() and ele.tag != f'{ns}ecNumber'
        ]
        # All names under gene element
        results[main_ac]["gene"] = [
            ele.text for ele in entry.iterfind(f".//{ns}gene/.//") if ele.text.strip()
        ]
        # All keywords
        results[main_ac]["keyword"] = [
            (ele.attrib, ele.text) for ele in entry.iterfind(f".//{ns}keyword") if ele.text.strip()
        ]
        # All organism names
        results[main_ac]["organism"] = [
            ele.text for ele in entry.iterfind(f'{ns}organism/{ns}name') if ele.text.strip()
        ]

    return results


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Get path to project root directory
    project_dir = Path(__file__).resolve().parents[2]

    # Initialise cli argument parser
    parser = argparse.ArgumentParser(
        description="Script to retrieve search keywords for a given taxid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--taxid",
        dest="taxid",
        default="10292",
        # Taxid 10292 = Herpesviridae
        # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=10292
        type=str,
        help="Full mapping file from EBI GOA. Omitting this will default to the online UniProt mapping service.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default=project_dir / "data" / "interim" / "annotation_data",
        help='Output directory. Files are saved as "keywords_taxid.txt" and "keywords_taxid.csv".',
    )
    args = parser.parse_args()

    # Query UniProt
    tree = fetch_search_queries(args.taxid)

    # Parse XML object
    lineages = extract_lineages(tree)
    results = extract_gene_names(tree)

    # Remove high level entries from lineage set
    print(lineages)
    # [lineages.remove(i) for i in ['Viruses', 'dsDNA viruses, no RNA stage', 'Herpesvirales']]

    # Save results...
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    # ...as csv with keywords (for later lookup)
    logger.info(
        f"Exporting {out_path}/keywords_{args.taxid}.csv."
    )
    df = pd.DataFrame.from_dict(results)
    df.to_csv(out_path / f"keywords_{args.taxid}.csv")

    # ...as organism keyword list for grep (without keywords and NO BLANK LINES)
    logger.info(
        f"Exporting {out_path}/lineages_{args.taxid}.txt."
    )
    organism_list = np.unique(np.hstack(df.transpose()["organism"].values.ravel()))
    new_line = ""
    with open(out_path / f"lineages_{args.taxid}.txt", "w") as f:
        for word in organism_list:
            f.write(new_line + word)
            new_line = "\n"

    # ...as keyword list for grep (without keywords and NO BLANK LINES)
    logger.info(
        f"Exporting {out_path}/keywords_{args.taxid}.txt."
    )
    keyword_list = np.unique(np.hstack(df.transpose().drop("keyword", axis=1).values.ravel()))
    # prevent blank line at end of file to avoid meaningless grep result
    new_line = ""
    with open(out_path / f"keywords_{args.taxid}.txt", "w") as f:
        for word in lineages:
            f.write(new_line + word)
            new_line = "\n"
    with open(out_path / f"keywords_{args.taxid}.txt", "a") as f:
        for word in keyword_list:
            f.write(new_line + word)
