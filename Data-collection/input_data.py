"""
Possible punctuation
"""
PUNCTUATION = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"  # adapted from string.punctuation (removed '-')

"""
The input data used to count the near occurrences for each virus
"""
hsv1_data = {
    "name": "HSV 1",
    "keywords_file": "keywords_10298.csv",
    "papers_directory": "Output/selectedPapers/hsv1_all_20191025-174132/",
    "counted_file": "Output/countingResults/hsv1_all_20191025-174132_keywords_10298.csv_10.p"
}

hsv2_data = {
    "name": "HSV 2",
    "keywords_file": "keywords_10310.csv",
    "papers_directory": "Output/selectedPapers/hsv2_all_20191025-174101/",
    "counted_file": "Output/countingResults/hsv2_all_20191025-174101_keywords_10310.csv_10.p"
}

vzv_data = {
    "name": "Varicella zoster virus",
    "keywords_file": "keywords_10335.csv",
    "papers_directory": "Output/selectedPapers/vzv_all_20191025-174034/",
    "counted_file": "Output/countingResults/vzv_all_20191025-174034_keywords_10335.csv_10.p"
}

ebv_data = {
    "name": "Epstein-Barr virus",
    "keywords_file": "keywords_10376.csv",
    "papers_directory": "Output/selectedPapers/ebv_all_20191025-173954/",
    "counted_file": "Output/countingResults/ebv_all_20191025-173954_keywords_10376.csv_10.p"
}

hcmv_data = {
    "name": "Human cytomegalovirus",
    "keywords_file": "keywords_10359.csv",
    "papers_directory": "Output/selectedPapers/hcmv_all_20191025-173918/",
    "counted_file": "Output/countingResults/hcmv_all_20191025-173918_keywords_10359.csv_10.p"
}

"""
The different phases
"""
MAIN_PHASES = ['immediate-early', 'early', 'early-late', 'late-early', 'late']

ALTERNATE_NAMES = {
    'immediate-early': ['ie', 'immediate early']
}

alpha = [
    'alpha',
    '&#x003b1;',
    '&#945;',
    '&alpha;'
]

"""
A list of alternative names for each virus to search in the papers
"""
HSV_1_KEYWORDS = [
    "hsv-1",
    "hsv1",
    "human alphaherpesvirus 1",
    "herpes simplex virus 1",
    "human herpesvirus 1",
    "human herpesvirus type 1",
    "herpes simplex virus 1 hsv-1",
    "herpes simplex virus hsv-1",
    "herpes simplex virus type 1 hsv-1",
    "herpes simplex virus type 1 hsv1",
    "herpes simplex virus type-1 hsv-1"
]

HSV_2_KEYWORDS = [
    "hsv-2",
    "hsv2",
    "hsv 2",
    "human alphaherpesvirus 2",
    "herpes simplex virus 2",
    "herpes simplex virus II",
    "human herpesvirus 2",
    "human herpesvirus type 2",
    "herpes simplex virus type 2",
    "herpes simplex virus (type 2)"
]

VZV_KEYWORDS = [
    "human alphaherpesvirus 3",
    "hhv3",
    "hhv-3",
    "hhv 3",
    "vzv",
    "human herpesvirus 3",
    "human herpes virus 3",
    "varicella zoster virus",
    "varicella-zoster virus"
]

EBV_KEYWORDS = [
    "human gammaherpesvirus 4",
    "epv",
    "ebv",
    "hhv4",
    "hhv-4",
    "hhv 4",
    "epstein-barr virus"
    "epstein barr virus",
    "human herpesvirus 4",
    "human herpesvirus type 4"
]

HCMV_KEYWORDS = [
    "human betaherpesvirus 5",
    "hhv5",
    "hhv-5"
    "hhv 5",
    "human herpesvirus 5",
    "human herpes virus 5",
    "human herpesvirus type 5",
    "human cytomegalovirus",
    "hcmv"
]


def get_viruses_data():
    return [hsv1_data, hsv2_data, vzv_data, ebv_data, hcmv_data]


def get_phases_data():
    all_phases = set(MAIN_PHASES)
    for k, v in ALTERNATE_NAMES.items():
        all_phases = all_phases.union(v)
    return MAIN_PHASES, ALTERNATE_NAMES, all_phases


if __name__ == "__main__":
    pass
