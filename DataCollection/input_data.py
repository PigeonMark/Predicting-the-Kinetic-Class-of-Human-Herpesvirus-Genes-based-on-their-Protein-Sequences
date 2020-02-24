"""
Possible punctuation
"""
PUNCTUATION = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"  # adapted from string.punctuation (removed '-')

"""
The input data used to count_all_viruses the near occurrences for each virus
"""
hsv1_data = {
    "name": "HSV 1",
    "keywords_file": "keywords_10298.csv",
    "papers_directory": "DataCollection/Output/selectedPapers/hsv1_all_20191025-174132/",
    "counted_file": "DataCollection/Output/countingResults/hsv1_all_20191025-174132_keywords_10298.csv_10.p",
    "debug_file": "DataCollection/Output/debug_info/hsv1_all_20191025-174132_keywords_10298.csv_10.p"
}

hsv2_data = {
    "name": "HSV 2",
    "keywords_file": "keywords_10310.csv",
    "papers_directory": "DataCollection/Output/selectedPapers/hsv2_all_20191025-174101/",
    "counted_file": "DataCollection/Output/countingResults/hsv2_all_20191025-174101_keywords_10310.csv_10.p",
    "debug_file": "DataCollection/Output/debug_info/hsv2_all_20191025-174101_keywords_10310.csv_10.p"
}

vzv_data = {
    "name": "Varicella zoster virus",
    "keywords_file": "keywords_10335.csv",
    "papers_directory": "DataCollection/Output/selectedPapers/vzv_all_20191025-174034/",
    "counted_file": "DataCollection/Output/countingResults/vzv_all_20191025-174034_keywords_10335.csv_10.p",
    "debug_file": "DataCollection/Output/debug_info/vzv_all_20191025-174034_keywords_10335.csv_10.p"
}

ebv_data = {
    "name": "Epstein-Barr virus",
    "keywords_file": "keywords_10376.csv",
    "papers_directory": "DataCollection/Output/selectedPapers/ebv_all_20191025-173954/",
    "counted_file": "DataCollection/Output/countingResults/ebv_all_20191025-173954_keywords_10376.csv_10.p",
    "debug_file": "DataCollection/Output/debug_info/ebv_all_20191025-173954_keywords_10376.csv_10.p"
}

hcmv_data = {
    "name": "Human cytomegalovirus",
    "keywords_file": "keywords_10359.csv",
    "papers_directory": "DataCollection/Output/selectedPapers/hcmv_all_20191025-173918/",
    "counted_file": "DataCollection/Output/countingResults/hcmv_all_20191025-173918_keywords_10359.csv_10.p",
    "debug_file": "DataCollection/Output/debug_info/hcmv_all_20191025-173918_keywords_10359.csv_10.p"
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


def get_viruses_data():
    return [hsv1_data, hsv2_data, vzv_data, ebv_data, hcmv_data]


def get_phases_data():
    all_phases = set(MAIN_PHASES)
    for k, v in ALTERNATE_NAMES.items():
        all_phases = all_phases.union(v)
    return MAIN_PHASES, ALTERNATE_NAMES, all_phases


if __name__ == "__main__":
    pass
