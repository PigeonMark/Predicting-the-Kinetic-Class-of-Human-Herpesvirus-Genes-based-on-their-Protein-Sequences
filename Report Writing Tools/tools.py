import json


def virus_alternate_names_overview():
    configfile = open('../config/selection_config.json')
    config = json.load(configfile)


    an = config['alternate_names']
    viruses = list(an.keys())
    viruses1 = viruses[:4]
    viruses2 = viruses[4:]

    remaining = True
    i = 0
    while remaining:
        remaining = False
        to_print = ""
        for virus in viruses1:
            if i < len(an[virus]):
                to_print += f"{an[virus][i]} & "
                remaining = True
            else:
                to_print += " & "
        to_print = to_print[:-2]
        to_print += '\\\\\n'
        print(to_print, end='')
        i += 1

    print()
    print()
    remaining = True
    i = 0
    while remaining:
        remaining = False
        to_print = ""
        for virus in viruses2:
            if i < len(an[virus]):
                to_print += f"{an[virus][i]} & "
                remaining = True
            else:
                to_print += " & "
        to_print = to_print[:-2]
        to_print += '\\\\\n'
        print(to_print, end='')
        i += 1


if __name__ == "__main__":
    virus_alternate_names_overview()
