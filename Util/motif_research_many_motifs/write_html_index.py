def main():
    out = open("motif_index.html", "w")
    out.write("""<!DOCTYPE html>
<html lang="en">
<head>
<style>
        * {
            box-sizing: border-box;
        }

        /* Create two unequal columns that floats next to each other */
        .column {
            float: left;
            padding: 10px;
        }

        .left {
            width: 15%;
        }

        .right {
            width: 85%;
        }

        /* Clear floats after the columns */
        .row:after {
            content: "";
            display: table;
            clear: both;
        }
    </style>
    <meta charset="UTF-8">
    <title>Find Motifs</title>
</head>
<body>
<div class="row">
    <div class="column left">\n""")

    out.write("""<h1>Search Motifs</h1>""")
    scripts = ""
    for phase in ['immediate-early', 'early', 'late', 'latent']:
        out.write(f'<a id="search_{phase}" href="#">{phase}</a><br>\n')
        scripts += f"document.getElementById('search_{phase}').onclick = function () " + "{\n" + \
                       "\tdocument.getElementById(" + """'right-column').innerHTML = '<h1>""" + \
                       f"Motifs of {phase} sequences" + """</h1><iframe src="./motifs/""" + \
                       f"{phase}" + """/meme.html" style="position:fixed; width:84%; height:89%; """ + \
                       """border:none; margin:0; padding:0; overflow:hidden; z-index:999999;"></iframe>';
            };\n"""

    # out.write("""<h1>Find Motifs</h1>\n""")
    # for phase in ['immediate-early', 'early', 'late', 'latent']:
    #     out.write(f"<h2>{phase} motifs</h2>\n")
    #     for phase2 in ['immediate-early', 'early', 'late', 'latent']:
    #         out.write(f'<a id="find_{phase}_{phase2}" href="#">{phase2} sequences</a><br>\n')
    #         scripts += f"document.getElementById('find_{phase}_{phase2}').onclick = function () " + "{\n" + \
    #                    "\tdocument.getElementById(" + """'right-column').innerHTML = '<h1>""" + \
    #                    f"{phase} motifs found in {phase2} sequences" + """</h1><iframe src="./find_motifs/""" + \
    #                    f"{phase}_{phase2}" + """/fimo.html" style="position:fixed; width:84%; height:89%; """ + \
    #                    """border:none; margin:0; padding:0; overflow:hidden; z-index:999999;"></iframe>';
    #         };\n"""
    #
    # out.write("""<h1>Compare Motifs</h1>\n""")
    # for phase in ['immediate-early', 'early', 'late', 'latent']:
    #     out.write(f'<a id="compare_{phase}" href="#">{phase}</a><br>\n')
    #     scripts += f"document.getElementById('compare_{phase}').onclick = function () " + "{\n" + \
    #                    "\tdocument.getElementById(" + """'right-column').innerHTML = '<h1>""" + \
    #                    f"Compared motifs of {phase} against known motifs" + """</h1><iframe src="./compare_motifs/""" + \
    #                    f"{phase}" + """/tomtom.html" style="position:fixed; width:84%; height:89%; """ + \
    #                    """border:none; margin:0; padding:0; overflow:hidden; z-index:999999;"></iframe>';
    #         };\n"""

    out.write("""</div>
    <div id="right-column" class="column right">
    </div>
</div>
</body>
<script>
""")
    out.write(scripts)
    out.write("""</script>
</html>""")


if __name__ == "__main__":
    main()
