#!/bin/bash
for PHASE in 'immediate-early' 'early' 'late' 'latent'
do
  meme Util/multi_sequence_fasta/$PHASE.fasta -oc Util/motif_research_short_motifs/motifs/$PHASE -nmotifs 5 -minsites 3 -minw 3 -maxw 10
done
