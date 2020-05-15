#!/bin/bash
for PHASE1 in 'immediate-early' 'early' 'late' 'latent'
do
  for PHASE2 in 'immediate-early' 'early' 'late' 'latent'
  do
    fimo -oc Util/find_motifs/"$PHASE1"_$PHASE2 Util/motifs/$PHASE1/meme.txt Util/multi_sequence_fasta/$PHASE2.fasta
  done
done
