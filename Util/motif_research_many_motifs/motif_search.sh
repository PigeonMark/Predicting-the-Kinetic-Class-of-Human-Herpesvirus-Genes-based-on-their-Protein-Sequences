#!/bin/bash
for PHASE in 'immediate-early'
do
  meme Util/multi_sequence_fasta/$PHASE.fasta -oc Util/motif_research_many_motifs/motifs/$PHASE -nmotifs 20 -minsites 3
done
