#!/bin/bash
for PHASE in 'immediate-early' 'early' 'late' 'latent'
do
  meme Util/multi_sequence_fasta/$PHASE.fasta -oc Util/motifs/$PHASE -nmotifs 5 -minsites 3
done
