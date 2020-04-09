#!/bin/bash
for PHASE in 'immediate-early' 'early' 'late' 'latent'
do
  glam2 p Util/multi_sequence_fasta/$PHASE.fasta -O Util/gapped_motifs/$PHASE
done
