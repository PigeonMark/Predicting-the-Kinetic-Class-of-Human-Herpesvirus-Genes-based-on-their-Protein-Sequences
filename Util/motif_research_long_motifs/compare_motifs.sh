#!/bin/bash
for PHASE in 'immediate-early' 'early' 'late' 'latent'
do
  tomtom Util/motifs/$PHASE/meme.txt Util/motifs/elm2018.meme -oc Util/compare_motifs/$PHASE
done
