#!/bin/bash
for PHASE in 'immediate-early' 'early' 'late' 'latent'
do
  tomtom Util/motif_research_short_motifs/motifs/$PHASE/meme.txt Util/motif_research_short_motifs/motifs/elm2018.meme -oc Util/motif_research_short_motifs/compare_motifs/$PHASE
done
