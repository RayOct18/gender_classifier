#!/bin/bash

dir=("gan_morph_wgangp_z512_arcface_10llr_Din_flip_0001l2_0_3_ENB_G-4layer_EBDgdr/100k" \
     "gan_morph_wgangp_z512_arcface_10llr_Din_flip_0001l2_0_3_ENB_G-4layer_EBDgdr/90k" \
     "gan_morph_wgangp_z512_arcface_10llr_Din_flip_0001l2_0_3_ENB_G-4layer_EBDgdr/50k" \
     "gan_morph_wgangp_z512_arcface_10llr_Din_flip_0001l2_0_3_ENB_G-4layer_EBDgdr/10k" \
     )

for i in ${dir[@]}
do
path="/media/ray/Ray/GoogleDrive/Avlab/program/Self-Attention-GAN-Experiment/generate/${i}/fold0"
python test.py --data_dir ${path} || exit
done
