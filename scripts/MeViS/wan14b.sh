#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate imap
cd IMAP

model_id="14b"
input_json="json/MeViS_504/1.json"

sep_score="CHI"
imap_qk_matching_target="prompt"

output_dir="./result"
mkdir -p "$output_dir"

python main_Renoising.py \
    --model_id "$model_id" \
    --input_json "$input_json" \
    --renoise_timestep 20 \
    --test_full_denoise "false" \
    --output_dir "$output_dir" \
    --width 832 \
    --maps_except_softmax "true" \
    --maps_timesteps 0 50 \
    --maps_layers 0 30 \
    --imap_layer 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --imap_sep_score "$sep_score" \
    --imap_sep_topk 5 \
    --imap_qk_matching_target "$imap_qk_matching_target"

# python main_Sampling.py \
#     --model_id "$model_id" \
#     --input_json "$input_json" \
#     --output_dir "$output_dir" \
#     --width 832 \
#     --maps_except_softmax "true" \
#     --maps_timesteps 0 50 \
#     --maps_layers 0 30 \
#     --imap_layer 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#     --imap_sep_score "$sep_score" \
#     --imap_sep_topk 5 \
#     --imap_qk_matching_target "$imap_qk_matching_target"
