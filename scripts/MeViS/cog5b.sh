#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate imap
cd IMAP

model_id="5b"
input_json="json/MeViS_504/1.json"

sep_score="CHI"
imap_qk_matching_target="concepts"

output_dir="./result"
mkdir -p "$output_dir"

# python main_Renoising.py \
#     --model_id "$model_id" \
#     --input_json "$input_json" \
#     --renoise_timestep 30 \
#     --test_full_denoise "false" \
#     --output_dir "$output_dir" \
#     --maps_except_softmax "true" \
#     --maps_timesteps 0 50 \
#     --maps_layers 0 42 \
#     --imap_layer 1 2 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 \
#     --imap_sep_score "$sep_score" \
#     --imap_sep_topk 5 \
#     --imap_qk_matching_target "$imap_qk_matching_target"

python main_Sampling.py \
    --model_id "$model_id" \
    --input_json "$input_json" \
    --output_dir "$output_dir" \
    --maps_except_softmax "true" \
    --maps_timesteps 0 50 \
    --maps_layers 0 42 \
    --imap_layer 1 2 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 \
    --imap_sep_score "$sep_score" \
    --imap_sep_topk 5 \
    --imap_qk_matching_target "$imap_qk_matching_target"
