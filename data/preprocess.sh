# Prepare data for training
# conda activate voicecraft

# Download the encodec from https://github.com/descriptinc/descript-audio-codec (?)

python phonemize_encodec_encode_hf.py \
    --dataset dataset \
    --save_dir phonemes \
    --encodec_model_path downloads/encodec_4cb2048_giga.th \
    --mega_batch_size 120 \
    --batch_size 32 \
    --max_len 30000