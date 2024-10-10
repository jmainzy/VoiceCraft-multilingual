# Prepare data for training
# conda activate voicecraft

# Download the encodec from https://github.com/descriptinc/descript-audio-codec (?)


$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"

# phonemize the dataset
python phonemize_encodec_encode_hf.py `
    --dataset mydataset `
    --save_dir phonemes `
    --encodec_model_path downloads/encodec_4cb2048_giga.th `
    --mega_batch_size 120 `
    --batch_size 32 `
    --max_len 30000

# combine phoneme vocabularies
python combine_phonemes.py `
    --english '.\phonemes\vocab-en.txt' `
    --other '.\phonemes\vocab-u.txt' `
    --output '.\phonemes\vocab-out.txt'

# move output vocabulary to dataset
mv phonemes\vocab-out.txt mydataset\vocab.txt