# conda activate voicecraft

$dataset='mydataset'
$exp_root="../experiments/custom01"
$exp_name='e830M'
$dataset_dir="../data/mydataset"
$encodec_codes_folder_name="encodec_16khz_4codebooks"
$load_model_from="../pretrained_models/giga830M.pth"

torchrun --nnodes=1 `
../main.py `
--load_model_from $load_model_from `
--reduced_eog 1 `
--drop_long 1 `
--eos 2051 `
--n_special 4 `
--pad_x 0 `
--codebook_weight "[3,1,1,1]" `
--encodec_sr 50 `
--num_epochs 4 `
--lr 0.00001 `
--warmup_fraction 0.1 `
--optimizer_name "AdamW" `
--d_model 2048 `
--audio_embedding_dim 2048 `
--nhead 16 `
--num_decoder_layers 16 `
--max_num_tokens 20000 `
--gradient_accumulation_steps 32 `
--val_max_num_tokens 6000 `
--num_buckets 6 `
--audio_max_length 20 `
--audio_min_length 2 `
--text_max_length 400 `
--text_min_length 10 `
--mask_len_min 1 `
--mask_len_max 600 `
--tb_write_every_n_steps 10 `
--print_every_n_steps 400 `
--val_every_n_steps 1600 `
--text_vocab_size 100 `
--text_pad_token 100 `
--phn_folder_name "phonemes" `
--manifest_name "manifest" `
--encodec_folder_name $encodec_codes_folder_name `
--audio_vocab_size 2048 `
--empty_token 2048 `
--eog 2049 `
--audio_pad_token 2050 `
--n_codebooks 4 `
--max_n_spans 3 `
--shuffle_mask_embedding 0 `
--mask_sample_dist poisson1 `
--max_mask_portion 0.9 `
--min_gap 5 `
--num_workers 8 `
--dynamic_batching 0 `
--dataset $dataset `
--exp_dir $exp_root/$dataset/$exp_name `
--dataset_dir $dataset_dir
# #  >> ./logs/${dataset}/${exp_name}.log 2>&1