import argparse
import glob
import pandas as pd
from datasets import Audio, Dataset
import process_text as prep
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--dataset", type=str, default='dataset', help='name of input data')
    parser.add_argument('--save_dir', type=str, default="/data/scratch/pyp/datasets/gigaspeech_phn_enc_manifest_debug", help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--encodec_model_path', type=str, default="/data/scratch/pyp/exp_pyp/audiocraft/encodec/xps/6f79c6a8/checkpoint.th")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument('--mega_batch_size', type=int, default=100, help="Number of samples in each mega batch for multiprocess dataloading")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=35.0, help='will drop audios that are longer than this number')
    parser.add_argument('--max_len', type=int, default=30000, help='max length of audio in samples, if exceed, will cut a batch into half to process, decrease this number if OOM on your machine')
    return parser.parse_args()


class mydataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = dataset[split]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, ind):
        try:
            segment_id, audio, duration, text= \
                    self.data[ind]['segment_id'], \
                    torch.from_numpy(self.data[ind]['audio']['array']).float(), \
                    self.data[ind]['duration'], \
                    self.data[ind]['text']
        except:
            return None, None, None, None
        
        return segment_id, audio, duration, text
    def collate(self, batch):
        res = {'segment_id': [], "audio": [], "duration": [], "text": [], }
        for item in batch:
            if item[0] != None:
                res['segment_id'].append(item[0])
                res['audio'].append(item[1])
                res['duration'].append(item[2])
                res['text'].append(item[3])
        return res


def load_custom_data(dir):
    '''
    Make an audio dataset dict from a folder
    '''
    metadata={"audio":[], "duration":[], "segment_id":[], "text":[],"text_latin":[], "file":[]}
    if dir[-1] != '/':
        dir = dir+'/'
    for file in glob.glob(dir+"*.tsv"):
        print('reading file '+str(file))
        data = pd.read_csv(file, sep='\t')
        for index, row in data.iterrows():
            metadata["audio"].append(dir+row['filename'])
            metadata["duration"].append(float(row['duration']))
            metadata["segment_id"].append(row['filename'].split('.')[-2])
            metadata["text"].append(row['text'])
            metadata["text_latin"].append(row['latin'])
            metadata["file"].append(file)
    
    return Dataset.from_dict(metadata).cast_column("audio", Audio(sampling_rate=16_000))

if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    import numpy as np
    import torch
    import tqdm
    import time
    from datasets import load_dataset, DownloadConfig

    from tokenizer import TextTokenizer, tokenize_text
    
    # get the path
    phn_save_root = os.path.join(args.save_dir, "phonemes")
    codes_save_root = os.path.join(args.save_dir, "encodec_16khz_4codebooks")
    vocab_fn = os.path.join(args.save_dir, "vocab.txt")
    os.makedirs(phn_save_root, exist_ok=True)
    os.makedirs(codes_save_root, exist_ok=True)

    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEFAULT_DEVICE = "mps"
    else:
        DEFAULT_DEVICE = "cpu"

    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        logging.info(f"longest: {lens[inds[-1]]*args.model_code_sr} encodec codes, {lens[inds[-1]]:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]*args.model_code_sr} encodec codes, {lens[inds[0]]:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]*args.model_code_sr} encodec codes, {lens[inds[len(inds)//2]]:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]*args.model_code_sr} encodec codes, {lens[inds[int(len(inds)*0.95)]]:.2f} sec.")
        return inds[::-1]
    
    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1])))
    

    ### phonemization
    # load tokenizer
    # load the encodec model
    from audiocraft.solvers import CompressionSolver
    model = CompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.to(DEFAULT_DEVICE)
    model = model.eval()
    text_tokenizer = TextTokenizer()


    # https://github.com/SpeechColab/GigaSpeech
    # there are only four different punctuations
    # need to check whether there are other < started strings
    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    # dc = DownloadConfig(cache_dir=args.download_to)
    stime = time.time()
    logging.info("loading the dataset...")
    # gs = load_dataset("speechcolab/gigaspeech", args.dataset_size, use_auth_token=True, cache_dir = args.download_to, download_config=dc)
    dataset = load_custom_data(args.dataset)
    dataset = dataset.map(prep.remove_special_chars)
    logging.info(f"time spend on loading the dataset: {time.time() - stime:.2f} seconds")

    splits = ['test', 'train']
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
            
    logging.info(f"dataset info: {dataset}")
    logging.info(f"phonemizing...")
    phn_vocab = set()
    all_lens = []
    
    # you will see a ton of [WARNING] words_mismatch.py:88......, it's not a issue
    for split in tqdm.tqdm(splits):
        skip = 0
        logging.info(f"now processing split {split}...")
        for item in tqdm.tqdm(dataset[split]):
            save_fn = os.path.join(phn_save_root, item['segment_id']+".txt")
            print(item)
            text = item['text']
            if sum(word in forbidden_words for word in text.split(" ")):
                logging.info(f"skip {item['segment_id']}, because it contains forbiden words. It's transcript: {text}")
                skip += 1
                continue
            for k, v in punc2sym.items():
                text = text.replace(k, v)
            phn = tokenize_text(text_tokenizer, text)
            phn_seq = " ".join(phn)
            for k, v in word2sym.items():
                phn_seq = phn_seq.replace(k, v)
            phn_vocab.update(phn_seq.split(" "))
            all_lens.append(len(phn_seq.split(" ")))
            with open(save_fn, "w") as f:
                f.write(phn_seq)
        logging.info(f"split {split} has {len(dataset[split])} samples in total, skipped {skip} due to forbiden words")

    print(f"phn vocab size: {len(list(phn_vocab))}")
    print("phn sequence stats: ")
    print(f"longest: {max(all_lens)}")
    print(f"shortest: {min(all_lens)}")
    print(f"median: {np.quantile(all_lens, 0.5)}")
    print(f"95 percentile longest: {np.quantile(all_lens, 0.95)}")
    print("write vocabulary to ", vocab_fn)
    with open(vocab_fn, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")

    ## encodec codes extraction
    logging.info("encodec encoding...")
    train_dataset = mydataset('train')
    train_loader = torch.torch.utils.data.DataLoader(train_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=train_dataset.collate)
    # validation_dataset = mydataset('validation')
    # validation_loader = torch.torch.utils.data.DataLoader(validation_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=validation_dataset.collate)
    test_dataset = mydataset('test')
    test_loader = torch.torch.utils.data.DataLoader(test_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=test_dataset.collate)
    splits = ['test', 'train']
    loaders = [test_loader, train_loader]
    # splits = ['validation'] # for debug
    # loaders = [validation_loader]
    for split, loader in zip(splits, loaders):
        skip = 0
        logging.info(f"now processing split {split}...")
        mega_n_steps = int(np.ceil(len(dataset[split]) / args.mega_batch_size))
        logging.info(f"partition the split {split} into {mega_n_steps} parts, each has {args.mega_batch_size} samples")
        for m, mega_batch in enumerate(loader):
            logging.info(f"====================================")
            logging.info(f"====================================")
            logging.info(f"now processing mega step {m+1}/{mega_n_steps}")
            lengths = np.array(mega_batch['duration'])
            sorted_inds = sort_by_audio_len(lengths)
            for j in range(len(sorted_inds))[::-1]:
                if lengths[sorted_inds[j]] < 0.2 or lengths[sorted_inds[j]] > args.len_cap: # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                    skip += 1
                    del sorted_inds[j]
            
            n_steps = int(np.ceil(len(sorted_inds) / args.batch_size))
            for n in tqdm.tqdm(range(n_steps), disable=True):
                inds_used = sorted_inds[n*args.batch_size:(n+1)*args.batch_size]
                audio_batch = [mega_batch['audio'][id] for id in inds_used]
                segment_id_batch = [mega_batch['segment_id'][id] for id in inds_used]
                text_batch = [mega_batch['text'][id] for id in inds_used]
                padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
                all_lens = [lengths[id] for id in inds_used]
                with torch.no_grad():
                    if max(all_lens) > args.max_len and len(all_lens) > 1: # NOTE decrease args.max_len if OOM, or chunk it into more than 2 forward passes
                        codes = []
                        inwav = padded_wav.cuda()
                        codes.append(model.encode(inwav[:len(inwav)//2])[0].cpu())
                        codes.append(model.encode(inwav[len(inwav)//2:])[0].cpu())
                        codes = torch.cat(codes, dim=0)
                    else:
                        encoded_frames = model.encode(padded_wav.cuda())
                        # logging.info(f"encoded_frames: {encoded_frames[0].shape}")
                        codes = encoded_frames[0].cpu()

                for i, length in enumerate(all_lens):
                    save_fn = os.path.join(codes_save_root, segment_id_batch[i]+".txt")
                    actual_len = round(length * args.model_code_sr) # 320 is downsample rate for this model
                    cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                    write_array_to_txt_file(cur_code, save_fn)
