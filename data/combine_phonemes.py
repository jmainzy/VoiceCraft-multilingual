import pandas as pd

def read_file(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)
    data.columns = ['index', 'phoneme']
    return data

def combine_files(file1_path, file2_path, output_path):
    en_data = read_file(file1_path)
    lang_data = read_file(file2_path)
    
    # merge the language phonemes dataframe with the english phonemes
    # skipping phonemes which are already in english set
    combined_data = pd.concat([en_data, lang_data[~lang_data['phoneme'].isin(en_data['phoneme'])]])
    combined_data = combined_data.reset_index(drop=True)
    # drop the index column
    combined_data = combined_data.drop(columns=['index'])
    print(combined_data)
    
    # write the combined phonemes to a file
    combined_data.to_csv(output_path, sep=' ', header=False, index=True)

if __name__ == "__main__":
    english_path = './phonemes/vocab-en.txt'
    other_lang_path = './phonemes/vocab.txt'
    output_path = './phonemes/vocab-out.txt'
    
    combine_files(english_path, other_lang_path, output_path)