import re
from datasets import load_dataset
import pandas as pd


def initialize_expresso_df():
    
    # Load specific splits of the dataset

    train_base = load_dataset("ajd12342/paraspeechcaps", split="train_base")
    holdout = load_dataset("ajd12342/paraspeechcaps", split="holdout")
    dev = load_dataset("ajd12342/paraspeechcaps", split="dev")
    test = load_dataset("ajd12342/paraspeechcaps", split="test")

    dfs = [
        train_base.to_pandas(),
        holdout.to_pandas(),
        dev.to_pandas(),
        test.to_pandas(),
    ]

    # merge into one dataframe
    df = pd.concat(dfs, ignore_index=True)
    

    # Only want conversational, already segmented files
    df = df[df["source"] == "expresso"]
    
    
    df = df[df["relative_audio_path"].str.contains("conversational_vad_segmented")]


    return df 


def find_missing_annotations(df):
    # check for files that are missing from conversations, skipping style folders that arent represented at all
    import os
    
    valid_styles = set(df['high_lvl_style'].unique())

    wav_files = []
    base_dir = "data/raw/paraspeechcaps/audio/expresso/audio_48khz/conversational_vad_segmented"

    for root, dirs, files in os.walk(base_dir):
        # check if this directory is under a valid style folder
        style_match = re.search(r'conversational_vad_segmented/[^/]+/([\w-]+)', root.replace('\\', '/'))
        if style_match and style_match.group(1) not in valid_styles:
            continue
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    prefix = "data/raw/paraspeechcaps/audio/expresso/"
    wav_files_rel = [f.replace('\\', '/').removeprefix(prefix) for f in wav_files]

    existing = set(df['relative_audio_path'].values)
    missing = [f for f in wav_files_rel if f not in existing]
    
    print(f"Total wav files found: {len(wav_files_rel)}")
    print(f"Missing from dataframe: {len(missing)}")
    
    return missing


def parse_start_time(filename):
    match = re.search(r'_segment_(\d+\.\d+)_', filename)
    return float(match.group(1)) if match else float('inf')

def parse_end_time(filename):
    match = re.search(r'_segment_\d+\.\d+_(\d+\.\d+)\.wav', filename)
    return float(match.group(1)) if match else float('inf')


def add_conversation_index(df_in):
    df = df_in.copy()
    
    # parse conversation info
    df['speakers'] = df['relative_audio_path'].str.extract(r'(ex\d+-ex\d+)')
    df['high_lvl_style'] = df['relative_audio_path'].str.extract(r'ex\d+-ex\d+/([\w-]+)/')
    df['conv_num'] = df['relative_audio_path'].str.extract(r'_(\d+)_channel')
    df['start_time'] = df['relative_audio_path'].apply(parse_start_time)
    df['end_time'] = df['relative_audio_path'].apply(parse_end_time)
    
    # Build conv_id from structured columns
    df['_conv_id'] = df['speakers'] + '_' + df['high_lvl_style'] + '_' + df['conv_num']

    df = df.sort_values(['_conv_id', 'start_time']).reset_index(drop=True)

    df['turn_index'] = df.groupby('_conv_id').cumcount()
    df['prev_filename'] = df.groupby('_conv_id')['relative_audio_path'].shift(1)
    
    # look for files that are missing annotaitons
    print("Looking for filenames that don't have annotations, theses could cause interruptions in conversations.")
    missing = find_missing_annotations(df)
    
    
    # separate conversations at the missing file, to keep conversations contiguous
    print("Splitting up conversations that have gaps.")
    df_split = df.copy()
    df_split['_is_split_point'] = False

    for f in missing:
        match = re.search(r'(ex\d+-ex\d+)_([\w-]+)_(\d+)_channel', f)
        if not match:
            continue
        conv_id = f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        start = parse_start_time(f)
        if start is not None:
            # only mark the first row at or after the missing file's start time
            idx = df_split[(df_split['_conv_id'] == conv_id) & (df_split['start_time'] >= start)].index
            if len(idx) > 0:
                df_split.loc[idx[0], '_is_split_point'] = True

    df_split['_conv_id_split'] = df_split['_conv_id'] + '_part' + df_split.groupby('_conv_id')['_is_split_point'].transform(lambda x: x.cumsum().astype(str))

    df_split['turn_index'] = df_split.groupby('_conv_id_split').cumcount()
    df_split['prev_filename'] = df_split.groupby('_conv_id_split')['relative_audio_path'].shift(1)
    
    print(f"Original conversations: {df_split['_conv_id'].nunique()}")
    print(f"Split conversations: {df_split['_conv_id_split'].nunique()}")
    
    return df_split


def main():
    df = initialize_expresso_df()
    
    df = add_conversation_index(df)
    
    df.to_csv("psc_annotations_expresso_conversation_CLEANED.csv", index=False)
    
    
if __name__=="__main__":
    main()