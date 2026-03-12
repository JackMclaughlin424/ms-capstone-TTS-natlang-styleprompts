# Style prompt generator from multi-modal,multi-scale dialogue contexts for instruction-guided text-to-speech

This paper proposes a model for use in a conversational text-to-speech (TTS) framework that improves user control and usability by automatically generating natural-language style prompts from dialogue context. Motivated by the limitations of open-ended prompt interfaces in existing instruction-guided TTS systems, the approach leverages multimodal conversational history (speech and text) alongside a user-provided script to produce contextually appropriate and speaker-consistent style descriptions. The model is trained in a supervised manner using the richly annotated conversational dataset ParaSpeechCaps, enabling it to capture the flow of expressive speech in natural dialogue. Evaluation combines objective semantic similarity metrics with subjective user studies to assess prompt quality, interpretability, and alignment with synthesized speech. The results aim to demonstrate that guided style prompt generation are a viable solution for user-friendly instruction-guided text-to-speech systems.

## What's in this repo

* Jupyter notebooks for the exploratory data analysis I performed to generate the charts seen in my paper.
* Python framework for running model-training experiments
* Python script for retrieving the datasets used for training
* Python script for transforming the data

## Setup

### 1. Datasets

Create the dataset preprocessing and EDA conda environment using the environment file ```src/dataset_preprocessing_environment.yml``` :

```conda env create -f 'src/dataset_preprocessing_environment.yml'```

#### Downloading data

This project utilizes multiple datasets:

1. ParaSpeechCaps
    a. This dataset annotates a few pre-existing audio datasets, EARS, Expresso, Emilia, and VoxCeleb. This project only uses the conversational Expresso portions. This is around 20 hours of expressive, conversational speech.
2. StyleTalk
    a. This is speech data with minimal dialogue history and weaker annotations than ParaSpeechCaps

Run the following to download the dataset(s):

```python
# download all datasets
src/get_data.py 

# download specific
src/get_data.py --dataset paraspeechcaps

```

#### Preprocessing

1. Preprocessing ParaSpeechCaps:Expresso

    Run the script ```src/data_helpers/preprocess_expresso.py``` to complete the preprocessing steps for Expresso outlined in the original paper's README

2. Preprocessing StyleTalk

    Run the script ```src/data_helpers/preprocess_styletalk.py``` to reformat the dataset to align with ParaSpeechCaps

3. Merge annotations

    Run the script ```src/data_helpers/build_merged_annotation_dataset.py``` to combine the annotations for StyleTalk and ParaSpeechCaps into one dataset.

4. Build HDF5 audio dataset

    Run the script ```src/data_helpers/build_h5py_dataset.py``` to output one data file containing all training data used for the project, and one metadata file containing annotations and other metadata.

    Example run:
    ```python
    python .\build_h5py_dataset.py --audio_root_PSC '../data_TEMP/paraspeechcaps/audio/expresso' --audio_root_ST '../data_TEMP/styletalk/audio' --out_h5 '..\data_TEMP\merged_audio.h5' --out_meta '..\data_TEMP\merged_metadata.parquet' --DEBUG_MAX_ROW 500 --DEBUG_MAX_TURN 5
    ```

### 2. Model

#### Test Model (WIP)

The Jupyter notebook ```src/style-prompt-generator/model.ipynb``` contains the model code and progressive sanity checks.
