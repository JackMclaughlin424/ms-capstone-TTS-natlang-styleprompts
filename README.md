## Abstract
Conversational text-to-speech (TTS) systems have shown strong performance in spoken dialogue agents but typically rely on end-to-end models that predict a single latent style vector, limiting user control and interpretability. This project proposes a human-interactive conversational TTS framework that uses a natural language style prompt interface like InstructTTS, allowing users to control how their text is spoken, rather than relying on an opaque internal predictor. Furthermore, the system generates natural language style prompts conditioned on the dialogue history to improve usability and conversational flow. The system aims to recommend style prompts that are both context-relevant and meaningfully diverse, addressing a key challenge in expressive speech synthesis assistive apps. The approach is evaluated on the LibriTTS-P and StyleTalk datasets using embedding-based metrics for accuracy and diversity, with subjective user studies as a potential extension to assess perceived quality and user experience.

## Project
* InstructTTS-like model. TTS conditioned on natural
language style prompt
* Style prompt suggestion model
    * Generates natural language style prompts to use
as input to the final TTS model, conditioned on
dialogue context (audio and textual) and current
utterance text
* Proof-of-concept web application
    * records and transcribes dialogue for model input
    * performs speaker diariazation?
    * user interface
    * inputs for typing current utterance
    * inputs for typing style prompt
    * button for ”generate style suggestions

## Data

This project utilizes multiple datasets:

1. ParaSpeechCaps
    a. This dataset annotates a few pre-existing audio datasets, EARS, Expresso, Emilia, and VoxCeleb. This project only uses the EARS and Expresso portions. Together these datasets cover around 140 hours of speech.
    b. 

Run the following to download the dataset(s):

```python
# download all datasets
src/get_data.py 

# download specific
src/get_data.py --dataset libritts 
src/get_data.py --dataset libritts styletalk
...
```