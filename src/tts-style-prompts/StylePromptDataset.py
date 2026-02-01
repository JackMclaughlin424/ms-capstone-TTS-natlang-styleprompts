from torch.utils.data import Dataset, DataLoader
import random

class StylePromptDataset(Dataset):
    def __init__(self, audio_df, prompt_df):
        
        self.audio_df = audio_df
        self.prompt_map = self._get_prompt_map(prompt_df)

    def __getitem__(self, idx):
        row = self.audio_df.iloc[idx]

        prompt = random.choice(
            self.prompt_map[row["style_prompt_key"]]
        )

        # TODO
        audio = load_audio(row["audio_path"])

        return {
            "audio": audio,
            "style_prompt": prompt
        }
        
    def _get_prompt_map(self, prompt_df):
        prompt_map = (
            prompt_df.groupby("style_prompt_key")["prompt"]
            .apply(list)
            .to_dict()
        )
        
        return prompt_map
