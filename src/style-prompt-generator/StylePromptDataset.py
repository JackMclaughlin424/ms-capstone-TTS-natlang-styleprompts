from torch.utils.data import Dataset, DataLoader
import random

class StylePromptDatasetDynamicPrompt(Dataset):
    def __init__(self, audio_df, prompt_df):
        """
        Randomly choose a prompt during __get_item__ from an audio file's options.
        """
        self.audio_df = audio_df
        self.prompt_map = self._get_prompt_map(prompt_df)

    def __getitem__(self, idx):
        """Retrieve sample with dynamic prompt selection"""
        row = self.audio_df.iloc[idx]

        prompt = random.choice(
            self.prompt_map[row["style_prompt_key"]]
        )

        # TODO: actually make this
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
    
    
    
class StylePromptDatasetExploded(Dataset):
    def __init__(self, audio_df, prompt_df):
        """
        Dataset class for audio<->style prompt data. Explodes the data to make 
        a df with one propmt per audio
        """
        self.audio_df = audio_df
        self.prompt_df = prompt_df
        # check if prompts is already split
        def is_list_column(series):
            return series.dropna().apply(lambda x: isinstance(x, list)).all()
        if not is_list_column(self.prompt_df["prompts"]):
            self.prompt_df["prompts"] = self.prompt_df["prompts"].str.split(";")

        self.exploded_audio_prompt_df = self._explode_df(audio_df, prompt_df)
        

    def __getitem__(self, idx):
        """Retrieve sample with dynamic prompt selection"""
        row = self.exploded_audio_prompt_df.iloc[idx]

        prompt = row["style_prompt_key"]

        # TODO: actually make this
        audio = load_audio(row["audio_path"])

        return {
            "audio": audio,
            "style_prompt": prompt
        }
        

    def _explode_df(self, audio_df, prompt_df):
        
        # explode for one propmt per row
        prompt_df = prompt_df.assign(prompt=prompt_df["prompts"]).explode("prompt")
        prompt_df["prompt"] = prompt_df["prompt"].str.strip()
        prompt_df = prompt_df.drop(columns="prompts")
        
        
        merged = audio_df.merge(
            prompt_df,
            on="style_prompt_key",
            how="inner"
        )
        merged