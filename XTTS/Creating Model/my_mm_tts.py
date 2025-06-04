from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.configs.vits_config import VitsConfig

class MyMultilingualTTS(Vits):
    def __init__(self, config: VitsConfig):
        super().__init__(config)
        self.language_ids = {lang: idx for idx, lang in enumerate(config.data["languages"])}
    
    def forward(self, text, text_len, audio, audio_len, language, *args, **kwargs):
        lang_id = self.language_ids[language]
        return super().forward(text, text_len, audio, audio_len, language_ids=lang_id, *args, **kwargs)