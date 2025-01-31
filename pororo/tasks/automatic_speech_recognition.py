"""Automatic Speech Recognition related modeling class"""

import logging
import os
from typing import Optional

import numpy as np

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoAsrFactory(PororoFactoryBase):
    """
    Recognized speech sentence using trained model.
    Currently English, Korean and Chinese supports.

    English (`wav2vec.en`)

        - dataset: LibriSpeech
        - metric: WER (clean: 1.9 / other: 4.3)

    Korean (`wav2vec.ko`)

        - dataset: KsponSpeech
        - metric: CER (clean: 4.9 / other: 5.4)

    Chinese (`wav2vec.zh`)

        - dataset: AISHELL-1
        - metric: CER (6.9)

    Args:
        audio_path (str): audio path for asr (Supports WAV, FLAC, MP3, and PCM format)
        top_db (int): the threshold (in decibels) below reference to consider as silence
        vad (bool): flag indication whether to use voice activity detection or not, If it is False, it is split into
             dB criteria and then speech recognition is made. Applies only when audio length is more than 50 seconds.
        batch_size (int): inference batch size

    Returns:
        dict: result of speech recognition          # audio name, duration, results (dict): speech time section, length in ms, text

    Examples:
        >>> asr = Pororo(task='asr', lang='ko')
        >>> asr('korean_speech.wav')
        {
            'audio': 'example.wav',
            'duration': '0:00:03.297250',
            'results': [
                {
                    'speech_section': '0:00:00 ~ 0:00:03',
                    'length_ms': 3300.0,
                     speech': '이 책은 살 만한 가치가 없어'
                }
            ]
        }
        >>> asr = Pororo(task='asr', lang='en')
        >>> asr('english_speech.wav')
        {
            'audio': 'english_speech.flac',
            'duration': '0:00:12.195000',
            'results': [
                {
                    'speech_section': '0:00:00 ~ 0:00:12',
                    'length_ms': 12200.0,
                    'speech': 'WELL TOO IF HE LIKE LOVE WOULD FILCH OUR HOARD WITH PLEASURE TO OURSELVES SLUICING
                               OUR VEIN AND VIGOUR TO PERPETUATE THE STRAIN OF LIFE BY SPILTH OF LIFE WITHIN US STORED'
                }
            ]
        }

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)                             # inherit from PororoFactoryBase

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["wav2vec.en"],
            "ko": ["wav2vec.ko"],
            "zh": ["wav2vec.zh"],
        }

    def load(self, device: str):                    # Load & instantiate user-selected task-specific model (vad_model, PororoASR)
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.lang not in self.get_available_langs():      # raise error if unsupported language
            raise ValueError(
                f"Unsupported Language : {self.config.lang}",
                'Support Languages : ["ko", "en", "zh"]',
            )
        from pororo.models.wav2vec2.recognizer import BrainWav2Vec2Recognizer       # Wav2Vec 2.0 Speech Recognizer

        model_path = download_or_load(                      # Download or load model based on model information (to be passed into instances of vad_model and model) (/home/kris/.pororo/misc/wav2vec.ko.pt)
            f"misc/{self.config.n_model}.pt",
            self.config.lang,
        )
        dict_path = download_or_load(                       # Download or load dict based on dict path (/home/kris/.pororo/misc/ko.ltr.txt)
            f"misc/{self.config.lang}.ltr.txt",
            self.config.lang,
        )
        vad_model_path = download_or_load(                  # download or load voice activity detection model ( /home/kris/.pororo/misc/vad.pt)
            "misc/vad.pt",
            lang="multi",
        )

        try:
            import librosa  # noqa
            logging.getLogger("librosa").setLevel(logging.WARN)
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install librosa with: `pip install librosa`")

        from pororo.models.vad import VoiceActivityDetection

        vad_model = VoiceActivityDetection(                 # instantiate vad_model
            model_path=vad_model_path,
            device=device,
        )

        model = BrainWav2Vec2Recognizer(                    # instantiate model (BrainWav2Vec2Recognizer)
            model_path=model_path,
            dict_path=dict_path,
            vad_model=vad_model,
            device=device,
            lang=self.config.lang,
        )
        return PororoASR(model, self.config)                # instantiate object of PororoASR


class PororoASR(PororoSimpleBase):                          """preprocesses audio + calls on model (Wav2Vec2Recognizer) to predict (conduct speech recognition for audio in a given path and return dict result+dict)"""

    def __init__(self, model, config):                      # model = BrainWav2Vec2Recognizer
        super().__init__(config)
        self._model = model
        self.SAMPLE_RATE = 16000
        self.MAX_VALUE = 32767

    def _preprocess_audio(self, audio_path: str):           # loads audio file and returns 'signal'/MAX_VALUE
        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")
        try:
            # Using the pydub because the speed of the resample is the fastest.
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("Please install pydub: pip install pydub")

        audio_extension = audio_path.split('.')[-1].lower()             # "wav"
        assert audio_extension in (                                     # raise error if not one of these
            'wav', 'mp3', 'flac',
            'pcm'), f"Unsupported format: {audio_extension}"

        if audio_extension == 'pcm':
            signal = np.memmap(
                audio_path,
                dtype='h',
                mode='r',
            ).astype('float32')

        else:
            sample_rate = librosa.get_samplerate(audio_path)            # sample_rate: number of samples (==frames) per second (or per other unit) taken from a continuous signal to make a discrete or digital signal.
            signal = AudioSegment.from_file(                            # 'signal' = audio file (from_file opens and loads audio file); <pydub.audio_segment.AudioSegment object at 0x7f5e34cf70f0>
                audio_path,
                format=audio_extension,
                frame_rate=sample_rate,                                 # sample_rate: 44100 -> 'frame_rate': 44100
            )

            if sample_rate != self.SAMPLE_RATE:
                signal = signal.set_frame_rate(frame_rate=self.SAMPLE_RATE)     # fix frame rate to match SAMPLE_RATE (16000) -> 'signal.frame_rate' = 16000 (going from 44100 frames/samples per sec to 16000)

            channel_sounds = signal.split_to_mono()                             # splits stereo (which has 2 speakers) into 2 individual channels (AudioSegment files w/ individual speakers)
            signal = np.array(                                                  # 'signal' converted to np.array of samples [211883]
                [s.get_array_of_samples() for s in channel_sounds])[0]          # for s (each channel): get_array_of_samples (returns raw audio data as array of numeric samples)

        return signal / self.MAX_VALUE                                          # returns (signal as np.array) / MAX_VALUE that can be made by bits (to normalize from 0-1?); signal.shape: (211883,)

    def predict(                                                                # Called when asr('filepath') made bc 'predict' is called when an object of PororoSimpleBase is called as a fn (__call__)
        self,
        audio_path: str,
        **kwargs,                                                               # kwargs (dict) is empty in our case! bc always called as asr(audio_path) and nothing else
    ) -> dict:
        """
        Conduct speech recognition for audio in a given path

        Args:
            audio_path (str): the wav file path
            top_db (int): the threshold (in decibels) below reference to consider as silence (default: 48)
            batch_size (int): inference batch size (default: 1)
            vad (bool): flag indication whether to use voice activity detection or not, If it is False, it is split into
             dB criteria and then speech recognition is made. Applies only when audio length is more than 50 seconds.       # ?

        Returns:
            dict: result of speech recognition

        """
        top_db = kwargs.get("top_db", 48)                                       # returns value of 'top_db' (threshold); returns 48 if doesn't exist in dict
        batch_size = kwargs.get("batch_size", 1)                                # returns 1
        vad = kwargs.get("batch_size", False)                                   # returns False in our case        # so if batch_size exists, vad = batch_size value that is input TODO: Why? --> figure out how this number is used

        signal = self._preprocess_audio(audio_path)                             # preprocess audio; 'signal' = ndarray form of audio

        return self._model.predict(                                             # BrainWav2Vec2Recognizer.predict(); returns result_dict (containing “audio”, “duration”, “results”)
            audio_path=audio_path,
            signal=signal,
            top_db=top_db,
            vad=vad,
            batch_size=batch_size,
        )
