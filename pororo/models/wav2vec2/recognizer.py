# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import datetime
import math
import unicodedata
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import Dictionary

from pororo.models.vad import VoiceActivityDetection
from pororo.models.wav2vec2.submodules import (
    BrainWav2VecCtc,
    W2lViterbiDecoder,
)
from pororo.models.wav2vec2.utils import collate_fn, get_mask_from_lengths


class BrainWav2Vec2Recognizer(object):
    """ Wav2Vec 2.0 Speech Recognizer """                   # TODO: figure out purpose

    graphemes = {                                           # smallest functional unit of writing system
        "ko": [
            "ᅡ", "ᄋ", "ᄀ", "ᅵ", "ᆫ", "ᅳ", "ᅥ", "ᅩ", "ᄂ", "ᄃ", "ᄌ", "ᆯ", "ᄅ",
            "ᄉ", "ᅦ", "ᄆ", "ᄒ", "ᅢ", "ᅮ", "ᆼ", "ᆨ", "ᅧ", "ᄇ", "ᆻ", "ᆷ", "ᅣ",
            "ᄎ", "ᄁ", "ᅯ", "ᄄ", "ᅪ", "ᆭ", "ᆸ", "ᄐ", "ᅬ", "ᄍ", "ᄑ", "ᆺ", "ᇂ",
            "ᅭ", "ᇀ", "ᄏ", "ᅫ", "ᄊ", "ᆹ", "ᅤ", "ᅨ", "ᆽ", "ᄈ", "ᅲ", "ᅱ", "ᇁ",
            "ᅴ", "ᆮ", "ᆩ", "ᆾ", "ᆶ", "ᆰ", "ᆲ", "ᅰ", "ᆱ", "ᆬ", "ᆿ", "ᆴ", "ᆪ", "ᆵ"
        ],
        "en": None,
        "zh": None,
    }

    def __init__(
        self,
        model_path: str,
        dict_path: str,
        device: str,
        lang: str = "en",
        vad_model: VoiceActivityDetection = None,
    ) -> None:
        self.SAMPLE_RATE = 16000
        self.MINIMUM_INPUT_LENGTH = 1024

        self.target_dict = Dictionary.load(dict_path)

        self.lang = lang
        self.graphemes = BrainWav2Vec2Recognizer.graphemes[lang]        # None if en or zh
        self.device = device

        self.collate_fn = collate_fn                                    # merges a list of samples to form a mini-batch of Tensor(s) (returns tensor 'inputs' and int tensor 'input_lengths')
        self.model = self._load_model(model_path, device, self.target_dict)
        self.generator = W2lViterbiDecoder(self.target_dict)
        self.vad_model = vad_model

    def _load_model(self, model_path: str, device: str, target_dict) -> list:
        w2v = torch.load(model_path, map_location=device)               # w2v = load model (pre-trained?) as object
        model = BrainWav2VecCtc.build_model(                            # build wav2vec model
            w2v["args"],                                                    # w/ info from pretrained model?
            target_dict,
            w2v["pretrain_args"],
        )
        model.load_state_dict(w2v["model"], strict=True)                # copies parameters and buffers from w2v["model"]
        model.eval().to(self.device)                                    # sets model in evaluation mode
        return [model]                                                  # return model as a list?

    @torch.no_grad()
    def _audio_postprocess(self, feats: torch.FloatTensor) -> torch.FloatTensor:        # TODO: examine later
        if feats.dim == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        return F.layer_norm(feats, feats.shape)

    def _parse_audio(                                                       # determining syntactical structure by analyzing words based on grammar
        self,
        signal: np.ndarray,
    ) -> Tuple[torch.FloatTensor, float]:
        duration = round(librosa.get_duration(signal, sr=self.SAMPLE_RATE), 2)  # Compute the duration (in seconds) of an audio time series, feature matrix, or filename.
        feature = torch.from_numpy(signal).float().to(self.device)              # turns signal into tensor
        feature = self._audio_postprocess(feature)                              # postprocess feature (signal)
        return feature, duration                                                # return feature (tensor ver of signal) and duration (in sec)

    def _grapheme_filter(self, sentence: str) -> str:                       # get rid of any lone graphemes (that aren't actual words)  # check?
        new_sentence = str()
        for item in sentence:                                                   # for Korean, item is one syllable
            if item not in self.graphemes:
                new_sentence += item                                            # add to new_sentence if item is not a grapheme
        return new_sentence

    def _text_postprocess(self, sentence: str) -> str:                      # clean up text?
        """
        Postprocess model output
        Args:
            sentence (str): naively inferenced sentence from model
        Returns:
            str: post-processed, inferenced sentence
        """
        if self.graphemes:
            # grapheme to character
            sentence = unicodedata.normalize("NFC", sentence.replace(" ", ""))      # Return the normal form 'form' for the Unicode string unistr.
            sentence = sentence.replace("|", " ").strip()
            return self._grapheme_filter(sentence)                                  # delete lone graphemes?

        return sentence.replace(" ", "").replace("|", " ").strip()

    def _split_audio(self, signal: np.ndarray, top_db: int = 48) -> list:           # break up audio into a list of non-silent intervals
        speech_intervals = list()
        start, end = 0, 0

        non_silence_indices = librosa.effects.split(signal, top_db=top_db)      # splits audio into non-silent intervals (anything below top_db considered silence) + returns array of tuples (start_index, end_index) of interval i

        for _, end in non_silence_indices:                                          # why use _ here? (syntax Q)
            speech_intervals.append(signal[start:end])                              # to the list, append non-silent segment from signal
            start = end                                                             # move window

        speech_intervals.append(signal[end:])                                       # append very end of signal

        return speech_intervals

    @torch.no_grad()
    def predict(                                                                # called from automatic_speech_recognition.py l217?
        self,
        audio_path: str,
        signal: np.ndarray,
        top_db: int = 48,
        vad: bool = False,                                                          # vad = False first
        batch_size: int = 1,
    ) -> dict:
        result_dict = dict()

        duration = librosa.get_duration(signal, sr=self.SAMPLE_RATE)                # get duration
        batch_inference = True if duration > 50.0 else False                        # True if duration > 50s--connection to using vad if >50s? (if less than 50s, split into dB criteria and speech recognition happens)

        result_dict["audio"] = audio_path
        result_dict["duration"] = str(datetime.timedelta(seconds=duration))         # str(duration of audio found by subtracting)
        result_dict["results"] = list()

        if batch_inference:                                                         # if duration > 50s:
            if vad:                                                                     # if vad:
                speech_intervals = self.vad_model(                                          # 'speech_intervals' = model output (VoiceActivityDetection; vad.py); since call on object, goes to __call__ (vad.py l178)
                    signal,
                    sample_rate=self.SAMPLE_RATE,
                )
            else:                                                                       # else:
                speech_intervals = self._split_audio(signal, top_db)                        # get list of non-silent intervals from audio

            batches, total_speech_sections, total_durations = self._create_batches(     # return lists: 'batches' (of 'batch': tensors), 'total_speech_sections' (of dicts 'speech_section' of time (keys: start, end)), 'total_durations' (of 'duration')
                speech_intervals,
                batch_size,
            )

            for batch_idx, batch in enumerate(batches):                                 # for each batch:
                net_input, sample = dict(), dict()

                net_input["padding_mask"] = get_mask_from_lengths(                          # What's going on?
                    inputs=batch["inputs"],
                    seq_lengths=batch["input_lengths"],
                ).to(self.device)
                net_input["source"] = batch["inputs"].to(self.device)                       # ?
                sample["net_input"] = net_input                                             # ?

                # yapf: disable
                if sample["net_input"]["source"].size(1) < self.MINIMUM_INPUT_LENGTH:
                    continue
                # yapf: enable

                hypos = self.generator.generate(                                            # Generate a batch of inferences
                    self.model,
                    sample,
                    prefix_tokens=None,
                )

                for hypo_idx, hypo in enumerate(hypos):                                     # For each inference:       # what is hypo--is it a word? a letter?
                    hypo_dict = dict()
                    hyp_pieces = self.target_dict.string(                                       # dict = target dict loaded from FB; hyp_pieces: all possible tokens?
                        hypo[0]["tokens"].int().cpu())
                    speech_section = total_speech_sections[batch_idx][hypo_idx]                 # get dict 'speech_section' of time (keys: start, end) for this section

                    speech_start_time = str(                                                    # get rounded str ver of start time (e.g. 0:00:00)
                        datetime.timedelta(
                            seconds=int(round(
                                speech_section["start"],
                                0,
                            ))))
                    speech_end_time = str(                                                      # get rounded str ver of end time
                        datetime.timedelta(
                            seconds=int(round(
                                speech_section["end"],
                                0,
                            ))))

                    # yapf: disable                                                         # hypo_dict: what's printed out when asr is run (inside dict 'results')
                    hypo_dict["speech_section"] = f"{speech_start_time} ~ {speech_end_time}"    # time stamps for segment
                    hypo_dict["length_ms"] = total_durations[batch_idx][hypo_idx] * 1000        # 'total_durations': list (of 'duration')
                    hypo_dict["speech"] = self._text_postprocess(hyp_pieces)                    # clean up text from dict?
                    # yapf: enable

                    if hypo_dict["speech"]:                                                     # if the text is not empty:
                        result_dict["results"].append(hypo_dict)                                    # append this dict to overall 'result_dict'

                del hypos, net_input, sample                                                # 'hypos': batch of inferences, 'net_input': ?, 'sample': input?

        else:                                                                           # if duration <= 50s:
            net_input, sample, hypo_dict = dict(), dict(), dict()

            feature, duration = self._parse_audio(signal)                                   # feature (tensor ver of signal) and duration (in sec)

            net_input["source"] = feature.unsqueeze(0).to(self.device)                      # add a dimension of 1 in index 0

            padding_mask = torch.BoolTensor(
                net_input["source"].size(1)).fill_(False)
            net_input["padding_mask"] = padding_mask.unsqueeze(0).to(
                self.device)

            sample["net_input"] = net_input

            hypo = self.generator.generate(
                self.model,
                sample,
                prefix_tokens=None,
            )
            hyp_pieces = self.target_dict.string(
                hypo[0][0]["tokens"].int().cpu())

            speech_start_time = str(datetime.timedelta(seconds=0))
            speech_end_time = str(
                datetime.timedelta(seconds=int(round(duration, 0))))

            hypo_dict[
                "speech_section"] = f"{speech_start_time} ~ {speech_end_time}"
            hypo_dict["length_ms"] = duration * 1000
            hypo_dict["speech"] = self._text_postprocess(hyp_pieces)

            if hypo_dict["speech"]:
                result_dict["results"].append(hypo_dict)

        return result_dict

    def _create_batches(
        self,
        speech_intervals: list,                                         # speech_intervals: list of all non-silent intervals from audio
        batch_size: int = 1,
    ) -> Tuple[list, list, list]:
        batches = list()
        total_speech_sections = list()
        total_durations = list()

        cumulative_duration = 0
        num_batches = math.ceil(len(speech_intervals) / batch_size)     # num_batches = (total tokens / bath size)

        for batch_idx in range(num_batches):                            # for each batch:
            sample = list()                                                 # list of features (tensor ver of signal for an interval)
            speech_sections = list()                                        # list of dicts (dict: speech_section (keys: start, end))
            durations = list()                                              # list of duration (in secs)

            for idx in range(batch_size):                                   # for each item in batch:
                speech_section = dict()                                         # speech_section dict (holds TIME for start and end)
                speech_intervals_idx = batch_idx * batch_size + idx             # index of this speech interval (#)

                if len(speech_intervals) > speech_intervals_idx:                # if there still are tokens left:
                    feature, duration = self._parse_audio(                          # feature (tensor ver of signal (specific interval)) and duration (in sec)
                        speech_intervals[speech_intervals_idx])

                    speech_section["start"] = cumulative_duration                   # set start to end of cum_dur
                    cumulative_duration += duration                                 # add dur to cum_dur
                    speech_section["end"] = cumulative_duration                     # move end  to new end of cum_dur

                    speech_sections.append(speech_section)                          # add this dict 'speech_section' to overall list 'speech_sections'
                    sample.append(feature)                                          # add feature tensor (that interval of the signal) to 'sample' list
                    durations.append(duration)                                      # append duration to list of durations
                else:                                                           # else (this is the last token):
                    speech_section["start"] = cumulative_duration                   # start and end both set to cum_dur
                    speech_section["end"] = cumulative_duration

                    speech_sections.append(speech_section)                          # add this dict 'speech_section' to overall list 'speech_sections'
                    sample.append(torch.zeros(10))                                  # append vector[10] = {0} to 'sample' list
                    durations.append(0)                                             # add 0 to durations list

            batch = self.collate_fn(sample, batch_size)                     # merges a list of samples to form a mini-batch of Tensor(s) (returns tensor 'inputs' and int tensor 'input_lengths')

            batches.append(batch)                                           # add batch to list 'batches'
            total_speech_sections.append(speech_sections)                   # add overall list 'speech_sections' (for one batch) to list 'total_speech_sections'
            total_durations.append(durations)                               # add list of duration (in secs; for one batch) to list 'total_durations'

        return batches, total_speech_sections, total_durations              # return lists: 'batches' (of 'batch': tensors), 'total_speech_sections' (of dicts 'speech_section' of time (keys: start, end)), 'total_durations' (of 'duration')
