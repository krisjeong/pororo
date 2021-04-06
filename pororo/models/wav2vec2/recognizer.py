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


class BrainWav2Vec2Recognizer(object):          """ Analyzes signal as array & builds and returns 'result_dict' (containing "audio", "duration", and "results") """
    """ Wav2Vec 2.0 Speech Recognizer """

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
        self.model = self._load_model(model_path, device, self.target_dict)     # Wav2Vec model
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
            sentence (str): naively inferenced sentence from model      ᄀ ᅳ ᄂ ᅳ ᆫ | ᄀ ᅫ ᆫ ᄎ ᅡ ᆭ ᄋ ᅳ ᆫ | ᄎ ᅥ ᆨ ᄒ ᅡ ᄅ ᅧ ᄀ ᅩ | ᄋ ᅢ | ᄊ ᅳ ᄂ ᅳ ᆫ | ᄀ ᅥ | ᄀ ᅡ ᇀ ᄋ ᅡ ᆻ ᄃ ᅡ | ᄀ ᅳ ᄂ ᅣ ᄋ ᅦ | ᄉ ᅡ ᄅ ᅡ ᆼ ᄋ ᅳ ᆯ | ᄋ ᅥ ᆮ ᄀ ᅵ ᄌ ᅵ ᄆ ᅡ ᆫ | ᄒ ᅥ ᆺ ᄉ ᅳ ᄀ ᅩ ᄋ ᅧ ᆻ ᄃ ᅡ |
        Returns:
            str: post-processed, inferenced sentence                    '그는 괜찮은 척하려고 애 쓰는 거 같았다 그냐에 사랑을 얻기 위해 애 썼지만 헛스고였다'}]
        """
        if self.graphemes:
            # grapheme to character
            sentence = unicodedata.normalize("NFC", sentence.replace(" ", ""))      # Return the normal form 'form' for the Unicode string unistr.
            sentence = sentence.replace("|", " ").strip()
            return self._grapheme_filter(sentence)                                  # delete lone graphemes?

        return sentence.replace(" ", "").replace("|", " ").strip()

    def _split_audio(self, signal: np.ndarray, top_db: int = 48) -> list:           # break up audio into a list of non-silent intervals (==splices from 'signal', an ndarray ver of audio)
        speech_intervals = list()
        start, end = 0, 0

        non_silence_indices = librosa.effects.split(signal, top_db=top_db)      # splits audio into non-silent intervals (anything below top_db considered silence) + returns array of tuples (start_index, end_index) of interval i

        for _, end in non_silence_indices:                                          # why use _ here? (syntax Q)
            speech_intervals.append(signal[start:end])                              # to the list, append non-silent segment from signal
            start = end                                                             # move window

        speech_intervals.append(signal[end:])                                       # append very end of signal

        return speech_intervals

    @torch.no_grad()
    def predict(                                                                # called from automatic_speech_recognition.py l217     # builds and returns result_dict (containing “audio”, “duration”, “results”)
        self,
        audio_path: str,
        signal: np.ndarray,                                                         # 'signal' = audio as ndarray
        top_db: int = 48,
        vad: bool = False,                                                          # vad = False first
        batch_size: int = 1,
    ) -> dict:
        result_dict = dict()

        duration = librosa.get_duration(signal, sr=self.SAMPLE_RATE)                # get overall duration (in seconds) from entire audio file
        batch_inference = True if duration > 50.0 else False                        # True if duration > 50s--so can use vad if >50s (if less than 50s, split into dB criteria and speech recognition happens)

        result_dict["audio"] = audio_path                                           # audio path--does NOT go through model
        result_dict["duration"] = str(datetime.timedelta(seconds=duration))         # str(duration of audio found by subtracting)--does NOT go through model
        result_dict["results"] = list()
        """ building up empty "results" list (by building dict 'hypo_dict' and appending to list 'results') """
        if batch_inference:                                                         # if duration > 50s:
            if vad:                                                                     # if vad:
                speech_intervals = self.vad_model(                                          # vad.py __call__ (l177): 'speech_intervals' = model output (of VoiceActivityDetection, which uses ConvVADModel to get the probability of the labels); list of lists of frequencies over interval frames (VoiceActivityDetection; vad.py); since call on object, goes to __call__ (vad.py l178)
                    signal,
                    sample_rate=self.SAMPLE_RATE,
                )
            else:                                                                       # else:
                speech_intervals = self._split_audio(signal, top_db)                        # rule-based: get list of non-silent intervals (==splices from 'signal', an ndarray ver of audio)from audio
            # either way, we get a list 'speech_intervals' of splices (of non-silent intervals) of the original 'signal' ndarray
            batches, total_speech_sections, total_durations = self._create_batches(      # return lists: 'batches' (of 'batch': tensors), 'total_speech_sections' (list of list 'speech_sections (for 1 batch)' of dicts 'speech_section' for 1 interval, shape {"start": START_TIME, "end": END_TIME}), 'total_durations' (of 'duration')
                speech_intervals,
                batch_size,
            )

            for batch_idx, batch in enumerate(batches):                                 # for each batch:       # TODO: ==for each audio sample?
                net_input, sample = dict(), dict()

                net_input["padding_mask"] = get_mask_from_lengths(                          # What's going on?
                    inputs=batch["inputs"],                                                 # net_input["padding_mask"] =
                    seq_lengths=batch["input_lengths"],
                ).to(self.device)
                net_input["source"] = batch["inputs"].to(self.device)                       # ?
                sample["net_input"] = net_input                                             # ?

                # yapf: disable
                if sample["net_input"]["source"].size(1) < self.MINIMUM_INPUT_LENGTH:       # ?
                    continue
                # yapf: enable

                hypos = self.generator.generate(                                            # Generate a batch of inferences (for each batch (sentence?), generate tensor using wav2vec model
                    self.model,                                                             # list of list of dict {'tokens': tensor of ints, 'score: 0}
                    sample,                                                                 # [{'tokens': tensor([ 8, 11, 14, 11, 10,  5,  8, 48, 10, 32,  6, 37,  7, 11, 10,  5, 32, 12, 26, 22,  6, 18, 27,  8, 13,  5,  7, 23,  5, 49, 11, 14, 11, 10,  5,  8, 12,  5,  8,  6, 46,  7,  6, 29, 15,  6,  5,  8, 11, 14, 31,  7, 20,  5, 19,  6, 18,  6, 25,  7, 11, 17,  5,  7, 12, 59,  8,  9,  5,  7, 56, 22, 23,  5,  7, 23,  5, 49, 12, 29, 16,  9, 21,  6, 10,  5, 22, 12, 43, 19,11,  8, 13,  7, 27, 29, 15,  6,  5]), 'score': 0}]

                    prefix_tokens=None,
                )

                for hypo_idx, hypo in enumerate(hypos):                                     # For each inference (i.e. for 1 section):       # what is hypo--is it a word? a letter?
                    hypo_dict = dict()
                    hyp_pieces = self.target_dict.string(                                       # convert tensor of ints to string (letters) using dict --> hyp_pieces: all tokens (letter by letter),
                        hypo[0]["tokens"].int().cpu())                                          # dict = target dict loaded from FB;
                    speech_section = total_speech_sections[batch_idx][hypo_idx]                 # get dict 'speech_section' (for this section); {"start": START_TIME, "end": END_TIME}

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

                    # yapf: disable                                                         # hypo_dict: dict printed out when asr is run (inside dict 'results')
                    hypo_dict["speech_section"] = f"{speech_start_time} ~ {speech_end_time}"    # time stamps for segment
                    hypo_dict["length_ms"] = total_durations[batch_idx][hypo_idx] * 1000        # 'total_durations': list (of 'duration'); getting ith duration
                    hypo_dict["speech"] = self._text_postprocess(hyp_pieces)                    # puts individual letters together to make proper sentence
                    # yapf: enable

                    if hypo_dict["speech"]:                                                     # if the text is not empty:
                        result_dict["results"].append(hypo_dict)                                    # append this dict to overall 'result_dict'

                del hypos, net_input, sample                                                # 'hypos': batch of inferences, 'net_input': ?, 'sample': input?

        else:                                                                           # if duration <= 50s:
            net_input, sample, hypo_dict = dict(), dict(), dict()

            feature, duration = self._parse_audio(signal)                                   # feature (tensor ver of signal) and duration (in sec)

            net_input["source"] = feature.unsqueeze(0).to(self.device)                      # add a dimension of 1 in index 0 to feature        # TODO: figure out math

            padding_mask = torch.BoolTensor(                                                # ?
                net_input["source"].size(1)).fill_(False)
            net_input["padding_mask"] = padding_mask.unsqueeze(0).to(
                self.device)

            sample["net_input"] = net_input                                                 # add dict 'net_input' to dict 'sample'

            hypo = self.generator.generate(                                                 # Generate a batch of inferences using wav2vec model
                self.model,
                sample,
                prefix_tokens=None,
            )
            hyp_pieces = self.target_dict.string(                                           # target_dict = target dict loaded from FB; hyp_pieces?
                hypo[0][0]["tokens"].int().cpu())                                           # hypo[0][0] (Cf. hypo[0])

            speech_start_time = str(datetime.timedelta(seconds=0))                          # start_time set to 0
            speech_end_time = str(
                datetime.timedelta(seconds=int(round(duration, 0))))                        # end_time

            hypo_dict[
                "speech_section"] = f"{speech_start_time} ~ {speech_end_time}"              # fill up 'hypo_dict' (dict inside 'results')
            hypo_dict["length_ms"] = duration * 1000                                        # total_durations[batch_idx][hypo_idx] * 1000
            hypo_dict["speech"] = self._text_postprocess(hyp_pieces)                        # TODO: further examine

            if hypo_dict["speech"]:
                result_dict["results"].append(hypo_dict)

        return result_dict

    def _create_batches(
        self,
        speech_intervals: list,                                         # speech_intervals: list of all non-silent intervals from audio; for 'korean_sample2.wav', it's 2
        batch_size: int = 1,                                            # batch_size default 1
    ) -> Tuple[list, list, list]:
        batches = list()
        total_speech_sections = list()                                  # total_speech_sections: [
        total_durations = list()

        cumulative_duration = 0
        num_batches = math.ceil(len(speech_intervals) / batch_size)     # num_batches = (total num intervals / batch size); for 'korean_sample2.wav', it's 2 (how many of the speech intervals you'll pass through the model at once)

        for batch_idx in range(num_batches):                            # for each batch (==each speech interval in 'speech_intervals'):
            sample = list()                                                 # list of feature tensors (tensor ver of signal for an interval) for each interval
            speech_sections = list()                                        # list of dicts (dict: speech_section (keys: start, end))
            durations = list()                                              # list of duration (in secs)

            for idx in range(batch_size):                                   # for each i in batch: (i.e. just once as default)
                speech_section = dict()                                         # speech_section dict for item {"start": START_TIME, "end: END_TIME}
                speech_intervals_idx = batch_idx * batch_size + idx             # index of this speech interval (#) in set 'speech_intervals'

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
            total_durations.append(durations)                               # add list of duration (in secs; 1 duration for each batch) to list 'total_durations'

        return batches, total_speech_sections, total_durations              # return lists: 'batches' (of 'batch': tensors), 'total_speech_sections' (list of list 'speech_sections' for 1 section of dicts 'speech_section' of time list 'total_speech_sections' of list 'speech_sections (for 1 batch)' of dicts 'speech_section' for 1 interval, shape {"start": START_TIME, "end": END_TIME)), 'total_durations' (of 'duration')
