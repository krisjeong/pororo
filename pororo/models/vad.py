# Copyright (c) Hyeon Kyu Lee and Kakao Brain. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn


def same_padding(kernel):           # kernel = 8
    pad_val = (kernel - 1) / 2      # output will be same size if stride = 1        # 3.5

    if kernel % 2 == 0:
        out = (int(pad_val - 0.5), int(pad_val + 0.5))      # (3,4)
    else:
        out = int(pad_val)

    return out                      # output = padding on 1 side


class VoiceActivityDetection(object):           """returns speech intervals (after removing silence)"""
    """
    Voice activity detection (VAD), also known as speech activity detection or speech detection,
    is the detection of the presence or absence of human speech, used in speech processing.

    Args: model_path, device
        model_path: path of vad model
        device: 'cuda' or 'cpu'
    """

    def __init__(self, model_path: str, device: str):
        import librosa

        self.librosa = librosa              # package for music and audio analysis

        self.sample_rate = 16000            # sampling rate of y (# samples taken per second, from continuous signal to make it discrete/digital)
        self.n_mfcc = 5                     # ? Mel-frequency cepstral coefficients (MFC = represents short-term power spectrum (frequency components) of a sound)
        self.n_mels = 40                    # kind of like frequency?
        self.device = device

        self.model = ConvVADModel()         # model = ConvVADModel (residual network)
        self.model.load_state_dict(torch.load(model_path, map_location=device))     # Loads a model’s parameter dictionary (from the one we downloaded in asr.py l115) using a deserialized state_dict.

        self.model.to(device).eval()        # sets module on evaluation mode

    def extract_features(  # extracts features (mfcc, delta, delta-delta, and rmse) as one transposed array
        self,
        signal,
        size: int = 512,
        step: int = 16,
    ):
        # Mel Frequency Cepstral Coefficents
        mfcc = self.librosa.feature.mfcc(                           # MFCCs = most used feature in speech recognition
            y=signal,                                                    # audio time series (as array)
            sr=self.sample_rate,                                         # sampling rate of y (# samples taken per second, from continuous signal to make it discrete/digital)
            n_mfcc=self.n_mfcc,                                          # number of MFCCs to return
            n_fft=size,                                                  # length of FFT (fast Fourier transforms) window
            hop_length=step,                                             # number of samples between successive frames/adjacent STFT columns
        )
        mfcc_delta = self.librosa.feature.delta(mfcc)                # compute MFCC deltas; local estimate of the derivative of the input data along the selected axis.
        mfcc_delta2 = self.librosa.feature.delta(mfcc, order=2)      # compute MFCC delta-deltas (second derivative)

        # Root Mean Square Energy
        melspectrogram = self.librosa.feature.melspectrogram(       # compute a mel-scaled spectrogram
            y=signal,                                                   # audio time series (as array)
            n_mels=self.n_mels,                                         # number of Mel bands to generate
            sr=self.sample_rate,                                        # sampling rate of y
            n_fft=size,                                                 # length of FFT (fast Fourier transforms) window
            hop_length=step,                                            # number of samples between successive frames/adjacent STFT columns
        )
        rmse = self.librosa.feature.rms(                            # Compute RMS (Root Mean Square) for each frame in mel spectrogram
            S=melspectrogram,
            frame_length=self.n_mels * 2 - 1,
            hop_length=step,
        )

        mfcc = np.asarray(mfcc)                                     # turn everything into array
        mfcc_delta = np.asarray(mfcc_delta)
        mfcc_delta2 = np.asarray(mfcc_delta2)
        rmse = np.asarray(rmse)

        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)  # concatenate all outputs into 1 array
        features = np.transpose(features)                                         # transpose

        return features

    def smooth_predictions_v1(self, label):                         # Smooth 'predict_label' (get rid of any sharp outliers) I think?
        smoothed_label = list()                                     # makes empty list

        # Smooth with 3 consecutive windows
        for i in range(2, len(label), 3):                           # 2 to len(label) w/ step of 3
            cur_pred = label[i]                                     # label passed in: 'predict_label'
            if cur_pred == label[i - 1] == label[i - 2]:            # if three labels in a row same:
                smoothed_label.extend([cur_pred, cur_pred, cur_pred])   # add 3 of that label to list
            else:                                                   # else:
                if len(smoothed_label) > 0:                             # if length of list > 0:
                    smoothed_label.extend([                                 # add last elem 3 more times
                        smoothed_label[-1], smoothed_label[-1],
                        smoothed_label[-1]
                    ])
                else:                                                   # else:
                    smoothed_label.extend([0, 0, 0])                        # add 0 3 more times

        n = 0
        while n < len(smoothed_label):                              # while n < length of list      # Why do this?
            cur_pred = smoothed_label[n]
            if cur_pred == 1:                                       # if smoothed_label[n] == 1:
                if n > 0:                                               # if n is not first elem:
                    smoothed_label[n - 1] = 1                               # prev elem = 1
                if n < len(smoothed_label) - 1:                         # if n is not last elem:
                    smoothed_label[n + 1] = 1                               # next elem = 1
                n += 2
            else:
                n += 1

        for idx in range(len(label) - len(smoothed_label)):
            smoothed_label.append(smoothed_label[-1])               # pad w/ last elem to match size w/ original label

        return smoothed_label

    def smooth_predictions_v2(self, label):                         # Smooth 'predict_label' (get rid of any sharp outliers)
        smoothed_label = list()
        # Smooth with 3 consecutive windows
        for i in range(2, len(label)):                              # instead of going by 3, stride = 1
            cur_pred = label[i]                                     # label passed in: 'predict_label'
            if cur_pred == label[i - 1] == label[i - 2]:            # if 3 in a row same:
                smoothed_label.append(cur_pred)                         # append
            else:                                                   # else:
                if len(smoothed_label) > 0:                             # if length of list > 0:
                    smoothed_label.append(smoothed_label[-1])               # append last elem
                else:                                                   # else:
                    smoothed_label.append(0)                                # append 0

        n = 0
        while n < len(smoothed_label):                              # same as above; why do this?
            cur_pred = smoothed_label[n]
            if cur_pred == 1:
                if n > 0:
                    smoothed_label[n - 1] = 1
                if n < len(smoothed_label) - 1:
                    smoothed_label[n + 1] = 1
                n += 2
            else:
                n += 1

        for _ in range(len(label) - len(smoothed_label)):
            smoothed_label.append(smoothed_label[-1])               # pad w/ last elem to match size w/ original label

        return smoothed_label
    # TODO: must figure out how 'label' is generated
    def get_speech_intervals(self, data, label):                    # data: numpy.ndarray type; length 211883; (mel?) frequency per frame, over 211883 frames; divided by sequence_length (1024) and makes 206 sequences, which are each given a label of 0 (silence) or 1 (voice) by the model?
        """uses labels list to build 'speech_interval' list of interval lists [start_frame, end_frame] that it extracted from 'data'"""
        def get_speech_interval(labels):                            # 'label': list of 0 (silence)s and 1 (voice), length 206; list of labels for each part/sequence; each group of consecutive 1s is a speech_interval
            seguence_length = 1024
            speech_interval = [[0, 0]]                              # list of intervals (start_frame, end_frame)
            pre_label = 0                                           # label before 'label'
            for idx, label in enumerate(labels):                    # for each of the 206 labels:
                # skip past index if label == 0 (so, skips 0s & stops once label = 0)
                if label:                                           # if label != 0:
                    if pre_label == 1:                              # if right before is 1 (i.e. middle of continuous sequence):
                        speech_interval[-1][1] = (idx + 1) * seguence_length    # extend interval by changing end of sequence by 1024
                    else:                                           # if start of sequence:
                        speech_interval.append([                        # add new interval to 'speech_interval' (start_idx, end_idx)
                            idx * seguence_length, (idx + 1) * seguence_length      #
                        ])

                pre_label = label                                   # update pre_label to current label

            return speech_interval[1:]                              # return all the speech intervals exc. the placeholder [0,0]

        speech_intervals = list()                                   # list of arrays (of various sizes)
        interval = get_speech_interval(label)                       # 'interval' = 'speech_interval' list of interval lists [start_frame, end_frame]    #  [[2048, 31744], [41984, 65536], [78848, 122880], [134144, 151552], [164864, 178176], [198656, 210944]]

        for start, end in interval:
            speech_intervals.append(data[start:end])                # using start and end indices from 'speech_interval', index into data (to get frequencies) between them and add them to speech_intervals

        return speech_intervals                                     # 'speech_intervals' = list of arrays (sequences of frequencies from start_frame to end_frame)

    def __call__(self, signal: np.ndarray, sample_rate: int = 16000):       # called when instance (vad_model) called as a fn (recognizer.py l143)
        seguence_signal = list()

        self.sample_rate = sample_rate
        start_pointer = 0
        end_pointer = 1024
                                                                        # Cut up signal using start and end pointers and append to list ‘seguence_signal’
        while end_pointer < len(signal):                                    # while end pointer is smaller than signal length:
            seguence_signal.append(signal[start_pointer:end_pointer])           # append: slice of signal from start to end ptr

            start_pointer = end_pointer                                     # move window by 1024
            end_pointer += 1024

        feature = [self.extract_features(signal) for signal in seguence_signal]  # list of features (mfcc, delta, delta-delta, and rmse) as one transposed array from each section of signal

        feature = np.array(feature)                                         # turn into array
        feature = np.expand_dims(feature, 1)                                # expand array shape + insert axis at 1     # what does it mean to insert an axis in 1? Ask about axes in general?
        x_tensor = torch.from_numpy(feature).float().to(self.device)        # x_tensor = tensor version of array (of the features)

        output = self.model(x_tensor)                                       # ‘Output’ = tensor shaped [206, 2] (206 tuples); result of running model on ‘x_tensor’; shows, for each sequence, the probability that it is label (==index) 0 (silence) vs. label 1 (voice)?
        predicted = torch.max(output.data, 1)[1]                            # 'predicted' = tensor of labels for each interval (with the highest probability); tensor of 206x1 (each row = argmax of the two)

        predict_label = predicted.to(torch.device("cpu")).detach().numpy()  # array version of tensor

        predict_label = self.smooth_predictions_v2(predict_label)           # smoothen predictions in 2 ways (probably just getting rid of outliers?)
        predict_label = self.smooth_predictions_v1(predict_label)           # v2 seems to do the job though?    # TODO: figure out later (not that important)

        return self.get_speech_intervals(signal, predict_label)             # get speech intervals (list of lists of frequencies over interval frames) by using list of labels (0 or 1)


class ResnetBlock(nn.Module):               # one Resnet block

    def __init__(
        self,
        in_channels: int,                   # 1
        out_channels: int,                  # 32
        num_kernels1: tuple,                # (8, 5, 3)
        num_kernels2: int,                  # 16
    ):
        super(ResnetBlock, self).__init__()     # is this trying to inherit from itself?

        padding = same_padding(num_kernels1[0])     # (3,4)
        self.zero_pad = nn.ZeroPad2d((              # 0 0s to left/right, padding[0] (3) 0s to top, padding[1] (4) 0s to bottom
            0,
            0,
            padding[0],
            padding[1],                             # Seems to assume that the first kernel size is always even, i.e. always returns a tuple
        ))
        self.conv1 = nn.Conv2d(
            in_channels,                            # 1
            out_channels,                           # 32
            (num_kernels1[0], num_kernels2),        # Kernel size (8, 16)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)     # batch normalization (rescale + recenter)      # out_channels == num of features (32)
        self.relu1 = nn.ReLU()                      # relu
        self.conv2 = nn.Conv2d(
            out_channels,                           # in_channels = 32
            out_channels,                           # out_channels = 32
            num_kernels1[1],                        # kernel size (5,5)
            padding=same_padding(num_kernels1[1]),  # padding = 2
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            out_channels,                           # in_channels = 32
            out_channels,                           # out_channels = 32
            num_kernels1[2],                        # kernel size (3,3)
            padding=same_padding(num_kernels1[2]),  # padding = 1
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, num_kernels2))     # (1, 32, kernel size (1,16))           # math? --> #TODO: figure out input dimensions
        self.bn_shortcut = nn.BatchNorm2d(out_channels)                             # batch normalization (32)
        self.out_block = nn.ReLU()                  # final relu

    def forward(self, inputs):
        x = self.zero_pad(inputs)                   # 0 0s to left/right, padding[0] (3) 0s to top, padding[1] (4) 0s to bottom
        x = self.conv1(x)                           # pass x through several conv layers to get F(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)                             # F(x) dimensions:

        shortcut = self.shortcut(inputs)            # transform input to the right dimensions to add to F(x)?
        shortcut = self.bn_shortcut(shortcut)
        x = torch.add(x, shortcut)
        out_block = self.out_block(x)               # Send F(x) + x through relu

        return out_block


class ConvVADModel(nn.Module):                      # Residual network (made up of multiple ResnetBlocks; type of CNN): 4 residual blocks of 3 conv layers each, followed by 3 fc layers
                                                    # Output: tensor [num_sequences (total frames/sequence length), 2]: shows, for each sequence, the probability that it is label (==index) 0 (silence) vs. label 1 (voice)
    def __init__(self):
        super(ConvVADModel, self).__init__()

        self.block1 = ResnetBlock(                  # Help w/ math?
            in_channels=1,
            out_channels=32,
            num_kernels1=(8, 5, 3),
            num_kernels2=16,
        )
        self.block2 = ResnetBlock(
            in_channels=32,
            out_channels=64,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )
        self.block3 = ResnetBlock(
            in_channels=64,
            out_channels=128,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )
        self.block4 = ResnetBlock(
            in_channels=128,
            out_channels=128,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )

        self.flat = nn.Flatten()                    # flattens output (output of layers converted into a single feature vector (1D array) for inputting into next layer)

        self.fc1 = nn.Linear(128 * 65, 2048)        # fully connected layers
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 2)               # should make prediction?

    def forward(self, inputs):
        out_block1 = self.block1(inputs)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)

        x = self.flat(out_block4)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output
