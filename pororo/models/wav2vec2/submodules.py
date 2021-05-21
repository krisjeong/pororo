# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import contextlib
import itertools as it

import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder
from fairseq.models.wav2vec.wav2vec2_asr import (
    Linear,
    Wav2VecCtc,
    base_architecture,
)
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from wav2letter.decoder import CriterionType


class BrainWav2VecEncoder(FairseqEncoder):                                      # build wav2vec model w/ pretrained args?
    """ Modified from https://github.com/pytorch/fairseq """

    def __init__(self, args, tgt_dict=None, pretrain_args=None):                # run when BrainWav2VecEncoder is initialized (as w2v_encoder when BrainWav2VecCtc is initialized in submodules.py: 126)
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        w2v_args = pretrain_args                                               # w2v_args = pretrain_args
        assert (args.normalize == w2v_args.normalize
               ), "Fine-tuning works best when data normalization is the same"

        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

        w2v_args.data = args.data
        task = AudioPretrainingTask.setup_task(w2v_args)        # set up AudioPretrainingTask;  # w2v_args.arch='wav2vec2'
        model = task.build_model(w2v_args)                      # Build the :class:`~fairseq.models.BaseFairseqModel` instance for task 'AudioPretrainingTask'     # build model (AudioPretrainingTask.build_model) by using info: w2v_args.arch='wav2vec2'
                                                                # model (checked structure and type via log) = Wav2Vec2Model
        model.remove_pretraining_modules()
        super().__init__(task.source_dictionary)                # task.source_dictionary: None

        d = w2v_args.encoder_embed_dim                          # d = 1024

        self.w2v_model = model                                  # w2v_model = Wav2Vec2Model (l53)

        self.final_dropout = nn.Dropout(args.final_dropout)     # set up dropout (to prevent overfitting)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))                # changes this to vocab size so that we can run argmax (create learnable variables (out_features, in_features) to fit length of tgt_dict)
        elif getattr(args, "decoder_embed_dim", d) != d:        # self.proj.weight.shape: torch.Size([108, 1024])
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):            # Gets run automatically when BrainWav2VecEncoder class called, b/c nn.Module works like that
        w2v_args = {
            "source": source,                                                   # 'source' = signal tensor; 'padding_mask' = same-sized tensor as 'source' but filled w/ False; 'mask' = bool
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates                   # ft: False, self.freeze_finetune_updates: 10000, self.num_updates: 0

        with torch.no_grad() if not ft else contextlib.ExitStack():             # disables gradient calculations (since inferring, no need for backtracking)
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)       # w2v_model (Wav2Vec2Model (fairseq.models.wav2vec.wav2vec2) -> extract_features

            if tbc:                                                             # default: True
                # B x T x C -> T x B x C                                        # TODO: check if B = Batch size, T = length of output representation from encoder (timesteps?), C = input size (num of tokens)
                x = x.transpose(0, 1)                                           # x before: torch.Size([1, 95, 1024]), after: [95, 1, 1024], padding_mask.shape: torch.Size([1, 95])

        x = self.final_dropout(x)                                               # x.shape: [95, 1, 1024]

        if self.proj:                                                           # if applicable, change size to vocab size (len(tgt_dict)) so that we can run argmax
            x = self.proj(x)                                                    # after projection, x.shape: [95, 1, 108], padding_mask.shape: [1, 95]

        return {
            "encoder_out": x,  # T x B x C                                      # [95, 1, 108]
            "encoder_padding_mask": padding_mask,  # B x 2T                     # B x 2T        # ?
            "padding_mask": padding_mask,                                       # 'padding_mask' = same-sized tensor as 'source' but filled w/ False
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class BrainWav2VecCtc(Wav2VecCtc):                                                  # BrainWav2VecCtc.forward -> uses the instance 'forward' from superclass Wav2VecCtc, which returns result of running through w2v_encoder!
    """ Modified from https://github.com/pytorch/fairseq """

    @classmethod
    def build_model(cls, args, target_dict, pretrain_args):                         # returns new instance of class # args: w2v["args"], target_dict: target_dict, pretrain_args: w2v["pretrain_args"]
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = BrainWav2VecEncoder(args, target_dict, pretrain_args)         # w2v_encoder = BrainWav2VecEncoder(args, target_dict, pretrain_args)
        return cls(w2v_encoder, args)                                               # constructs + returns a BrainWav2VecCtc model


class W2lDecoder(object):

    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (tgt_dict.index("<ctc_blank>")
                      if "<ctc_blank>" in tgt_dict.indices else tgt_dict.bos())
        self.asg_transitions = None

    def generate(self, models, sample, **unused):           # == self.model of recognizer.py 170)
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {       # if multiple sections: encoder_input: dict {'padding_mask': tensor([[False, False, False,  ..., False, False, False]], device='cuda:0'), 'source': tensor([[ 6.4412e-05, -1.0509e-04,  6.4412e-05,  ..., -1.2988e-02, -1.2818e-02, -1.3835e-02]], device='cuda:0')}, source.shape: torch.Size([1, 30720]), padding_mask.shape: torch.Size([1, 30720])     # if in one piece: encoder_input: dict {'source': tensor([[ 0.0001, -0.0002,  0.0001,  ..., -0.0038, -0.0035, -0.0045]], device='cuda:0') of shape [1, 211883] (signal), 'padding_mask': tensor([[False, False, False,  ..., False, False, False]], device='cuda:0') of shape [1, 211883]}
            k: v
            for k, v in sample["net_input"].items()
            if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)       # 'emissions': normalized output produced by encoder; pass tensors through encoder  # emissions (encoder output): tensor; shape: [1, 95, 108]
        return self.decode(emissions)                               # now send to decoder 'W2lViterbiDecoder'.decode -> return [[{"tokens": tensor([ 8, 11, 14, 11, 10,  5,  8, 48, 10, 32,  6, 37,  7, 11, 10,  5, 32, 12, 26, 22,  6, 18, 27,  8, 13,  5]), "score": 0}]]

    def get_emissions(self, models, encoder_input):             # models: a list just containing BrainWav2VecCtc model; encoder_input: dict {'padding_mask': tensor, 'source': tensor (both sized same (e.g. [1, 30720]))
        """Run encoder and normalize emissions"""               # models[0]: BrainWav2VecCtc; when calling on model, forward fn automatically gets run
        encoder_out = models[0](**encoder_input)                # BrainWav2VecCtc(**encoder_input) # **encoder_input: unpacks and passes in 'source' and 'padding_mask' tensors into dict 'encoder_input'    # 'encoder_out': result of running BrainWav2VecCtc on {'source': tensor([[ 0.0001, -0.0002,  0.0001,  ..., -0.0038, -0.0035, -0.0045]], device='cuda:0'), 'padding_mask': tensor([[False, False, False,  ..., False, False, False]], device='cuda:0')}
        # encoder_out = dict {'encoder_out': tensor [95, 1, 108], 'padding_mask'/'encoder_padding_mask': BoolTensor filled w/ False [1, 95]}

        if self.criterion_type == CriterionType.CTC:            # if True: (set to True earlier in file) -->
            emissions = models[0].get_normalized_probs(             # emissions: normalized version of encoder output encoder_out['encoder_output'] (done by log softmax layer)
                encoder_out,                                        # emissions: torch.cuda.FloatTensor; shape: [95, 1, 108] (same as BrainWav2VecEncoder.forward 'x')
                log_probs=True,
            )
        # emissions (normalized (0<x<1) version of encoder_out): [95, 1, 108]
        return emissions.transpose(0, 1).float().cpu().contiguous()    # emissions: tensor shaped [95, 1, 108], which is transposed to [1, 95, 108]

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):                                    # Viterbi decoder (Kakao) instead of beam search decoder (wav2vec)   # this decoder prob turns tokens into text?

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):                                        # TODO: figure out decode fn (should be tokens to text)
        batch_size, time_length, num_classes = emissions.size()         # batch_size = B, time_length = T, num_classes = C (num of tokens/input size)?

        if self.asg_transitions is None:                                # default None  # asg_transitions: probabilities of each letter pair in corpus; asg = auto segmentation
            transitions = torch.FloatTensor(                                # transitions: torch.FloatTensor size [108, 108], filled w/ zeros
                num_classes,
                num_classes,
            ).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view( #
                num_classes,
                num_classes,
            )

        viterbi_path = torch.IntTensor(batch_size, time_length)         # size [1, 95]   # stores results that decoder returns?
        workspace = torch.ByteTensor(
            CpuViterbiPath.get_workspace_size(                          # get_workspace_size: allocates contiguous memory space for arrays the Viterbi decoder uses
                batch_size,
                time_length,
                num_classes,
            ))
        CpuViterbiPath.compute(                                         # runs Viterbi algorithm and returns most likely token sequence; pass in tensor pointers to the C++ method that implements Viterbi algorithm
            batch_size,
            time_length,
            num_classes,
            get_data_ptr_as_bytes(emissions),                           # gets the pointers of the tensors we made
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [[{                                                      # for each batch:
            "tokens": self.get_tokens(viterbi_path[b].tolist()),            # "tokens": tensor([ 8, 11, 14, 11, 10,  5,  8, 48, 10, 32,  6, 37,  7, 11, 10,  5, 32, 12, 26, 22,  6, 18, 27,  8, 13,  5]), tokens.size(): torch.Size([26])
            "score": 0                                                      # tokens: normalized tokens from Viterbi algorithm?
        }] for b in range(batch_size)]
