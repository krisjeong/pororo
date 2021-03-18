# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import BoolTensor, Tensor


def get_mask_from_lengths(inputs: Tensor, seq_lengths: Tensor) -> Tensor:
    mask = BoolTensor(inputs.size()).fill_(False)                   # mask = Boolean tensor of size input

    for idx, length in enumerate(seq_lengths):                      #
        length = length.item()

        if (mask[idx].size(0) - length) > 0:                        #
            mask[idx].narrow(
                dim=0,
                start=length,
                length=mask[idx].size(0) - length,
            ).fill_(True)

    return mask


def collate_fn(batch: list, batch_size: int) -> dict:               # merges a list of samples to form a mini-batch of Tensor(s) (returns tensor 'inputs' and int tensor 'input_lengths')

    def seq_length_(p):                                                 # do you have to write out a separate fn 'seq_length'? (syntax q)
        return len(p)

    input_lengths = torch.IntTensor([len(s) for s in batch])            # input_lengths = tensor of lengths for each item in batch
    max_seq_length = max(batch, key=seq_length_).size(0)                # find the longest sequence length in batch
    inputs = torch.zeros(batch_size, max_seq_length)                    # tensor of 0s, shape: (batch_size, max_seq_length)?

    for idx in range(batch_size):                                       # for each elem in batch:
        sample = batch[idx]                                                 # sample = elem
        input_length = sample.size(0)                                       # input_length = elem size
        inputs[idx].narrow(dim=0, start=0, length=input_length).copy_(sample)   # make 0 tensor's dims fit actual elem

    return {"inputs": inputs, "input_lengths": input_lengths}
