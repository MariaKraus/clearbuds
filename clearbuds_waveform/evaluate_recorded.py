import argparse

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from conv_tasnet import ConvTasNet
import soundfile as sf

LOOKAHEAD = 700
PADDING_AMOUNT = 25400 - LOOKAHEAD

LOOKAHEAD = 700
PADDING_AMOUNT = 25400 - LOOKAHEAD


def evaluate(model_path, data_l, data_r, use_cuda):
    # Load model with num channels = 2 (like in the paper)
    model = ConvTasNet.load_model(model_path, input_channels=2)
    model.eval()
    if use_cuda:
        model.cuda()

    min_length = min(data_l.shape[0], data_r.shape[0])
    data_l = data_l[:min_length]
    data_r = data_r[:min_length]

    data = np.stack([data_l, data_r], axis=0)

    mixture = torch.FloatTensor(data).unsqueeze(0)

    if use_cuda:
        mixture = mixture.cuda()

    mixture = F.pad(mixture, (PADDING_AMOUNT, 0))

    estimate_source = model(mixture)  # [B, C, T]
    output = estimate_source[0, 0].detach().cpu().numpy()
    return output
