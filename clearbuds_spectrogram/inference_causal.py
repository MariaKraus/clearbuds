import os
import sys

import cv2
import torch
import numpy as np
import librosa
sys.path.append("/home/maria/clearbuds_small/clearbuds_spectrogram")
sys.path.append("/home/maria/clearbuds_small/clearbuds_spectrogram/UNet")
from clearbuds_spectrogram.UNet import unet
TIME_DIM = 64
PROCESS_SIZE = 2


def save_spectrogram(spectrogram, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(filename, spectrogram)

def log_mel_spec_original(audio_files, sample_rate=None):
    audio = []
    for a in audio_files:
        a = a / ((a.std() + 1e-4) / 0.05)
        audio.append(a)

    min_length = min([x.shape[0] for x in audio])
    y = sum([x[:min_length] for x in audio])
    # sf.write("mixed.wav", y, sample_rate)

    return log_mel_spec(y, sample_rate)


def log_mel_spec(y, sample_rate):
    n_fft = 1024
    hop_length = 400
    n_mels = 128  # 128 is better for the direction part
    fmin = 20
    fmax = sample_rate / 2

    original_spectrogram = librosa.stft(y, n_fft=n_fft)
    power_spectrogram = np.abs(original_spectrogram) ** 2
    S = librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(S)

    return mel_spec_db, original_spectrogram

def apply_mask(mask, spectrogram):
    """
    """
    assert(mask.shape == spectrogram.shape)
    silence = np.ones_like(mask) * -100.0
    output = np.where(mask, spectrogram, silence)
    return output

def generate_input(eval_output, audio_l, audio_r, sample_rate, spectrogram_only=False):
    """
    Generates the spectrograms for the mixed mic array inputs
    """
    mixed_spec, original_spec = log_mel_spec_original([audio_l, audio_r], sample_rate=sample_rate)
    save_spectrogram(mixed_spec, os.path.join("mixed_spec.png"))

    mixed_spec2, original_spec2 = None, None
    if not spectrogram_only:
        # We are doing a cascaded processing where the mask is applied to the time domain output
        mixed_spec2, original_spec2 = log_mel_spec_original([eval_output], sample_rate=sample_rate)
        save_spectrogram(mixed_spec2, os.path.join("mixed_spec2.png"))
    return mixed_spec, original_spec, mixed_spec2, original_spec2



def infer(model_path, eval_output, audio_l, audio_r, sample_rate, use_cuda, cutoff):
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    model = unet().to(device)
    state_dict = torch.load(model_path, map_location=device).state_dict()
    model.load_pretrain(state_dict)

    model.eval()
    mixed_spec, original_spec, mixed_spec2, original_spec2 = generate_input(eval_output, audio_l, audio_r, sample_rate)

    outputs = []

    curr_idx = 0
    mixed_spec_padded = np.pad(mixed_spec, ((0, 0), (TIME_DIM - PROCESS_SIZE, 0)))

    while (curr_idx + TIME_DIM) < mixed_spec_padded.shape[1]:
        curr_input = mixed_spec_padded[:, curr_idx:curr_idx + TIME_DIM]
        curr_input = (curr_input - curr_input.mean()) / (curr_input.std() + 1e-8)

        output = model(torch.FloatTensor(curr_input).unsqueeze(0).unsqueeze(0).to(device))

        mask = output.cpu().detach().numpy()[0, 0, :, -2:]
        outputs.append(mask)
        curr_idx += PROCESS_SIZE

    mask = np.concatenate(outputs, axis=-1)


    cv2.imwrite("mask_causal.png", (mask * 255).astype(np.uint8))

    # Select noisy or conv-tasnet input
    apply_spectrogram_mel = mixed_spec if False else mixed_spec2
    apply_spectrogram_orig = original_spec if False else original_spec2

    total_length = min(mask.shape[-1], apply_spectrogram_orig.shape[-1])
    apply_spectrogram_orig = apply_spectrogram_orig[:, :total_length]
    mask = mask[:, :total_length]

    # Undo the mel transform
    filter_bank = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=128, fmin=20, fmax=sample_rate / 2)

    spec_mask = np.matmul(filter_bank.transpose(), mask) > cutoff

    separated_spec = np.where(spec_mask, 1, 0) * apply_spectrogram_orig
    output = librosa.istft(separated_spec)
    # not sure about this (but the in the original code it is like this)
    output = output * 2
    output_spectrogram, _ = log_mel_spec(output, sample_rate)
    save_spectrogram(output_spectrogram, "output_spec.png")

    return output
