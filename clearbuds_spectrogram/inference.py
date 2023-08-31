import os
import torch
import numpy as np
import librosa
from clearbuds_spectrogram.UNet import unet


def save_spectrogram(spectrogram, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(filename, spectrogram)


def log_mel_spec_original(audio_files, sample_rate):
    """
    Generates the spectrograms for the mixed mic array inputs
    """
    audio = []
    for a in audio_files:
        a = a / ((a.std() + 1e-4) / 0.05)
        audio.append(a)

    # add two audio files together
    y = sum(audio)
    # sf.write("mixed.wav", y, sample_rate)
    n_fft = 1024
    hop_length = 256
    n_mels = 128  # 128 is better for the direction part
    fmin = 20
    fmax = sample_rate / 2

    original_spectrogram = librosa.stft(y, n_fft=n_fft,
                                        hop_length=hop_length)
    power_spectrogram = np.abs(original_spectrogram) ** 2
    S = librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(S)

    return mel_spec_db, original_spectrogram


def generate_input(eval_output, audio_l, audio_r, sample_rate):
    """
    Generates the spectrograms for the mixed mic array inputs
    """
    mixed_spec, original_spec = log_mel_spec_original([audio_l], sample_rate=sample_rate)
    # save_spectrogram(mixed_spec, os.path.join("mixed_spec.png"))
    mixed_spec2, original_spec = log_mel_spec_original([eval_output], sample_rate=sample_rate)
    # save_spectrogram(mixed_spec2, os.path.join("mixed_spec2.png"))
    return mixed_spec, original_spec


def infer(model_path, eval_output, audio_l, audio_r, sample_rate, use_cuda, cutoff):
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    model = unet().to(device)
    state_dict = torch.load(model_path).state_dict()
    model.load_pretrain(state_dict)

    model.eval()
    mixed_spec, original_spec = generate_input(eval_output, audio_l, audio_r, sample_rate)

    cropped_dim = (mixed_spec.shape[1] // 32 - 2) * 32
    mixed_spec = mixed_spec[:, :cropped_dim]
    original_spec = original_spec[:, :cropped_dim]

    mixed_spec = (mixed_spec - mixed_spec.mean()) / (mixed_spec.std() + 1e-8)

    output = model(torch.FloatTensor(mixed_spec).unsqueeze(0).unsqueeze(0).to(device))

    mask = output.cpu().detach().numpy()[0, 0]

    save_spectrogram(mask, os.path.join("mask.png"))
    # Undo the mel transform
    filter_bank = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=128, fmin=20, fmax=sample_rate / 2)
    spec_mask = np.matmul(filter_bank.transpose(), mask) > 0.003

    separated_spec = original_spec * spec_mask
    output = librosa.istft(separated_spec, hop_length=256)
    return output
