import argparse
import os
import soundfile as sf
import librosa
from clearbuds_waveform import evaluate_recorded
from clearbuds_spectrogram import inference


def main(file_path, left_channel, right_channel, output_dir, sample_rate, use_cuda, cutoff):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio = librosa.core.load(file_path, sr=sample_rate, mono=False)[0]
    data_l = audio[left_channel - 1]
    data_r = audio[right_channel - 1]
    eval_output = evaluate_recorded.evaluate("models/final.pth.tar", data_l, data_r, sample_rate, use_cuda)
    output = inference.infer(model_path="models/model.pt", eval_output=eval_output, audio_l=data_l, audio_r=data_r, sample_rate=sample_rate, output_dir=output_dir, use_cuda=use_cuda, cutoff=cutoff)

    sf.write(os.path.join(output_dir, "output.wav"), output, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--left-channel", type=int, default=11)
    parser.add_argument("--right-channel", type=int, default=9)
    parser.add_argument("output_dir", type=str, help="Path to save model")
    parser.add_argument('--sample-rate', default=15625, type=int,
                        help='Sample rate')
    parser.add_argument('--use-cuda', type=int, default=0,
                        help='Whether use GPU')
    parser.add_argument("--cutoff", type=float)

    args = parser.parse_args()
    main(args.file_path, args.left_channel, args.right_channel, args.output_dir, args.sample_rate, args.use_cuda, args.cutoff)