import argparse
import glob
import os

import numpy as np
import soundfile as sf
import librosa
import tqdm
from clearbuds_waveform import evaluate_recorded
from clearbuds_spectrogram import inference_causal


def main(dir_src, left_channel, right_channel, output_dir, sample_rate, chunk_duration, use_cuda, cutoff):
    files = sorted(glob.glob(os.path.join(dir_src, '**', '*.wav'), recursive=True))
    print('Found {} label files in {}'.format(len(files), dir_src))

    # load audio files and labels
    for file in tqdm.tqdm(files, "Processing files"):

        directories = file.split(os.path.sep)
        sep = "/"
        # create output directory if it does not exist
        output_dir_file = os.path.expanduser('~') + output_dir + sep + sep.join(directories[5:-1])
        if not os.path.exists(output_dir_file):
            os.makedirs(output_dir_file)
        output_file = output_dir_file + sep + directories[-1]
        print("\n The file will be saved in: ", output_file)
        # Get the total duration of the audio file
        total_duration = librosa.get_duration(filename=file)
        num_chunks = int(np.ceil(total_duration / chunk_duration))
        processed_chunks = []

        for i in tqdm.tqdm(range(num_chunks), "Processing chunks"):
            offset = i * chunk_duration
            audio, sample_rate = librosa.load(file, sr=sample_rate, mono=False, offset=offset, duration=chunk_duration)
            data_l = audio[left_channel]
            data_r = audio[right_channel]
            eval_output = evaluate_recorded.evaluate(os.path.expanduser('~') + "/clearbuds_small/models/final.pth.tar",
                                                     data_l, data_r, use_cuda)
            output = inference_causal.infer(model_path=os.path.expanduser('~') + "/clearbuds_small/models/model.pt",
                                            eval_output=eval_output, audio_l=data_l, audio_r=data_r,
                                            sample_rate=sample_rate, use_cuda=use_cuda, cutoff=cutoff)
            # Append the processed chunk to the list
            processed_chunks.append(output)

        # Combine the processed chunks into one audio file
        combined_output = np.concatenate(processed_chunks)
        # Save the combined audio file
        sf.write(output_file, combined_output, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Use clearbuds algorithm to remove noise from audio files')
    parser.add_argument("--dir_data", type=str, default=os.path.expanduser('~') + "/clearbuds_small/data/",
                        help="Path to the dataset")
    parser.add_argument("--left_channel", type=int, default=(11 - 1))
    parser.add_argument("--right_channel", type=int, default=(9 - 1))
    parser.add_argument("--output_dir", type=str, help="Path to save model",
                        default=os.path.expanduser('~') + "/clearbuds_small/results")
    parser.add_argument('--sample-rate', default=15625, type=int,
                        help='Sample rate')
    parser.add_argument('--use-cuda', type=int, default=1,
                        help='Whether use GPU')
    parser.add_argument("--cutoff", type=float, default=.003)
    parser.add_argument("--chunk_duration", type=int, default=30)

    args = parser.parse_args()
    main(args.dir_data, args.left_channel, args.right_channel, args.output_dir, args.sample_rate, args.chunk_duration,
         args.use_cuda, args.cutoff)
