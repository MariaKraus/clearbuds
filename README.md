Installation

GitHubRepo: https://github.com/vivjay30/clearbuds

conda create -n clearbuds python=3.8

conda activate clearbuds

export PYTHONPATH=$PYTHONPATH:pwd`` #add your own path

(z. B.: export PYTHONPATH=$PYTHONPATH:"/home/maria/clearbuds")

navigate to the clearbuds directory: cd clearbuds

install the requirements: pip install -r requirements.txt

Unzip: unzip 2voices_synthetic_test.zip

cd into clearbuds_waveform/src/

Run:

CUDA\_VISIBLE\_DEVICES=0 python evaluate\_cascaded.py \\ --model-path checkpoints/clearvoice\_iphone\_causal\_mixed\_l1spec\_loss\_large/final\_37epochs.pth.tar \\ --data-dir ../../test \\ --n-mics 2 \\ --n-speakers 2 \\ --sample-rate 15625 \\ --chunk-size 46850 --unet-checkpoint unet.pt

run with --use-cuda 0 flag, the GPU version just doesn’t work!!!!

Run own example
resampled audio data to 16khz
put our own audio files in /home/maria/clearbuds/clearbuds_waveform/src/real_examples/
conda activate clearbuds_env2
export PYTHONPATH=$PYTHONPATH:"/home/maria/clearbuds”
run ./cascaded_cpu.sh 20210909_16.30.53