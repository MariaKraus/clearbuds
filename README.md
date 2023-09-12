# Clearbuds example

The repository adapts the example from https://github.com/vivjay30/clearbuds.
It uses the same models and the example use case from 'cascaded.sh'.

### Installation with Anaconda

```
conda create -n clearbud_env python=3.8

conda activate clearbud_env

pip3 install torch torchvision torchaudio

cd clearbuds

pip install -r requirements.txt
```
### Models

- Download the clearbud repo: https://drive.google.com/file/d/1C1m7UjZRNVq84jzx49krJ5tsaaMwalRG/view?usp=sharing 
- Copy the models: final.pth.tar and model.pt to a folder called 'models' in this clearbuds directory. 


### Run

```
python3 main.py --dir_data "/path/to/data_directory" --left_channel 0 , --right_channel 1 
```