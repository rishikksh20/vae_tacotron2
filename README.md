# VAE Tacotron-2:
Tensorflow Implementation of [Learning latent representations for style control and transfer in end-to-end speech synthesis](https://arxiv.org/pdf/1812.04342.pdf)


# Repository Structure:
	Tacotron-2
	├── datasets
	├── LJSpeech-1.1	(0)
	│   └── wavs
	├── logs-Tacotron	(2)
	│   ├── mel-spectrograms
	│   ├── plots
	│   ├── pretrained
	│   └── wavs
	├── papers
	├── tacotron
	│   ├── models
	│   └── utils
	├── tacotron_output	(3)
	│   ├── eval
	│   ├── gta
	│   ├── logs-eval
	│   │   ├── plots
	│   │   └── wavs
	│   └── natural
	└── training_data	(1)
	    ├── audio
	    └── mels





The previous tree shows what the current state of the repository.

- Step **(0)**: Get your dataset, here I have set the examples of **Ljspeech**.
- Step **(1)**: Preprocess your data. This will give you the **training_data** folder.
- Step **(2)**: Train your Tacotron model. Yields the **logs-Tacotron** folder.
- Step **(3)**: Synthesize/Evaluate the Tacotron model. Gives the **tacotron_output** folder.


# Requirements
first, you need to have python 3.5 installed along with [Tensorflow v1.6](https://www.tensorflow.org/install/).

next you can install the requirements :

> pip install -r requirements.txt

else:

> pip3 install -r requirements.txt

# Dataset:
This repo tested on the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/), which has almost 24 hours of labeled single actress voice recording.

# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using:

> python preprocess.py

or

> python3 preprocess.py

dataset can be chosen using the **--dataset** argument. Default is **Ljspeech**.

# Training:
Feature prediction model can be **trained** using:

> python train.py --model='Tacotron'

or

> python3 train.py --model='Tacotron'

# Synthesis
There are **three types** of mel spectrograms synthesis for the Spectrogram prediction network (Tacotron):

- **Evaluation** (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.

> python synthesize.py --model='Tacotron' --mode='eval' --reference_audio='ref_1.wav'

or

> python3 synthesize.py --model='Tacotron' --mode='eval' --reference_audio='ref_1.wav'

**Note:**
- This implementation not completly tested for all scenarios but training and synthesis with reference audio working.
- Though it only tested on synthesize without GTA and with `eval` mode.
- After training 250k step with 32 batch size on LJSpeech, KL error settled down near to zero (around 0.001) still not get good style transfer and control, may be because this model trained on LJSpeech which is not quite expressive datasets and only have 24 hrs of data, it might be produce good result on expressive dataset like `Blizzard 2013 voice dataset` though author of the paper used 105 hrs of Blizzard Challenge 2013 dataset.
- In my testing, I havn't get good results so far on style transfer side may be some more tweaking required, this implementation easily integrated with `wavenet` as well as `WaveRNN`.
- Feel free to suggest some changes or even better raise PR.

# Pretrained model and Samples:
TODO
Claimed Samples from research paper : http://home.ustc.edu.cn/~zyj008/ICASSP2019

# References and Resources:
- [Tensorflow original tacotron implementation](https://github.com/keithito/tacotron)
- [Original tacotron paper](https://arxiv.org/pdf/1703.10135.pdf)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [r9y9/Tacotron-2](https://github.com/r9y9/Tacotron-2)
- [yanggeng1995/vae_tacotron](https://github.com/yanggeng1995/vae_tacotron)

**Work in progress**
