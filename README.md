# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using: 

> python preprocess.py

dataset can be chosen using the **--dataset** argument. Default is **Ljspeech**.

Example liepa-multi:

> python preprocess.py --dataset='liepa-multi' --voice='D245'

This should take no longer than a **few minutes.**

# Training:
To **train both models** sequentially (one after the other):

> python train.py --model='Tacotron-2'

or:

> python train.py --model='Both'

Feature prediction model can **separately** be **trained** using:

> python train.py --model='Tacotron'

checkpoints will be made each **250 steps** and stored under **logs-Tacotron folder.**

Naturally, **training the wavenet separately** is done by:

> python train.py --model='WaveNet'

logs will be stored inside **logs-Wavenet**.

**Note:**
- If model argument is not provided, training will default to Tacotron-2 model training. (both models)
- Please refer to train arguments under [train.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/train.py) for a set of options you can use.

# Synthesis
To **synthesize audio** in an **End-to-End** (text to audio) manner (both models at work):

> python synthesize.py --model='Tacotron-2'

For the spectrogram prediction network (separately), there are **three types** of mel spectrograms synthesis:

- **Evaluation** (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.

> python synthesize.py --model='Tacotron' --mode='eval'

- **Natural synthesis** (let the model make predictions alone by feeding last decoder output to the next time step).

> python synthesize.py --model='Tacotron' --GTA=False


- **Ground Truth Aligned synthesis** (DEFAULT: the model is assisted by true labels in a teacher forcing manner). This synthesis method is used when predicting mel spectrograms used to train the wavenet vocoder. (yields better results as stated in the paper)

> python synthesize.py --model='Tacotron' --GTA=True

Synthesizing the **waveforms** conditionned on previously synthesized Mel-spectrograms (separately) can be done with:

> python synthesize.py --model='WaveNet'
