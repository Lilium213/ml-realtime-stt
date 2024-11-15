# Attempt at making local real time speech to text

## Step 1: Model choice: small.en

Using CUDA. CPU is way too slow. 0.8s latency with _small.en_. Not worth testing the rest.

The models _medium.en_ and _turbo_ seem to have the same times. Going to _large_ increases latency too much.

_small.en_ is faster than _medium.en_ and _turbo_ and seems to perform just as well without any fine-tuning.

Going with ___small.en___ for now. Might change to _medium.en_ or _turbo_ if performance drops too much.


## Step 2: Figure out real time transcription
Got "real time" working. Listens to and transcribes 1-second chunks of audio.

Need to update to use the whisper library itself to figure out if the audio in the buffer ends on a \n or something like that

## Step 3: Fine-tuning

