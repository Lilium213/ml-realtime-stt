import torch
from transformers import TrainingArguments, Trainer, AutoProcessor, AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import librosa


from datasets import Dataset

#
def get_speech_array_for_file(file_path):
    speech_array, sampling_rate = librosa.load(file_path, sr=None, mono=True)
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
    return speech_array
#
#
def transcribe_audio(file_path):
    # Load audio file (48 kHz or other sample rate)
    speech_array = get_speech_array_for_file(file_path)

    # Preprocess the audio for the model
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription
# Load audio files and transcriptions
def load_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            audio_path, transcription = line.strip().split('|')
            data.append({"audio": audio_path, "transcription": transcription})
    return Dataset.from_list(data)

# Define function to load audio and prepare it for Wav2Vec2
def speech_file_to_array_fn(batch):
    speech_array, _ = sf.read(batch["audio"])
    batch["speech"] = speech_array
    return batch

# Prepare dataset with tokenized inputs
def prepare_dataset(batch):
    # Load audio file (48 kHz or other sample rate)
    speech_array = get_speech_array_for_file(batch["audio"])
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
    # Tokenize transcription (labels)
    transcription = batch["transcription"]
    labels = processor(text=transcription, return_tensors="pt").input_ids

    print(f"Labels shape: {labels.shape}")

    batch["input_values"] = input_values.squeeze()
    batch["labels"] = labels.squeeze()
    return batch


# Load the pre-trained Wav2Vec2 model and processor
# model = AutoModelForCTC.from_pretrained("my_wav2vec2_model")
# processor = AutoProcessor.from_pretrained("my_wav2vec2_model")
processor = Wav2Vec2Processor.from_pretrained("my_wav2vec2_model")

model = Wav2Vec2ForCTC.from_pretrained("my_wav2vec2_model")

print("Transcribed text: " + transcribe_audio("training_data_wavs/sentence_1.wav"))



