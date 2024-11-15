import torch
from transformers import TrainingArguments, Trainer, AutoProcessor, AutoModelForCTC
import soundfile as sf
import librosa
from datasets import Dataset


max_length = 0

# Load and preprocess audio files
def get_speech_array_for_file(file_path):
    speech_array, sampling_rate = librosa.load(file_path, sr=None, mono=True)
    global max_length
    max_length = max(max_length, len(speech_array))
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
    return speech_array

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

# Load data from a text file
def load_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            audio_path, transcription = line.strip().split('|')
            data.append({"audio": audio_path, "transcription": transcription})
    return Dataset.from_list(data)

# Preprocess and load audio for Wav2Vec2 model
def speech_file_to_array_fn(batch):
    speech_array1, sampling_rate = librosa.load(batch["audio"], sr=None, mono=True)
    global max_length
    max_length = max(max_length, len(speech_array1))

    speech_array, _ = sf.read(batch["audio"])
    batch["speech"] = speech_array
    return batch

# Prepare dataset by tokenizing audio and transcription
def prepare_dataset(batch):
    # Load audio file (48 kHz or other sample rate)
    speech_array = get_speech_array_for_file(batch["audio"])

    # Preprocess the audio for Wav2Vec2, ensuring padding and truncation
    print("Max length: " + str(max_length))
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length + 100).input_values

    # Tokenize transcription (labels)
    transcription = batch["transcription"]
    labels = processor(text=transcription, return_tensors="pt", padding="max_length", truncation=True, max_length=512).input_ids

    batch["input_values"] = input_values.squeeze()
    batch["labels"] = labels.squeeze()
    return batch


# Load dataset
dataset = load_data("training-data.txt")
dataset = dataset.map(speech_file_to_array_fn)

# Load the pre-trained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCTC.from_pretrained(model_name)

# Split dataset into training and evaluation sets (80% train, 20% eval)
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()

# Apply preprocessing to both train and eval datasets
train_dataset = train_dataset.map(prepare_dataset, remove_columns=["speech", "transcription"])
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=["speech", "transcription"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_wav2vec2_model",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=500,
)

# Initialize Trainer with evaluation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass evaluation dataset
)

# Start training
trainer.train()

model.save_pretrained("my_wav2vec2_model")
processor.save_pretrained("my_wav2vec2_model")
