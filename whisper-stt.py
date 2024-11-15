import whisper
import os
import torch
import time


data = {
    "training_data_wavs/sentence_1.wav": "Hello, how are you doing today?",
    "training_data_wavs/sentence_2.wav": "I hope you’re having a great day!",
    "training_data_wavs/sentence_3.wav": "Let’s go for a walk in the park.",
    "training_data_wavs/sentence_4.wav": "I will be there in a few minutes.",
    "training_data_wavs/sentence_5.wav": "What time does the meeting start tomorrow?",
    "training_data_wavs/sentence_6.wav": "Please pass me the salt and pepper.",
    "training_data_wavs/sentence_7.wav": "I can’t believe it’s already October!",
    "training_data_wavs/sentence_8.wav": "Do you like to read books or watch movies?",
    "training_data_wavs/sentence_9.wav": "I enjoy spending time with my friends and family.",
    "training_data_wavs/sentence_10.wav": "Have you tried the new restaurant in town?",
    "training_data_wavs/sentence_11.wav": "My phone number is 555-1234.",
    "training_data_wavs/sentence_12.wav": "I was born on January 15, 1995.",
    "training_data_wavs/sentence_13.wav": "I have 32 unread emails in my inbox.",
    "training_data_wavs/sentence_14.wav": "The meeting starts at 9 AM and ends at 11:30 AM.",
    "training_data_wavs/sentence_15.wav": "There are 24 hours in a day.",
    "training_data_wavs/sentence_16.wav": "The price of the item is $99.99.",
    "training_data_wavs/sentence_17.wav": "I’ll need 3 cups of flour and 2 eggs.",
    "training_data_wavs/sentence_18.wav": "The train arrives at 3:45 in the afternoon.",
    "training_data_wavs/sentence_19.wav": "It’s a sunny day with a few clouds in the sky.",
    "training_data_wavs/sentence_20.wav": "The temperature is 22 degrees Celsius.",
    "training_data_wavs/sentence_21.wav": "It started raining heavily after lunch.",
    "training_data_wavs/sentence_22.wav": "It’s windy outside, so hold on to your hat.",
    "training_data_wavs/sentence_23.wav": "Where is the nearest grocery store?",
    "training_data_wavs/sentence_24.wav": "What’s your favorite movie?",
    "training_data_wavs/sentence_25.wav": "How do I get to the bus station from here?",
    "training_data_wavs/sentence_26.wav": "Can you repeat that? I didn’t catch it.",
    "training_data_wavs/sentence_27.wav": "How far is the airport from downtown?",
    "training_data_wavs/sentence_28.wav": "What’s the weather like today?",
    "training_data_wavs/sentence_29.wav": "What is the capital of France?",
    "training_data_wavs/sentence_30.wav": "I love the sound of the waves crashing on the shore.",
    "training_data_wavs/sentence_31.wav": "It’s a beautiful day to explore the city.",
    "training_data_wavs/sentence_32.wav": "My favorite food is pizza with extra cheese.",
    "training_data_wavs/sentence_33.wav": "This is the best cup of coffee I’ve ever had.",
    "training_data_wavs/sentence_34.wav": "I’ll take a walk down by the beach later.",
    "training_data_wavs/sentence_35.wav": "She sells seashells by the seashore.",
    "training_data_wavs/sentence_36.wav": "Peter Piper picked a peck of pickled peppers.",
    "training_data_wavs/sentence_37.wav": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "training_data_wavs/sentence_38.wav": "Unique New York, unique New York.",
    "training_data_wavs/sentence_39.wav": "Six slippery snails slid silently southward.",
    "training_data_wavs/sentence_40.wav": "What did you do over the weekend?",
    "training_data_wavs/sentence_41.wav": "I’ve been working on a new project at work.",
    "training_data_wavs/sentence_42.wav": "I’m planning a trip to the mountains next month.",
    "training_data_wavs/sentence_43.wav": "Let’s grab a coffee and talk about it.",
    "training_data_wavs/sentence_44.wav": "It’s been a long day, I’m exhausted.",
    "training_data_wavs/sentence_45.wav": "We need to finalize the budget for the project.",
    "training_data_wavs/sentence_46.wav": "Can you send me the updated report by the end of the day?",
    "training_data_wavs/sentence_47.wav": "Let’s schedule a call for tomorrow morning.",
    "training_data_wavs/sentence_48.wav": "I need to review the contract before we proceed.",
    "training_data_wavs/sentence_49.wav": "I’m having trouble accessing the server.",
    "training_data_wavs/sentence_50.wav": "We need to discuss the new marketing strategy.",
    "training_data_wavs/sentence_51.wav": "I enjoy playing chess and solving puzzles.",
    "training_data_wavs/sentence_52.wav": "This is a test sentence for speech recognition.",
    "training_data_wavs/sentence_53.wav": "Could you please help me with this task?",
    "training_data_wavs/sentence_54.wav": "I’ve been learning to play the guitar.",
    "training_data_wavs/sentence_55.wav": "Let’s meet at the usual spot."
}


directory = "training_data_wavs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo']

csv_output_file_name = "output.csv"

with open(csv_output_file_name, "w", encoding="utf-8") as csv_file:
    csv_file.write(f'model|latency|file|prediction|truth\n')

for model_name in models:

    model = whisper.load_model(model_name)

    audio_files = os.listdir(directory)
    for audio_file in audio_files:
        for i in range(10):
            start = time.time()
            result = model.transcribe(directory + "/" + audio_file)
            end = time.time()
            latency = end - start
            output = (f'Transcription with {latency:.2f}s of latency for {model_name} model for {audio_file}:\n'
                      f'Prediction: {result["text"]}\nGround truth: {data[directory + '/' + audio_file]}\n')
            print(output)
            csv_output = (f'{model_name}|{latency:.2f}|{audio_file}|{result["text"]}|{data[directory + '/' + audio_file]}\n')
            with open(csv_output_file_name, "a", encoding="utf-8") as csv_file:
                csv_file.write(csv_output)
