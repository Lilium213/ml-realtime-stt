import os


def rename_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter out only .wav files
    wav_files = [f for f in files if f.endswith('.wav')]

    # Get the full path of the files
    full_paths = [os.path.join(folder_path, file) for file in wav_files]

    # Sort the files by creation time
    full_paths.sort(key=lambda x: os.path.getctime(x))

    # Rename files sequentially
    for i, file_path in enumerate(full_paths, start=1):
        # Extract the original file name
        old_name = os.path.basename(file_path)

        # Construct the new file name
        new_name = f"sentence_{i}.wav"

        # Construct the new full path
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(file_path, new_path)
        print(f"Renamed '{old_name}' to '{new_name}'")

rename_files_in_folder("F:\\vanbu\\Documents\\Sound Recordings")