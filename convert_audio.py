import os
import subprocess

INPUT_DIR = "data/raw_audio"
OUTPUT_DIR = "data/processed_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FFMPEG_PATH = r"C:\Users\shreyabhat\ffmpeg\bin\ffmpeg.exe"

for file in os.listdir(INPUT_DIR):

    if file.endswith(".m4a"):

        input_path = os.path.join(INPUT_DIR, file)

        output_file = file.replace(".m4a", ".wav")
        output_path = os.path.join(OUTPUT_DIR, output_file)

        command = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]

        try:

            subprocess.run(
                command,
                check=True
            )

            print(f"Converted: {file}")

        except subprocess.CalledProcessError as e:

            print(f"Failed: {file}")
            print(e)