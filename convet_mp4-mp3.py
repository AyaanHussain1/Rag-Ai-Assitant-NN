import os 
import subprocess
output_folder = "audios"
input_folder= r"videos\Deep Learning (ANN, RNN, Tranformers, RNN)"

if not os.path.exists(output_folder):

    os.mkdir(output_folder)
# COnverting Videos to Mp3 because we need to put minimum load to our models

files = os.listdir(input_folder)
for file in files:

    if not file.endswith(('.mp4','mkv')):
        continue

    split_file = file.split(" ｜ ")[0]
    # 3. FIX: Join the folder path to the filename so ffmpeg knows where to look!

    input_video_path = os.path.join(input_folder, file)
    output_audio_path = os.path.join(output_folder, f"{split_file}.mp3")

    subprocess.run(["ffmpeg","-i",input_video_path,output_audio_path])
    
print(f"Converting: {file} -> {split_file}.mp3")