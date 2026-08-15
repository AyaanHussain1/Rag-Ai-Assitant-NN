import whisper
import json
import os

model = whisper.load_model("large-v2")
audios = os.listdir("audios")

for audio in audios:
    if "-" in audio and (audio.endswith(".mp3") or audio.endswith(".mp4")):

        number = audio.split("-")[0]
        title = audio.split("-")[1]

        title = title.replace(".mp3","").replace(".mp4","")
        print(number,title)

        result= model.transcribe(audio = f"audios/{audio}",language="hi",task="translate",word_timestamps=False) 
        
        chunks = []
        for segment in result["segments"]:
             
             chunks.append({"numbers" : number , "Title" : title, "start":segment["start"],"end":segment["end"],"text":segment["text"]})
        
        chunks_with_metadata = {"chunks":chunks,"text":result["text"]}

        with open (f"jsons/{audio}.json","w") as  f:
            json.dump(chunks_with_metadata,f)
        