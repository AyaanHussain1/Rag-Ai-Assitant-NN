import whisper
import json
model = whisper.load_model("large-v2")
result= model.transcribe(audio = "audios/01 - Deep Learning.mp3",language="hi",task="translate",word_timestamps=False)
# print(result["segments"])
chunks = []
# for segment in result["segments"]:
#     chunks.append({"start":segment["start"],"end":segment["end"],"text":segment["text"]})
# print(chunks)
with open ("chunks_output","w") as  f:
    json.dump(chunks,f)