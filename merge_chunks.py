import os 
import json
import math
x = 10
for filename in os.listdir("jsons"):
    if filename.endswith(".json"):
        file_path = os.path.join("jsons",filename)
        with open (file_path,"r",encoding="utf-8") as f:
            data = json.load(f)
            # print(data)
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_groups = math.ceil(num_chunks/x)

            for i in range(num_groups):
                start_idx = i*x
                end_idx = min((i+1)*x,num_chunks)

                chunk_group = data["chunks"][start_idx:end_idx]
                new_chunks.append({"number":data["chunks"][0]["number"],
                                   "title":chunk_group[0]["title"],
                                   "start":chunk_group[0]["start"],
                                   "end":chunk_group[-1]["end"],
                                   "text":" ".join(c["text"] for c in chunk_group)})
            os.makedirs("new_jsons",exist_ok=True)
            with open(os.path.join("new_jsons",filename),"w",encoding="utf-8") as json_f:
                json.dump({"chunks":new_chunks,"text":data["text"]},json_f)
