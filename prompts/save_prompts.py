import json
import time
import os

def save_persona_prompt(user_name,prompt):
    localtime = time.localtime(time.time())
    user_name = user_name.strip().strip('[').strip(':').strip(']')
    prompt["info"].pop("prompt")
    if not os.path.exists(os.path.join("./dianbot/test/data/Userinfo",f"{user_name}")):
        os.makedirs(os.path.join("./dianbot/test/data/Userinfo",f"{user_name}"))
    with open('./dianbot/test/data/Userinfo/%s/%s-%s-%s:%sdata.json'%(user_name,localtime.tm_mon,localtime.tm_mday,localtime.tm_hour,localtime.tm_min),'w') as f:
        json.dump(prompt, f, indent='\t')