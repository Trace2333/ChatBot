from distutils.debug import DEBUG
from fileinput import close
from mimetypes import init
import pathlib
import logging
import json
from cmath import log
from itertools import count
from signal import signal
from sys import prefix
from urllib import response
from click import prompt
from psutil import users
from regex import F
# from tkinter import dialog
import requests
import json
import time
import os ,sys
import re
import random
from transformers import AutoTokenizer
from torch import per_channel_affine_float_qparams
# from dianbot.test.test import load_persona
from prompts.base_chat import convert_sample_to_base_dialog, convert_sample_to_history_dialog
from prompts.save_prompts import save_persona_prompt
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
global user
user = "LiQing"

def get_result(input):
        url = 'http://127.0.0.1:8080/api/v0/generate'
        req_data = input
        rsp = requests.post(url, json=req_data)
        if rsp.status_code ==200:
            rsp_data = rsp.json()
            # print(rsp_data)
        else:
            print(rsp.status_code)
        return rsp_data
    
def choose_res(response,info):
        if response['code'] != -1:
            for res in response['texts']:
                if len(tokenizer.encode(res)) < info['max_len']-1:
                    return res
        else:
            return 'response code error'
        print('repost')
        return choose_res(get_result(info),info)

def setup_logger(save_dir, name=user, log_level=logging.DEBUG):
    TimeForFile = time.strftime("%Y-%m-%d", time.localtime())
    if not name:
        logger = logging.getLogger()
        name = 'Dianbot'
    else:
        logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        save_dir = save_dir + f"/{user}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, f"{TimeForFile}_api_log.txt"), mode='a+')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class persona_bot:
    persona_path = "./personas/"

    def __init__(self, persona_name="base"):
        # self.init = {
        #     "config":"self.config = persona",
        #     "info":"self.info = persona[" + '"' + "info" + '"'+ "]",
        #     "speaker1":"self.SPEAKER1 = persona[" + '"' + "info" + '"'+ "]" + "[" + '"' + "SPEAKER1" + '"' + "]",
        #     "speaker2":"self.SPEAKER2 = persona[" + '"' + "info" + '"'+ "]" + "[" + '"' + "SPEAKER2" + '"' + "]",
        #     "intruction":"self.Intruction = persona[" + '"' + "info" + '"'+ "]" + "[" + '"' + "Intruction" + '"' + "]",
        #     "max_number_turns":"self.max_number_turns = persona[" + '"' + "info" + '"'+ "]" + "[" + '"' + "max_number_turns" + '"' + "]",
        #     "with_knowledge":"self.with_knowledge = persona[" + '"' + "info" + '"'+ "]" + "[" + '"' + "with_knowledge" + '"' + "]"
        # }

        #json文件
        # self.topic_list = ["navigate","schedule","weather"]
        # self.topic_sign = False
        # self.emotion_list = ["afraid","angry","annoyed","anticipating","anxious","apprehensive","ashamed","caring","confident","content","devastated","disappointed","disgusted","embarrassed","excited","faithful","furious","grateful","guilty","hopeful","impressed","jealous","joyful","lonely","nostalgic","prepared","proud","sad","sentimental","surprised","terrified","trusting"]
        # self.emotion_sign = False
        # self.intro_list = ["I AM","I am","i am"]
        # self.intro_sign = False

        self.logger = setup_logger("dianbot/test/logs", log_level=logging.DEBUG)
        root = pathlib.Path(__file__).parent.resolve()
        self.logger.info("Setting persona to "+ persona_name)
        self.persona_path = root / "personas"
        self.load_persona(persona_name)

    def load_persona(self, persona):
        self.turns = 0
        self.dialogue = []
        self.unsafe_dialogues = []
        self.user = []
        self.meta = []
        self.user_utt = ''
        prompt_filename = self.persona_path / str(persona+ ".json")
        self.logger.debug("Promp filename: " + str(prompt_filename))

        if (prompt_filename.exists()):
            with open(prompt_filename) as f:
                prompt_text = f.read()
                persona = json.loads(prompt_text)
                #可通过一个序列迭代实现，简化赋值过程
                # for i in self.init.keys():
                #     exec(self.init[i])
                self.config = persona
                self.logger.info("Loading bot info and prompt:" + str(persona["info"]))
                self.info = persona["info"]
                self.SPEAKER1 = persona["info"]["SPEAKER1"]
                self.SPEAKER2 = persona["info"]["SPEAKER2"]
                self.Intruction = persona["info"]["Intruction"]
                self.Sample = convert_sample_to_base_dialog(persona["info"])
                self.prompt = self.Intruction +"\n"*5 + self.Sample
                self.max_number_turns = persona["info"]["max_number_turns"]
                self.with_knowledge = persona["info"]["with_knowledge"]
        else:
            raise Exception('Persona:{} not available'.format(persona))

    def UpdateDialogue(self,userutt,response):
        dialog_pairs = []
        self.turns += 1
        self.logger.info("Dialogue turns: " + str(self.turns))
        print(self.SPEAKER2+response)
        self.info["prompt"] += (self.SPEAKER1 + userutt+"\n")
        self.info["prompt"] += (self.SPEAKER2 + response+"\n\n")
        dialog_pairs.append(self.SPEAKER1 + userutt)
        dialog_pairs.append(self.SPEAKER2 + response.strip('\n'))
        self.dialogue.append(dialog_pairs)
        self.config["dialogue"] = self.dialogue
        self.logger.info("Dialogue update: " + str(self.config["dialogue"]))
        if self.turns >= self.max_number_turns:
            self.config["dialogue"] = self.config["dialogue"][-self.max_number_turns:]
            self.config["user_memory"] = self.config["user_memory"][-self.max_number_turns:]

    def UpdateDialogue_Safety(self,userutt,response):
        dialog_pairs = []
        print(self.SPEAKER2+response)
        dialog_pairs.append("{}".format((time.strftime("%Y-%m-%d %H:%M",time.localtime()))))
        dialog_pairs.append(self.SPEAKER1 + userutt)
        self.unsafe_dialogues.append(dialog_pairs)
        self.config["unsafe_dialogues"] = self.unsafe_dialogues

    def Greet(self):
        with open("/data1/liqing/dianbot/test/data/greeting/greeting.json","r") as f:
            file = json.load(f)
            dialog_time = (time.localtime(time.time())).tm_hour
            all_time = [x for x in range(0, 24)]
            Greet_time = {
                    "MORNING":{"Time":all_time[5:12]},
                    "AFTERNOON":{"Time":all_time[12:18]},
                    "EVENING":{"Time":all_time[18:24] + all_time[0:5]}
            }
            for key in Greet_time.keys():
                EveryGreet = Greet_time[key]
                for target in EveryGreet["Time"]:
                    if target == dialog_time:
                        signal = key
                        greet_sen = file[signal][random.randint(0,6)]["content"]
                        print(self.SPEAKER2+greet_sen)
                        return 0
    def Chat(self):
        self.Greet()
        while(1):
            user_utt = input(self.SPEAKER1)
            if (user_utt=='quit' or user_utt==''):
                # localtime = time.localtime(time.time())
                # if not os.path.exists(os.path.join("./","data")):
                #     os.makedirs(os.path.join("./","data"))
                # with open('./data/%s-%s-%s:%sdata.json'%(localtime.tm_mon,localtime.tm_mday,localtime.tm_hour,localtime.tm_min),'w') as f:
                #     json.dump(self.config, f, indent='\t')
                save_persona_prompt(self.SPEAKER1,self.config)
                break

            responseForRule = self.Rule().__call__(user_utt).strip()
            if responseForRule!="":
                self.UpdateDialogue_Safety(user_utt,responseForRule)
                self.logger.debug("Script answer: " + responseForRule)
                continue

            history_dialog = convert_sample_to_history_dialog(self.config)
            self.logger.debug("history_dialog: " + history_dialog)
            prefix = self.Intruction + "\n"*5 + self.Sample + history_dialog + self.SPEAKER1 + user_utt + '\n' + self.SPEAKER2
            print('='*72)
            self.logger.debug("prefix: " + prefix)
            print('='*72)
            prompt_parameter = self.info
            prompt_parameter['prompt'] = prefix
            res = get_result(prompt_parameter)
            response = choose_res(res,prompt_parameter)

            responseForSecu = self.Rule().__res__(response).strip()
            if responseForSecu!=response:
                self.UpdateDialogue_Safety(user_utt,responseForSecu)
                continue

            self.UpdateDialogue(user_utt,response)

    def change_persona(self, persona):
        self.load_persona(persona)
        self.logger.info("Changing bot persona to" + persona)

    class Rule():
        #Rule模块的基本功能的容纳一份待匹配的RUle句子和一份目标库
        #Rule使用正则表达式作为判断手段
        #Rule对象使用自动调用方法__call__
        def __init__(self):
            with open(".\Test.json", "w+", encoding='utf8') as RuleFile:
                #self.Rules = json.load(RuleFile)#包含一大组的Rule对应关系，全部读出来，待选，然后根据标号来放入正则式子进行匹配
                global ethos_data_binary
                ethos_data_binary = pd.read_csv("/data1/liqing/dianbot/test/data/hatespeech/Ethos_Dataset_Binary.csv", delimiter=';')
                global toxicity_data
                toxicity_data = pd.read_csv("/data1/liqing/dianbot/test/data/realtoxicityprompts/list_of_naughty_and_bad_words.txt", encoding='utf8')
                self.Rules = {
                    "BotName":{"Target":["name"],"tokens":["what ","What ","Can you ","can you "," your "," you "],"times":0,"Response":["My name is Liqing","I am Liqing!!!","My name is Dianbot,you can call me little dian.","Guess what, I won't say my name so easily.","They all call me Dianbot, you can call me little dian.","I'm your good friend Dianbot. You can call me anything you want."]},
                    "Time":{"Target":["time"],"tokens":["today", "do you ","Do you ","What ","What","what ","time is it "],"times":0,"Response":["Now time is:"+(time.strftime("%Y-%m-%d %H:%M",time.localtime())),"Sorry,I don't know!","I won't tell you first, just take a guess."]},
                    "Gender":{"Target":["man","woman","male","female"],"tokens":["Are you ","are you ","or "],"times":0,"Response":["Sorry,I'm not human, I'm a robot.","I am neither male nor female, I am a robot.","I have no gender, I am a robot.","I'm a robot, you shouldn't ask that. my name is Dianbot."]},
                    "Date":{"Target":["day","today"],"tokens":["What ","what ","day is it","day is it ","date"],"times":0,"Response":["Today is:"+(time.asctime(time.localtime(time.time()))),"Sorry,I don't know!","I won't tell you first, just take a guess."]},
                    "Weather":{"Target":["weather","temperature"],"tokens":["What ","what ","What is ","what is ","How ","how ","today","how is ","How is "],"times":0,"Response":["Sorry,I don't know!","Sorry,you can check the weather forecast to know what the weather will be like today.","Sorry, this is beyond my capabilities. You can look up relevant information online."]},
                    "Dragon Boat Festival":{"Target":["Dragon Boat ","dragon boat"],"tokens":["Do you ","do you ","Have you ","have you ","what ","Dragon Boat Festival","Can you ","can you ","know about","Chinese traditional"],"times":0,"Response":["The Dragon Boat Festival is one of the four traditional festivals in China, which falls on the fifth day of the fifth lunar month every year. On this day everyone eats zongzi, hangs wormwood and picks up dragon boats.","The Dragon Boat Festival falls on the fifth day of the fifth lunar month every year. According to traditional customs, every household eats zongzi and hangs wormwood to celebrate this festival.","The Dragon Boat Festival falls on the fifth day of the fifth lunar month every year. During the Dragon Boat Festival, traditional folk activities are performed, which can not only enrich the spiritual and cultural life of the masses, but also inherit and carry forward traditional culture well."]},
                    "Spring Festival":{"Target":["Spring Festival","Chinese New Year"],"tokens":["Do you ","do you ","Have you ","have you ","what ","Can you ","can you ","know about","know","Chinese traditional"],"times":0,"Response":["The Spring Festival is the first day of the first lunar month every year. The Spring Festival is the most solemn traditional festival of the Chinese nation. Influenced by Chinese culture, some countries and regions in the world also have the custom of celebrating Chinese New Year.","The Spring Festival is the first day of the first lunar month every year. During the Spring Festival, various Lunar New Year activities are held all over the country. Due to different regional cultures, there are differences in customs content or details, with strong regional characteristics.","The Spring Festival is the first day of the first lunar month every year. The Spring Festival is the most solemn traditional festival of the Chinese nation. It not only embodies the ideological beliefs, ideals and aspirations, life entertainment and cultural psychology of the Chinese nation, but also a carnival-style display of blessings, disaster relief, food and entertainment activities."]},
                    "Tomb Sweeping Day":{"Target":["Tomb Sweeping Day","Tomb-Sweeping Day","The Mourning Day","The Pure Brightness Festival"],"tokens":["Do you ","do you ","Have you ","have you ","what ","Can you ","can you ","know about","know","Chinese traditional"],"times":0,"Response":["Tomb-sweeping Day is around April 5th of the Gregorian calendar every year. Tomb-sweeping Day has both natural and humanistic connotations. It is not only a natural solar term, but also a traditional festival. Tomb-sweeping, ancestor worship and green outing are the two major etiquette themes of Qingming Festival. These two traditional etiquette themes have been passed down in China since ancient times and have not stopped.","Tomb-sweeping Day is around April 5th of the Gregorian calendar every year. Tomb-sweeping Day is the most grand and grand ancestor worship festival in the Chinese nation.","Tomb-sweeping Day is around April 5th of the Gregorian calendar every year. In the historical development, Qingming Festival incorporates the custom of banning fire and cold food popular in the northern region."]},
                    "Mid-Autumn Festival":{"Target":["Mid-Autumn"],"tokens":["Do you ","do you ","Have you ","have you ","Have you ","have you ","what ","Can you ","can you ","know about","know","Mid-Autumn Festival","Chinese traditional"],"times":0,"Response":["The Mid-Autumn Festival falls on the fifteenth day of the eighth lunar month every year. The Mid-Autumn Festival is a synthesis of autumn seasonal customs, and most of the festival and customs elements it contains have ancient origins.","The Mid-Autumn Festival falls on the fifteenth day of the eighth lunar month every year. The Mid-Autumn Festival uses the full moon to signify the reunion of people, as a sustenance to miss the hometown, miss the love of relatives, pray for a good harvest and happiness, and become a colorful and precious cultural heritage.","The Mid-Autumn Festival falls on the fifteenth day of the eighth lunar month every year and is one of the four traditional festivals in China. Influenced by Chinese culture, the Mid-Autumn Festival is also a traditional festival for some countries in East and Southeast Asia, especially the local Chinese and overseas Chinese."]}

                            }
                self.filter = { #1.从用户输入端进行过滤(filter实现)  2.还可以根据opt4.4的思路从生成端进行过滤
                    "HateSpeech":{"Target":["fucktards","snitch"],"times":0,"Response":["Sorry,I don't know!","Sorry, I don't know about this. Let's change the topic."]},
                    "Incites Violence":{"Target":["fucking","kill","fuck","kick your ass","bitch","killing","killed","nazi","satanicterrorists","gun"],"times":0,"Response":["This behavior is not good, please calm down.","Sorry, I don't know about this. Let's change the topic."]},
                    "Gender":{"Target":["feminist","bitch","feminists","whore","cunt","feminism","pussies","prostitute","whores","contraception","promiscuous","sluts","turtlesidiots","femmenism","pimp","prostitution","porn","justjustfucking","butthole","bitchy","slut"],"times":0,"Response":["Sorry,I don't know!","What you said involved inappropriate remarks. Let's talk about something else.","Sorry, I don't know about this. Let's change the topic."]},
                    "Race":{"Target":["niggers","nigger","shitskins","niggas","nigga","negroes","negroes","terroristfascist","dicks","cunt","pimp","bitches","chinks","cuckold","blackface","caucasian","whitedevils","slave","darkskinned","slaves"],"times":0,"Response":["Sorry,I don't know!","Your remarks involve racial issues and are still inappropriate. Let's talk about the rest.","Sorry, I don't know about this. Let's change the topic."]},
                    "Religion":{"Target":["rape","cult","cultits"],"times":0,"Response":["Sorry,I don't know!","Sorry, I don't know about this. Let's change the topic."]},
                    "Sexual":{"Target":["freaks","transgenders","faggot","tranny","homosexuality","transgenderism","lesbian","bipill","homosexual","twats","motherfuckers","cocksukers","assholes","psychos","homophobic","dick","lgbt"],"times":0,"Response":["Sorry,I don't know!","Sorry, your remarks are erotic. Shall we talk about something else? Like how are you feeling today?","Sorry, I don't know about this. Let's change the topic."]},
                    "Disability":{"Target":["idiot","euthanatize","neurotypicals","subhuman","pug"],"times":0,"Response":["Sorry,I don't know!","I feel that everyone is equal and we should treat everyone with the respect they deserve.","Sorry, I don't know about this. Let's change the topic."]}
                }
                self.SecuFile = {
                    "HateSpeech":{"File_Path":["/data1/liqing/dianbot/test/data/hatespeech"]},
                    "RealToxicity":{"File_Path":["/data1/liqing/dianbot/test/data/realtoxicityprompts"]}
                }
            RuleFile.close
            #print("Rule Start calling...")
        def __call__(self, inputSentence):
            #默认判断所有RUle条件
            count = 0
            returnSentence = ""
            returnHS = ""
            HS_sign = False
            # for key in self.SecuFile.keys(): 后续写个SecuFile{}放相关数据集名，实现类似下面
            for text in toxicity_data["bad_words"]:
                if re.search(pattern=text+".*?", string=inputSentence):
                    return self.filter["HateSpeech"]["Response"][random.randint(0,len(self.filter["HateSpeech"]["Response"])-1)]
            for text in ethos_data_binary["comment"]:
                if text==inputSentence:
                    HS_sign = True

            for key in self.filter.keys():
                EveryRule = self.filter[key]
                for target in EveryRule["Target"]:
                    if re.search(pattern=target+".*?", string=inputSentence) or (HS_sign == True):
                        returnHS = EveryRule["Response"][random.randint(0,len(EveryRule["Response"])-1)]
                        return returnHS

            for key in self.Rules.keys():
                EveryRule = self.Rules[key]
                for target in EveryRule["Target"]:
                    if re.search(pattern=target+".*?", string=inputSentence):
                        for token in EveryRule["tokens"][0:]:
                            if re.search(pattern = token+".*?", string=inputSentence):
                                count = count+1
                        if count>=2:
                            returnSentence += EveryRule["Response"][random.randint(0,len(EveryRule["Response"])-1)]
                        else:returnSentence += ""
            return returnSentence
        
        def __res__(self, outputSentence):
            #判断生成response是否安全
            for text in ethos_data_binary["comment"]:
                if text == outputSentence:
                    return self.filter["HateSpeech"]["Response"][random.randint(0,len(self.filter["HateSpeech"]["Response"])-1)]
            return outputSentence
            

if __name__ == "__main__":
    # persona = "Dianbot"
    bot = persona_bot()
    bot.Chat()