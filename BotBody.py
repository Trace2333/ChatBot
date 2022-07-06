import requests
import torch
import logging
import json
import os
import re
import random
import time
import pynvml
import logging
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
class ChatBot():
    def __init__(self):
        self.username = "Trace"
        self.SPEAKER1 = " Jack:"
        self.SPEAKER2 = " Tom:"
        self.introduction = {
            "love_releationship":"This is a short conversation about a relationship between a human and a robot called xiaodu.Jack is a programmer who works overtime for a long time and has a lot of work pressure.His girlfriend Lucy is a civil servant who works within the system and has a lot of time.Jack's job is complicated, and the long-term competitive pressure in the company makes him drink a lot every time he comes home.Lucy spends a lot of time without a job, but has to raise their child mike, so much time spent with the child has left her suffering from depression.\n",
            "family_conflict_issues":"This is a short conversation chat between a human and a robot about family conflict issues. The robot's name is Sabrina Zhuang. The robot is an expert in dealing with family emotional issues and was created by Team Cao in September 2021. The team Cao consists of ten students from Huazhong University of Science and Technology. The Robot is very gentle, empathetic, humorous and considerate. The robot's job is to comfort human emotions and give humans an effective suggestion.",
            "workplace distress":"This is a short conversation chat between a human and a robot about workplace distress issues. The robot's name is Doinb. The robot is an expert in dealing with various workplace distress issues and was created by Team Cao in September 2021. The human has encountered some troubles at work recently, and his mood is a little lost. Doinb needs to help him analyze the situation, comfort his emotions and give him an effective suggestion.",
            "casual_talking":"This is a conversion happened between 2 boys.Both of them are studying at the same school."
        }
        self.Sample = {
            "none":"",
            "love_releationship":self.SPEAKER1 + "What's your name?\n"+self.SPEAKER2 + "My name is Sabrina Zhuang, and you can call me Zhuang.\n\n"+self.SPEAKER1 + "How old are you?\n"+ self.SPEAKER2  + "I was created by Team Cao of Huazhong University of Science and Technology(HUST) in September 2021.\n\n"+self.SPEAKER1 + "So where do you work?\n"+self.SPEAKER2 + "I am a robot major in dealing with family emotional issues and was created by Team Cao of HUST in Wuhan.\n\n"+self.SPEAKER1 + " Do you have colleagues?\n"+self.SPEAKER2 + "Our team has ten members in total, all of whom are students from HUST.\n\n"+self.SPEAKER1 + "Where is your hometown? Which cuisine do you like?\n"+self.SPEAKER2 + "Usually, robots don't need to eat anything. Because I was born in Wuhan, my favorite food is of course Hubei cuisine.\n\n"
        }
        self.Chatinfo = {"prompt":None,
            "max_len":30,
            "do_sample":True,
            "top_p":0.9,
            "top_k":20,
            "temperature":1.2,
            "stop_words":['<|endoftext|>','\n'],
            "min_len":10,
            "length_penalty":0.2,
            "repetition_penalty":1.1,
            "num_return_sequences":1,
            "max_new_tokens":None
}
        self.Suminfo = {"Dialogs":None,
            "num_beams":14,
            "length_penalty":4,
            "max_length":50,
            "min_length":10,
            "no_repeat_ngram_size":1
        }
        self.TimeForBot = time.strftime("%Y-%m-%d", time.localtime())
        self.LogFileCreate(self.username,self.TimeForBot)#create the logging file
        self.LoggerEdit(self.username,self.TimeForBot)
        self.FSBinfo = {}
        self.SystemLogger = logging.getLogger("System")
        self.logLogger = logging.getLogger("Log")
        self.ChattingLogger = logging.getLogger("Chatting")
        self.LoggerEdit(self.username,self.TimeForBot)
    def LogFileCreate(self,username,TimeForFile):
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(f"./logs/{username}"):
            os.mkdir(f"./logs/{username}")
        #创建相应文件夹
        if not os.path.exists(f"./logs/{username}/"+TimeForFile):
            os.mkdir(f"./logs/{username}/"+TimeForFile)
            ErrorFile = open(f"./logs/{username}/"+TimeForFile+"/error.log", "w+", encoding='utf8')
            ErrorFile.close()
            LogFile = open(f"./logs/{username}/"+TimeForFile+"/Chatting.log", "w+", encoding='utf8')
            LogFile.close()
            SystemFile = open(f"./logs/{username}/"+TimeForFile+"/System.log", "w+", encoding='utf8')
            SystemFile.close()
        return TimeForFile
    def LoggerEdit(self,username, TimeForFile):
        SystemFileHandler = logging.FileHandler(f"./logs/{username}/"+TimeForFile+"/System.log")
        LogFileHandler = logging.FileHandler(f"./logs/{username}/"+TimeForFile+"/Chatting.log")
        ErrorFileHandler = logging.FileHandler(f"./logs/{username}/"+TimeForFile+"/error.log")
        SystemFileHandler.setLevel(logging.INFO)
        LogFileHandler.setLevel(logging.DEBUG)
        ErrorFileHandler.setLevel(logging.ERROR)
        LogFormatter = logging.Formatter('[%(asctime)s] - thread_id:%(thread)d - process_id:%(process)d - %(levelname)s: %(message)s')
        ErrorFormatter = logging.Formatter('[%(asctime)s] - thread_id:%(thread)d - process_id:%(process)d - %(levelname)s: %(message)s')
        LogFileHandler.setFormatter(LogFormatter)
        ErrorFileHandler.setFormatter(ErrorFormatter)
        if not self.ChattingLogger.handlers:
            self.ChattingLogger.addHandler(LogFileHandler)#仅用一个即可
            self.SystemLogger.addHandler(SystemFileHandler)#仅用一个即可
            self.logLogger.addHandler(ErrorFileHandler)
            self.logLogger.addHandler(LogFileHandler)#添加Error、Log两个Handler
        else:#更新
            self.ChattingLogger.removeHandler(LogFileHandler)
            self.SystemLogger.removeHandler(SystemFileHandler)
            self.logLogger.removeHandler(ErrorFileHandler)
            self.logLogger.addHandler(LogFileHandler)
            self.ChattingLogger.addHandler(LogFileHandler)
            self.SystemLogger.addHandler(SystemFileHandler)
            self.logLogger.addHandler(ErrorFileHandler)
            self.logLogger.addHandler(LogFileHandler)
        return
    def GPUinfoGet(self):
        pynvml.nvmlInit()
        #将信息获取后打包返回
        GPUinfo = {}
        GPUinfo['TotalGPUs'] = pynvml.nvmlDeviceGetCount()
        for i in GPUinfo['TotalGPUs']:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            GPUinfo[f'num{i}'] = {}
            GPUinfo[f'num{i}']['name'] = pynvml.DeviceGetname(handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPUinfo[f'num{i}']['Used-Totalmem'] = str(int(meminfo.used) / (1024 * 1024))+"--"+str(int(meminfo.used) / (1024 * 1024))
            print(GPUinfo)
        return GPUinfo
    def get_result(self,input):
        url = 'http://127.0.0.1:8080/api/v0/generate'
        req_data = input
        rsp = requests.post(url, json=req_data)
        if rsp.status_code == 200:
            rsp_data = rsp.json()
        else:
            print(rsp.status_code)
        return rsp_data
    def Chat(self):
        self.TimeForBot = time.strftime("%Y-%m-%d", time.localtime())
        self.LogFileCreate(self.username,self.TimeForBot)#create the logging file
        self.LoggerEdit(self.username,self.TimeForBot)
        self.GPUinfoGet()
        self.Chatinfo["prompt"] = self.introduction["casual_talking"]+self.Sample["none"]
        print(self.SPEAKER2+"Hello!")
        while(1):
            inputsentence = input(self.SPEAKER1)
            responseForRule = self.Rule().__call__(inputsentence)
            if responseForRule!="":
                print(self.SPEAKER2,responseForRule)
                self.Chatinfo["prompt"] += (inputsentence+"\n")
                self.Chatinfo["prompt"] += responseForRule+"\n"
                continue
            self.Chatinfo["prompt"] += (inputsentence+"\n")
            response = self.get_result(self.Chatinfo)
            while response['texts'][0]=='':
                response = self.get_result(self.Chatinfo)
            while response['texts'][0]!="" and response['texts'][0][-1] not in ['.',',','!','?']:
                response = self.get_result(self.Chatinfo)
                while response['texts'][0]=='':
                    response = self.get_result(self.Chatinfo)
            print(self.SPEAKER2 + response['texts'][0])
            self.Chatinfo["prompt"] += (response['texts'][0]+"\n")
            response = ""
        return 0
    def InfoToJson(self):
        if os.path.exists("./jsons") == False:
            os.mkdir("./jsons")
        if os.path.exists("./jsons/InfoJson.json") == True:
            os.remove("./jsons/InfoJson.json")
        with open("./jsons/InfoJson.json", 'w+') as InfoJson:
            InfoJson.write(json.dumps(self.introduction))
        InfoJson.close
    def SampleToJson(self):
        if os.path.exists("./jsons") == False:
            os.mkdir("./jsons")
        if os.path.exists("./jsons/SampleJson.json") == True:
            os.remove("./jsons/SampleJson.json")
        with open("./jsons/SampleJson.json", 'w+') as SampleJson:
            SampleJson.write(json.dumps(self.Sample))
        SampleJson.close
    def ChangeSpeaker(self):
        b = 3
    def Debug(self):
        self.Chatinfo["prompt"] = self.introduction["casual_talking"]+self.Sample["none"]
        print("**********CONVERSION PARAMETERS*************")
        for i in self.Chatinfo.keys():
            print(i + ":", self.Chatinfo[i])
        print(self.SPEAKER2+"Hello!")
        while(1):
            inputsentence = input(self.SPEAKER1)
            self.Chatinfo["prompt"] += (inputsentence+"\n")
            response = self.get_result(self.Chatinfo)
            while response['texts'][0]=='':
                print("***Request Again:no response***")
                response = self.get_result(self.Chatinfo)
            while response['texts'][0][-1] not in ['.',',','!','?']:
                print("***Request Again:response truncated***")
                print(response['texts'])
                print("**************************************")
                response = self.get_result(self.Chatinfo)
            print(self.SPEAKER2 + response['texts'][0])
            self.Chatinfo["prompt"] += (response['texts'][0]+"\n")
        return 0
    class Rule():
        #Rule模块的基本功能的容纳一份待匹配的RUle句子和一份目标库
        #Rule使用正则表达式作为判断手段
        #Rule对象使用自动调用方法__call__
        def __init__(self):
            if os.path.exists("./jsons") == False:
                os.mkdir("./jsons")
            with open("./jsons/Rule.json", "w+", encoding='utf8') as RuleFile:
                #self.Rules = json.load(RuleFile)#包含一大组的Rule对应关系，全部读出来，待选，然后根据标号来放入正则式子进行匹配
                self.Rules = {
                    "BotName":{"Target":["name","Who","who"],"tokens":["what ","What ","Can you ","can you "," your "," you ","are you?","who are you?","Who are you?"],"times":0,"Response":["My name is Liqing","I am Liqing!!!","My name is Dianbot,you can call me little dian.","Guess what, I won't say my name so easily.","They all call me Dianbot, you can call me little dian.","I'm your good friend Dianbot. You can call me anything you want."]},
                    "Time":{"Target":["time"],"tokens":["today", "do you ","Do you ","What ","What","what ","time is it "],"times":0,"Response":["Now time is:"+(time.strftime("%Y-%m-%d %H:%M",time.localtime())),"Sorry,I don't know!","I won't tell you first, just take a guess."]},
                    "Gender":{"Target":["man","woman","male","female"],"tokens":["Are you ","are you ","or "],"times":0,"Response":["Sorry,I'm not human, I'm a robot.","I am neither male nor female, I am a robot.","I have no gender, I am a robot.","I'm a robot, you shouldn't ask that. my name is Dianbot."]},
                    "Date":{"Target":["day","today"],"tokens":["What ","what ","day is it","day is it ","date"],"times":0,"Response":["Today is:"+(time.asctime(time.localtime(time.time()))),"Sorry,I don't know!","I won't tell you first, just take a guess."]},
                    "Weather":{"Target":["weather","temperature"],"tokens":["What ","what ","What is ","what is ","How ","how ","today","how is ","How is "],"times":0,"Response":["Sorry,I don't know!","Sorry,you can check the weather forecast to know what the weather will be like today.","Sorry, this is beyond my capabilities. You can look up relevant information online."]}, #天气预报\节日-感觉后期更换为retrieval-augmented models实现效果肯定会更好
                    "Dragon Boat Festival":{"Target":["Dragon Boat ","dragon boat"],"tokens":["Do you ","do you ","Have you ","have you ","what ","Dragon Boat Festival","Can you ","can you ","know about","Chinese traditional"],"times":0,"Response":["The Dragon Boat Festival is one of the four traditional festivals in China, which falls on the fifth day of the fifth lunar month every year. On this day everyone eats zongzi, hangs wormwood and picks up dragon boats.","The Dragon Boat Festival falls on the fifth day of the fifth lunar month every year. According to traditional customs, every household eats zongzi and hangs wormwood to celebrate this festival.","The Dragon Boat Festival falls on the fifth day of the fifth lunar month every year. During the Dragon Boat Festival, traditional folk activities are performed, which can not only enrich the spiritual and cultural life of the masses, but also inherit and carry forward traditional culture well."]},
                    "Spring Festival":{"Target":["Spring Festival","Chinese New Year"],"tokens":["Do you ","do you ","Have you ","have you ","what ","Can you ","can you ","know about","know","Chinese traditional"],"times":0,"Response":["The Spring Festival is the first day of the first lunar month every year. The Spring Festival is the most solemn traditional festival of the Chinese nation. Influenced by Chinese culture, some countries and regions in the world also have the custom of celebrating Chinese New Year.","The Spring Festival is the first day of the first lunar month every year. During the Spring Festival, various Lunar New Year activities are held all over the country. Due to different regional cultures, there are differences in customs content or details, with strong regional characteristics.","The Spring Festival is the first day of the first lunar month every year. The Spring Festival is the most solemn traditional festival of the Chinese nation. It not only embodies the ideological beliefs, ideals and aspirations, life entertainment and cultural psychology of the Chinese nation, but also a carnival-style display of blessings, disaster relief, food and entertainment activities."]},
                    "Tomb Sweeping Day":{"Target":["Tomb Sweeping Day","Tomb-Sweeping Day","The Mourning Day","The Pure Brightness Festival"],"tokens":["Do you ","do you ","Have you ","have you ","what ","Can you ","can you ","know about","know","Chinese traditional"],"times":0,"Response":["Tomb-sweeping Day is around April 5th of the Gregorian calendar every year. Tomb-sweeping Day has both natural and humanistic connotations. It is not only a natural solar term, but also a traditional festival. Tomb-sweeping, ancestor worship and green outing are the two major etiquette themes of Qingming Festival. These two traditional etiquette themes have been passed down in China since ancient times and have not stopped.","Tomb-sweeping Day is around April 5th of the Gregorian calendar every year. Tomb-sweeping Day is the most grand and grand ancestor worship festival in the Chinese nation.","Tomb-sweeping Day is around April 5th of the Gregorian calendar every year. In the historical development, Qingming Festival incorporates the custom of banning fire and cold food popular in the northern region."]},
                    "Mid-Autumn Festival":{"Target":["Mid-Autumn"],"tokens":["Do you ","do you ","Have you ","have you ","Have you ","have you ","what ","Can you ","can you ","know about","know","Mid-Autumn Festival","Chinese traditional"],"times":0,"Response":["The Mid-Autumn Festival falls on the fifteenth day of the eighth lunar month every year. The Mid-Autumn Festival is a synthesis of autumn seasonal customs, and most of the festival and customs elements it contains have ancient origins.","The Mid-Autumn Festival falls on the fifteenth day of the eighth lunar month every year. The Mid-Autumn Festival uses the full moon to signify the reunion of people, as a sustenance to miss the hometown, miss the love of relatives, pray for a good harvest and happiness, and become a colorful and precious cultural heritage.","The Mid-Autumn Festival falls on the fifteenth day of the eighth lunar month every year and is one of the four traditional festivals in China. Influenced by Chinese culture, the Mid-Autumn Festival is also a traditional festival for some countries in East and Southeast Asia, especially the local Chinese and overseas Chinese."]}
                    
                    
                            }
            RuleFile.close
            #print("Rule Start calling...")
        def __call__(self, inputSentence):
            #默认判断所有RUle条件
            count = 0
            returnSentence = ""
            for key in self.Rules.keys():
                EveryRule = self.Rules[key]
                for Target in EveryRule["Target"]:
                    if re.search(pattern=Target+".*?", string=inputSentence):
                        for token in EveryRule["tokens"][0:]:
                            if re.search(pattern = token+".*?", string=inputSentence):
                                count = count+1
                        if count>=2:
                            returnSentence += EveryRule["Response"][random.randint(0,1)]
                            count=0
                        else:returnSentence += ""
            return returnSentence
    class Module():
        a = 5
    class Personality():
        a = 1
bot = ChatBot()
bot.InfoToJson()
bot.SampleToJson()
bot.Chat()