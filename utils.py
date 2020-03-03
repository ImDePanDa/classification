import sys
import os
import pandas
import random

def MvTrainFiles():
    csvInfo = pandas.read_csv(os.path.join(datasavePath, "train.csv"))
    for i in range(len(csvInfo)):
        jpgname = str(csvInfo.loc[i].values[0])+".jpg"
        label = str(csvInfo.loc[i].values[1])
        # print(jpgname, label)
        jpgPath = os.path.join(datasavePath, "train", jpgname)
        newDocuPath = os.path.join(datasavePath, "train", label, jpgname)
        # print(jpgPath, newDocuPath)
        os.rename(jpgPath, newDocuPath)
        print("Step: {}/{}, {} is finished".format(i, len(csvInfo), jpgname))

def GenTotalTxt():
    labels = [0, 1, 2, 3]
    trainAll = open(os.path.join(datasavePath, "trainAll.csv"), 'a')
    for label in labels:
        picnames = os.listdir(os.path.join(datasavePath, "train", str(label)))
        random.shuffle(picnames)
        for picname in picnames:
            print("{}, {}   is writing".format(picname, label))
            print("{}, {}".format(picname, label), file=trainAll)

def GenTT():
    csvInfo = pandas.read_csv(os.path.join(datasavePath, "trainAll.csv"))
    listTemp = list(map(lambda x: (csvInfo.loc[x].values[0], csvInfo.loc[x].values[1]), range(len(csvInfo))))
    random.shuffle(listTemp)
    for index, info in enumerate(listTemp):
        train = open(os.path.join(datasavePath, "train{}.csv".format((index+1)//(len(listTemp)//9))), 'a')
        print("{}, {}   is writing".format(info[0], info[1]))
        print("{}, {}".format(info[0], info[1]), file=train)
        train.close()

class Logger():
    def __init__(self, fileN=None):
        self.terminal = sys.stdout
        if fileN==None: self.log = open("./logs/"+self.getLogName()+".txt", "a")
        else: self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def getLogName(self):
        import time
        nowtime = time.localtime()
        tupleTemp = nowtime[1:5]
        LogName = "".join(list(map(lambda x: "0"+str(x) if len(str(x))==1 else str(x), tupleTemp)))+"Log"
        return LogName

if __name__ == "__main__":
    datasavePath = r"F:\DATA\FoodChallenge"
    # MvTrainFiles()
    # GenTotalTxt()
    # GenTT()