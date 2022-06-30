import os
from copy import copy, deepcopy
from locale import atoi
import re

import emoji as emoji

from Tools.files import readFile


class VideoInfo:
    emojiCount = 0
    hasTypicalPhrase = False
    superlatives = 0
    punctuationCount = 0
    bigNum = False
    sensationalistAdjectives = 0
    capitalLetterWordsCount = 0
    views = 0
    subsPerView = 0
    likes = 0
    #relaciónVisitas_Subs = 0
    category = 0



    def __init__(self,empty,title,encoded_title,visitas,likes,subs,category):
        if not empty:
            self.emojiCount = self.ComptaEmojis(title)
            self.hasTypicalPhrase = self.ScanForTypicalPhrase(title)
            self.superlatives = self.countSuperlatives(title)
            self.punctuationCount = self.countPunctuation(title)
            self.bigNum = self.HasBigNumber(title)
            #sensationalistAdjectives = self.hasSensationalistAdjectives(title)
            self.capitalLetterWordsCount = self.countCapitalLetterWords(title)
            self.wordCount = self.getWordCount(title)
            self.views = atoi(visitas)
            self.likes = atoi(likes)
            try:
                self.subsPerView = atoi(subs) / self.views #Subs Per view. Si es bajo significa que el video tiene mucho engagement y por tanto posibilidad de clickbait más alta
            except:
                #Esto es porque si views es 0 pues cagamos
                self.subsPerView=self.views
            self.encodedTitle=encoded_title
            #relaciónVisitas_Subs = relacion_visitas_subs
            self.category = category
        else:
            self.emojiCount = 0
            self.hasTypicalPhrase = False
            self.superlatives = 0
            self.punctuationCount = 0
            self.bigNum = False
            self.sensationalistAdjectives = 0
            self.capitalLetterWordsCount = 0
            self.views = 0
            self.subsPerView = 0
            self.likes = 0
            # relaciónVisitas_Subs = 0
            self.category = 0


    def returnAsArray(self):
        phrase = 1 if self.hasTypicalPhrase == True else 0
        bignumber=1 if self.bigNum == True else 0
        return [#self.encodedTitle,
                self.emojiCount,
                phrase,
                self.superlatives,
                self.punctuationCount,
                bignumber,
                self.sensationalistAdjectives,
                self.capitalLetterWordsCount,
                self.views,
                self.subsPerView,
                self.likes,
                self.category]

    def ComptaEmojis(self,title) -> int:
        result = re.findall(r'[^\w\s,.]',title)
        return len(result)

    def ScanForTypicalPhrase(self,title) -> bool:
        titlecopy = deepcopy(title)
        if "won’t believe what happens next" in titlecopy.casefold() or "gone wrong" in titlecopy.casefold() or "Here’s what happened" in titlecopy.casefold():
            return True
        else:
            return False

    def countSuperlatives(self,title) -> int:
        #TODO: Open file of superlatives
        counter = 0
        superlatives = readFile(os.getcwd()+"\..\Classification\DecisionTree\superlatives.json")
        for word in title.split(" "):
            if word in superlatives["list"]:
                counter += 1
        return counter

    def countPunctuation(self,title) -> int:
        count = 0
        for character in title:
            if character is '|' or\
                    character is '!' or\
                        character is '?' or \
                            character is '.':
                count += 1
        return count

    def HasBigNumber(self,title) -> bool:
        for part in title.split(" "):
            try:
                if part[0] is '$' or part[0] is '€':
                    part[0] = '0'
                if part[0].isdecimal():
                    if part[len(part)-1] is '$' or part[len(part)-1] is '€':
                        part = part[0:len(part) - 2]
                    if atoi(part) >= 1000:
                        return True
            except Exception as e:
                print(e)

    def hasSensationalistAdjectives(self,title) -> int:
        #TODO: Open file of adjectives
        subStrings = title.split(" ")

    def getWordCount(self,title) -> int:
        return len(title)

    def countCapitalLetterWords(self,title) ->int:
        counter = 0
        subStrings = title.split(" ")
        for part in subStrings:
            wordcheck = 0
            for character in part:
                if not character.isupper():
                    break
                wordcheck += 1
            if wordcheck == len(part):
                counter += 1
        return counter