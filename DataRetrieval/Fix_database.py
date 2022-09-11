from Tools.files import writeFile, readFile
import os

def ratingsFromAll():
    e_database = readFile(os.getcwd() + "\encoded_database.json")
    database = readFile(os.getcwd() + "\database.json")
    i = 0
    for item in e_database["list"]:
        database["list"][i]["rating"] = item["rating"]
        i += 1
    writeFile(os.getcwd() + "\database.json", database)

#We check if an element can lead to doubt when classifying
def isDubious(dubiousItem):
    return dubiousItem.capitalLetterWordsCount>=1 or dubiousItem.emojiCount >= 1 or dubiousItem.superlatives>=1 or dubiousItem.punctuationCount > 1

#We manually check the database
def dabasefix(revised_db, new_db, rating_limit, checkDubiousOnes,**kwargs):
    finish = False
    weblogs = kwargs.get("weblogs",False)
    for n_item in new_db["list"]:
        found = False
        next = False
        for r_item in revised_db["list"]:
            if r_item["title"] == n_item["title"]:
                n_item["rating"]= r_item["rating"]
                found = True
                break
        try:
            found = True if n_item["revised"] == 1 else found
        except:
            #Este elemento no tiene esto
            found = found
        if not found:
            from Classification.DecisionTree.TreeClass import VideoInfo
            dubiousItem = VideoInfo(False, n_item["title"],"",
                                  n_item["views"],
                                  n_item["likes"],
                                  n_item["subscribers"],
                                  n_item["category"])

            #Si no se da el caso pues no pasa nada porque no se borra el ya existente
            if (checkDubiousOnes is True and n_item["rating"] is 1 and isDubious(dubiousItem)) \
                    or (checkDubiousOnes is False):
                print(n_item["title"])
                print("rating: "+str(n_item["rating"]))
                if weblogs == True:
                    try:
                        print("Opinion:" + str(n_item["opinion"]))
                    except:
                        print("No opinion has been registered")
                try:
                    print("Subs: " + str(dubiousItem.subs))
                    print("SubsPerView: " + str(dubiousItem.subsPerView))
                    print("Views: " + str(dubiousItem.views))
                    print("Likes: " + str(dubiousItem.likes))
                except:
                    print("SubsPerView Unavailable")

                tryAgain = True
                while tryAgain:
                    try:
                        rating = input("No rating?")
                        print("\n")
                        if (rating != "f") and (rating != "n"): #f = finish; n = next
                            tryAgain = False
                            if ((int(rating) < rating_limit)):
                                rating = int(rating)
                            else:
                                tryAgain= True
                        else:
                            tryAgain = False
                            if rating == "f":
                                finish = True
                            elif rating == "n":
                                next = True
                            else:
                                tryAgain = True
                    except:
                        tryAgain = True

                if finish:
                    break
                if not next:
                    new_db["list"][new_db["list"].index(n_item)]["rating"] = rating
                    new_db["list"][new_db["list"].index(n_item)]["revised"] = 1
                    n_item["rating"] = rating
                    n_item["revised"] = 1
                    if("opinion" in n_item.keys()):
                        n_item.pop("opinion")
                    revised_db["list"].append(n_item)
                else:
                    #new_db["list"].pop(new_db["list"].index(n_item))
                    print("Saltem el element")
            else:
                revised_db["list"].append(n_item)
        else:
            found=False
    writeFile(os.getcwd() + "\\adjusted_database.json", revised_db)
    #writeFile(os.getcwd()+"\\webRequestLogs.json", new_db)

#We add a database to another, moving new elements to this "old_db" from "new_db"
def databaseAdding(old_db, new_db):
    found = False
    for r_item in old_db["list"]:
        found = False
        for n_item in new_db["list"]:
            if r_item["title"] == n_item["title"]:
                found = True
                break

        if not found:
            # We add the element missing from the old one to the new one
            new_db["list"].append(r_item)
        else:
            found=False

    #We clean possible mistakes (00 instead of 0, 11 instead of 1, 22 instead of 2)
    for item in new_db["list"]:
        if item["rating"] == 00:
            item["rating"] = 0
        elif item["rating"] == 11:
            item["rating"] = 1
        elif item["rating"] == 22:
            item["rating"] = 2

    print(len(new_db["list"]))
    writeFile(os.getcwd()+"\\adjusted_database.json", new_db)

#We check so no elements are repeated
def noRepeats(database):

    for item in database["list"]:
        firstLoop = True
        for check in database["list"]:
            if (check["title"] == item["title"]):
                if firstLoop:
                    firstLoop = False
                else:
                    database["list"].pop(database["list"].index(check))

    print(len(database["list"]))
    writeFile(os.getcwd()+"\\adjusted_database.json", database)

#We check al ratings from a single class
def ReviseAllRatingsFromClass(database, rating):
    import random
    #We randomize the array for better results
    random.shuffle(database["list"])
    finish = False
    for item in database["list"]:
        if item["rating"] == rating:
            print(item["title"])
            print("rating:" + str(item["rating"]) + "\n")

            tryAgain = True
            while tryAgain:
                try:
                    new_rating = input("No rating?")
                    if new_rating != "f" and new_rating != "n":  # f = finish; n = next
                        new_rating = int(new_rating)
                        database["list"][database["list"].index(item)]["rating"] = new_rating
                    elif new_rating == "f":
                        finish = True
                    tryAgain = False
                except:
                    tryAgain = True
            if finish:
                break

    writeFile(os.getcwd()+"\\adjusted_database.json", database)

#We change any rating written as an string to an integer
def removeTheStrings(database):

    for item in database["list"]:
        if item["rating"] == "" or item["rating"] == "0":
            database["list"][database["list"].index(item)]["rating"] = 0
        elif item["rating"] == "1":
            database["list"][database["list"].index(item)]["rating"] = 1
        elif item["rating"] == "2":
            database["list"][database["list"].index(item)]["rating"] = 2

    writeFile(os.getcwd()+"\\adjusted_database.json", database)

#Should a manual check fail due to unforseen errors, we use this to look at the terminal logs stored in a txt and recover the progress
def contingencyPlan(logFile,database):
    with open(logFile, encoding='utf8') as f:
        lines = f.readlines()
        counter = 0
        for line in lines:
            #Busquem el titol
            if "invalid literal" in line:
                continue
            if (("invalid literal" in lines[counter-1] ) or lines[counter-1] is '\n'):
                if "No rating?" not in line:
                    title = line
                else:
                    rating = int(line.split('?')[1])
                    if title is database["list"][counter]["title"]:
                        database["list"][counter]["rating"] = rating
                        counter += 1
                        print("Me cuadra el título")
                    else:
                        print("No me cuadra el título")
    writeFile(os.getcwd()+"\\adjusted_database.json", database)