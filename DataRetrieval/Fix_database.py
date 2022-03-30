from Tools.files import writeFile
import os

def dabasefix(revised_db, new_db):
    finish = False
    #TODO: REVISAR
    for n_item in new_db["list"]:
        found = False
        next = False
        for r_item in revised_db["list"]:
            if r_item["title"] == n_item["title"]:
                n_item["rating"]= r_item["rating"]
                found = True
                break;
        if not found:
            print(n_item["title"])
            print("rating:"+str(n_item["rating"])+"\n")

            tryAgain = True
            while tryAgain:
                try:
                    rating = input("No rating?")
                    if rating != "f" and rating != "n": #f = finish; n = next
                        rating = int(rating)
                    else:
                        if rating == "f":
                            finish = True
                        else:
                            next = True
                    tryAgain = False
                except:
                    tryAgain = True

            if finish:
                break
            if not next:
                new_db["rating"] = rating
            else:
                new_db["list"].pop(new_db["list"].index(n_item))
        else:
            found=False

    writeFile(os.getcwd()+"\\adjusted_database.json", new_db)

def databaseAdding(old_db, new_db):
    found = False
    #TODO: REVISAR
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