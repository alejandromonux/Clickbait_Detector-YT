from Tools.files import readFile, writeFile


def appendToLogs(filename,object, rating):
    db=readFile(filename)
    object["rating"] = rating
    for item in db["list"]:
        if item["title"]==object["title"]:
            return 0
    db["list"].append(object)
    writeFile(filename,db)
def modifyLogs(filename, object, opinion):
    db=readFile(filename)
    for i  in range(len(db["list"])):
        if db["list"][i]["title"]==object["title"]:
            db["list"][i]["opinion"] = opinion
            break
    writeFile(filename,db)