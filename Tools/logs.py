from Tools.files import readFile, writeFile


def appendToLogs(filename,object):
    db=readFile(filename)
    for item in db["list"]:
        if item["title"]==object["title"]:
            return 0
    db["list"].append(object)
    writeFile(filename,db)