import time

def printDefault(title,msg):
    currentTime = time.asctime(time.localtime())
    print('\x1b[6;30;42m' + title + " At Time "+ currentTime + '\x1b[0m')
    print(msg)
    return
