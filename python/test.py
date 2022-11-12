#read args
import codecs
import sys
import os

def main():
    if len(sys.argv) == 1:
        print("Usage: python test.py <filename>")
        return
    _in = sys.argv[1]

    # print("{0} {0}".format(_in))
    # print("hello")
    # print("Ā")#FIXME UnicodeEncodeError: 'charmap' codec can't encode character '\u0100' in position 0: character maps to <undefined>
    #print charmap
    # temp="hello4вый \\\ndsd"
    # print(temp)
    temp = open("scripts/python/test.txt","r").read()
    print(temp[4:11]+" hmm "+temp[4:12])
    # print(str(temp.encode('utf-8'))[2:-1])
    # print(str("ф".encode('utf-8'))[2:-1])

    # with open("python/test.txt","w") as f:
    #     f.write(temp)
    # temp2=codecs.encode(temp,encoding='utf-8')
    # temp3=codecs.decode(temp2,encoding='utf-8')
    # print(temp3)

    
    # with open("python/test.txt","r") as f:
    #     print(f.read())
    #print current directory
    # print(os.getcwd())
    return


main()
