import sys
from codecs import encode
from json import load, dumps


def recurse(funcitem,funcfolder,obj):
    if "children" in obj:
        funcfolder(obj)
        for child in obj["children"]:
            recurse(funcitem,funcfolder,child)
    else:
        funcitem(obj)

#!crudely reverse engineered so might not be 100% correct
def tab_suspender(url: str):
    head="chrome-extension://fiabciakcmgepblmdkmemdbbkilneeeh/park.html?"
    if not url.startswith(head):
        raise ValueError("Not a tab suspension url: {0}".format(url))
    url = url[len(head):]
    url = url.split("&")
    for i in range(len(url)):
        url[i] = url[i].split("=")
    #leave only url
    url = [x for x in url if x[0] == "url"]
    if len(url) != 1:
        raise ValueError("Invalid url: {0}".format(url))
    url = url[0][1]
    delims={"%3A":":", "%2F":"/", "%3F":"?", "%3D":"=", "%26":"&", "%25":"%"}
    for delim in delims:
        url = url.replace(delim,delims[delim])
    return url

def great_suspender(url: str):
    head="chrome-extension://ahmkjjgdligadogjedmnogbpbcpofeeo/html/suspended.html#"
    if not url.startswith(head):
        raise ValueError("Not a tab suspension url: {0}".format(url))
    url = url[len(head):]
    url = url.split("&")
    for i in range(len(url)):
        url[i] = url[i].split("=")
    #leave only uri
    url = [x for x in url if x[0] == "uri"]
    if len(url) != 1:
        raise ValueError("Invalid url: {0}".format(url))
    url = url[0][1]
    return url


extension_map = {
    "fiabciakcmgepblmdkmemdbbkilneeeh": tab_suspender,
    "ahmkjjgdligadogjedmnogbpbcpofeeo": great_suspender
}

def process_url(url:str):
    if url.startswith("chrome-extension://"):
        extension=url[len("chrome-extension://"):].split("/")[0]
        if extension not in extension_map:
            raise ValueError("Unknown extension: {0}".format(extension))
        url = extension_map[extension](url)
    return url


def process_data(data:dict,out:dict):
    if "url" in data:
        out["url"]=process_url(data["url"])
        if "name" in data:
            out["name"]=data["name"]
        else:
            out["name"]=""
            print("No name for url: {0}".format(out["url"]))
    elif "children" in data:
        out["children"]=[]
        for child in data["children"]:
            childdict=dict()
            out["children"].append(childdict)
            process_data(child,childdict)
        if "name" in data:
            out["name"]=data["name"]
        else:
            out["name"]=""
            print("No name for folder")
    else:
        print("Unknown data: {0}".format(data))


#!relies on no urls being named [ or ]
def encode_data(data:dict,l=[]):
    l.append("\"{0}\"".format(data["name"]))
    if "children" in data:
        l.append("[")
        for child in data["children"]:
            encode_data(child,l)
        l.append("]")
    else:
        l.append(data["url"])

def main():

    if len(sys.argv) == 1:
        print("Usage: python test.py <filename>")
        return
    current_file_path=__file__
    last_slash=current_file_path.rfind("\\")
    buffer_file=current_file_path[:last_slash+1]+"buffer.txt"
        
    _in = sys.argv[1]
    b_path="C:\\Users\\Anton Bogun\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Bookmarks"
    with open(b_path, "r", encoding="utf-8") as f:
        data = load(f)
        structure = dict()
        process_data(data["roots"]["bookmark_bar"],structure)
        # structure = {"name":"hello","children":[{"name":"world","url":"http://www.google.com"},{"name":"more","url":"aaaaa"},{"name":"bbb","children":[{"name":"ccc","url":"http://www.google.com"},{"name":"ddd","url":"http://www.google.com"}]},
        # {"name":"eee","url":"http://www.google.com"}
        # ]}
        # print(structure["children"][7]["children"][0]["name"])
        # with open("out.json", "w", encoding="utf-8") as f:
        #     f.write(dumps(structure, indent=4)) #!note that json turns unicode into \uXXXX
        l=[]#list of list of bytes
        encode_data(structure,l)

        print(buffer_file,end="")
        with open(buffer_file,"w",encoding="utf-8") as f:
            f.write("\n".join(l))
        # sys.stdout.buffer.write(b)
        
        # print(b[0],b[1],b[2],b[3])
        # print(len(b))
        return
        # newline=encode("\n", "utf-8")
        # print(type(newline))
        # print(type(l))
        # print(type(l[0]))
        #join l into one list of bytes, with newline as delimiter
        # sys.stdout.buffer.write(newline.join(l))

if __name__ == "__main__":
    main()