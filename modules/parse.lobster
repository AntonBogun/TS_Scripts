

def decode_csv(s_raw:string) -> [string], string?:
    let a,b=decode_csv(string_to_unicode(s_raw))
    return a,b



def decode_csv(s:[int]) -> [string], string?:
    let curr_str=[]
    let out=[]
    var i=0
    while i<length(s):
        if s[i]=='\\':
            if i+1>=length(s):
                return [], "unexpected end of string"
            if s[i+1]=='\\':
                curr_str.push("\\")
                i+=2
            elif s[i+1]==',':
                curr_str.push(",")
                i+=2
            else:
                return [], "unexpected character code after \\: {s[i+1]}"
        elif s[i]==',':
            out.push(concat_string(curr_str,""))
            curr_str.remove_range(0,length(curr_str))
            i+=1
        else:
            curr_str.push(unicode_to_string([s[i]]))
            i+=1
    out.push(concat_string(curr_str,""))
    return out, nil

def decode_int(s:string) -> int, string?:
    let a,b=string_to_int(s)
    if not b:
        return 0, "Could not parse integer: {s}"
    return a, nil


def encode_string(s:string) -> string:
    return encode_string(string_to_unicode(s))
def encode_string(s:[int]) -> string:
    let out=[]
    for(s) c:
        if c=='\\':
            out.push("\\\\")
        elif c=='\"':
            out.push("\\\"")
        else:
            out.push(unicode_to_string([c]))
    return "\""+concat_string(out,"")+"\""


// print(encode_string("a\"b\\c"))
// print(encode_string(""))

def encode_csv(strings) -> string:
    let out=[]
    for(strings) s_raw:
        let s =
        if s_raw is string:
            string_to_unicode(s_raw)
        else://[int]
            s_raw
        let curr_str=[]
        for(s) c:
            if c=='\\':
                curr_str.push("\\\\")
            elif c==',':
                curr_str.push("\\,")
            else:
                curr_str.push(unicode_to_string([c]))
        out.push(concat_string(curr_str,""))
        out.push(",")
    if length(out)>0:
        out.remove(length(out)-1,1)
    return concat_string(out,"")

// print(encode_csv(["a","b"]))
// print(encode_csv([[65,66],[67,68]]))

//num, len of num, error
def escaped_num(s:string,e_left:int='{', e_right:int='}',start:int=0) -> int, int, string?:
    return escaped_num(string_to_unicode(s),e_left,e_right,start)
def escaped_num(s:[int],e_left:int='{', e_right:int='}',start:int=0) -> int, int, string?:
    var i=start
    while i<length(s):
        if s[i]==e_left:
            if i+1>=length(s):
                return 0, 0, "unexpected end of string"
            if s[i+1]==e_right:
                return 0, 0, "empty escape sequence"
            var j=i+1
            while j<length(s):
                if s[j]==e_right:
                    var num=0
                    for(j-i-1) d:
                        num+=pow(10,d)*(s[j-d-1]-48)
                    return num, j-i-1, nil
                elif s[j]<48 or s[j]>57:
                    return 0, 0, "unexpected character in escape sequence: {s[j]}"
                j+=1
            return 0, 0, "unexpected end of escape sequence"
        i+=1
    return 0, 0, "unexpected end of string"


// test3(fn():escaped_num("\{1\}"))
// test3(fn():escaped_num("lkfglk2g5oe6kra2224blll3",'a','b',6))

//num, len of num
def uninterrupted_num(s:string,start:int=0) -> int, int:
    return uninterrupted_num(string_to_unicode(s),start)

def uninterrupted_num(s:[int],start:int=0) -> int, int:
    var i=start
    while 1:
        if i>=length(s) or s[i]<48 or s[i]>57:
            var num=0
            for(i-start) d:
                num+=pow(10,d)*(s[i-d-1]-48)
            return num, i-start
        i+=1
// test2(fn():uninterrupted_num("12345"))
// test2(fn():uninterrupted_num("ab 34 5",3))
// test2(fn():uninterrupted_num("ab 34 5",2))

def is_num(char:int) -> int:
    return char>=48 and char<=57
// print(is_num('a'))


def starts_with(s:string, prefix:string) -> bool:
    if length(s)<length(prefix):
        return false
    return substring(s,0,length(prefix))==prefix

def safe_slice(v, start:int, len:int):
    if start<0:
        len+=start
        start=0
    if start>=length(v):
        len-=start-length(v)
        start=length(v)
    if start+len>length(v):
        len=length(v)-start
    if len<0:
        len=0
    return slice(v,start,len)
// print(safe_slice([1,2,3],7,1))

def strip(s:string) -> string:
    return strip(string_to_unicode(s))
def strip(s:[int]) -> string:
    var start=0
    var end=length(s)
    while start<end and (s[start]==32 or s[start]==9 or s[start]==10 or s[start]==13):
        start+=1
    while end>start and (s[end-1]==32 or s[end-1]==9 or s[end-1]==10 or s[end-1]==13):
        end-=1
    return unicode_to_string(safe_slice(s,start,end-start))