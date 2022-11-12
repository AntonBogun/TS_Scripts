private def test2(f):
    let a,b=f()
    print("{a} {b}")
//for example: test2(fn():func(arg,arg2...))
private def test3(f):
    let a,b,c=f()
    print("{a} {b} {c}")

// def test0()-> int,int:
//     return 1,2
// def test1()-> int,int:
//     return test0()

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

// let a,b=decode_csv("1,2,3\n4,5,6")
// print("{a},{b}")
// print(decode_csv("a,b,c\n1,2"))
// print(decode_csv([48,49,50,',',51]))
// test2(fn():decode_csv(string_to_unicode("1,2,3\n4,5,6")))
// test2(fn():decode_csv(string_to_unicode("\\,,\\,\\\\e,aad")))
// test2(fn():decode_csv(string_to_unicode("\\,,\\,\\e,aad")))


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

