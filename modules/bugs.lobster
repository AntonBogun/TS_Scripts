//this is intended to be a selection of functions that replicate the functionality
//that should be available in the builtin functions, but is not there due to bugs or missing features
//before using please check if the desired functionality has already been implemented
import std

def str_to_chars(str:string) -> [string]:
    return map(string_to_unicode(str), fn(x):unicode_to_string([x]))

def strip_whitespace(s:string,whitespace=" ") -> string:
    var i=0
    while i<length(s) and find_string(whitespace,s.substring(i,1))>=0:
        i+=1
    if i==length(s):
        return ""
    var j=length(s)-1
    while j>=0 and find_string(whitespace,s.substring(j,1))>=0:
        j-=1
    return s.substring(i,j-i+1)

// print(strip_whitespace("aabedjjefkkleoebbebadaeba","eab"))
// print(tokenize("Hello,\r\nworld","\n","\r"))
// print(map(tokenize("Hello,\r\nworld","\n",""),fn(x):strip_whitespace(x,"\r"))) //fixed tokenize bug


//arguably redundant, but showcases that this is missing
// def u_to_str(u):
//     if u is [int]:
//         return unicode_to_string(u)
//     if u is int:
//         return unicode_to_string([u])
//     return ""