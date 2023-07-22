//typically define "let l_l=[]", and then "print_f(l_l,x)"
//at the end of the file, "write_log(l_l)"

def print_f(log_list:[string],str,end="\n") -> void:
    if str is string:
        push(log_list,str+end)
    if str is [string]:
        for(str) s:
            push(log_list,s+end)
    if str is int:
        push(log_list,str+end)
    if str is float:
        push(log_list,str+end)

//!FIX: breaks when no \ or / in path
//**never forget unicode_to_string
//save to the same folder as the cts file. works (?) with both \ and /
def write_log(log_list:[string],name="log.txt") -> int:
    let curr=ts_get_filename()
    let pos=max(find_string_reverse(curr,"\\"),find_string_reverse(curr,"/"))
    let path=substring(curr,0,pos)+unicode_to_string([curr[pos]])+name
    // ts_set_status_message("Saving log to {substring(curr,0,pos)};{curr[pos]};{name}") //GOD DAMN ASCII
    // ts_set_status_message("{[path]}")//!fix when relative/full is inferred
    let success=write_file(path,concat_string(log_list,""),0,1)
    return success

def write_log(str:string,name="log.txt") -> int:
    return write_log([str],name)