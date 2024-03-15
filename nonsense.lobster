
def test_f():
    return 2

def write_log(log_list:[string],name="log.txt"):
    let curr=ts_get_filename()
    // let pos=max(find_string_reverse(curr,"\\"),find_string_reverse(curr,"/"))
    // // let path=substring(curr,0,pos)+unicode_to_string([curr[pos]])+name
    // let path=substring(curr,0,pos)+unicode_to_string([curr[pos]])+name
    // // ts_set_status_message("Saving log to {substring(curr,0,pos)};{curr[pos]};{name}") //GOD DAMN ASCII
    // // ts_set_status_message("{[path]}")//!fix when relative/full is inferred
    // write_file(path,concat_string(log_list,""),0,1)
    ts


let l_l=["eee"]
let a=test_f()
let b="{a}"
write_log(l_l,"log.txt")



import navigation
import parse
import selection
import modify
let x=string_to_int(ts_get_text())
if x<=0 or x>=100:
 return "bad size"
ts_goto_parent()
delete_subgrid()
let text=[]::string
ts_goto_parent()
foreach_cell(fn(): text.push(ts_get_text()))
resize_subgrid(x,max(1,ceiling(float(length(text))/float(x))))
var i=0
let f=fn():
 ts_set_text(text[i])
 i+=1
if i>=length(text):
 return "done"
foreach_cell(f)
