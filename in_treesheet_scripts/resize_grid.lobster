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
