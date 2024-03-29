import navigation
import parse
import selection
import modify
import print_f

let str_list=[]::string
let command=[]::string
def f() -> void:
  let t=ts_get_text()
  if length(t)>0:
    command.push("*{length(str_list)}*")
    str_list.push(encode_string(t))
  let x,y=subgrid_size()
  if has_subgrid():
    command.push("<*"+x+","+y+";")
    for (y) i:
      for (x) j:
        goto_subcell(j,i)
        f()
        ts_goto_parent()
        if j<x-1:
          command.push(",")
      if i<y-1:
        command.push(";")
    command.push(">")
  return

f()
let l_l=[]
print_f(l_l,"ran preset dump")
print_f(l_l,"str_list: ")
let str_list_str="["+concat_string(str_list,",")+"]"
print_f(l_l,str_list_str)
print_f(l_l,"command: ")
let command_str="\""+concat_string(command,"")+"\""
print_f(l_l,command_str)
write_log(l_l)
if 1:
  return "wrote log"




import modify
let s="*0*<*2,6;*1*,*2*<*1,4;*3*;*4*;*5*;*6*>;,*7*;,*8*<*4,1;*9*,*10*,*11*,*12*>;,*13*<*9,1;*14*,*15*,*16*,*17*,*18*,*19*,*20*,*21*<*1,3;*22*<*2,5;*23*,*24*;*25*,*26*;*27*,*28*;*29*,*30*;*31*,<*1,1;*32*>>;*33*<*2,2;*34*,*35*;*36*,*37*>;*38*<*2,1;*39*,*40*>>,*41*>;,*42*<*13,1;*43*,*44*,*45*,*46*,*47*,*48*,*49*,*50*,*51*<*2,2;*52*,*53*;*54*,*55*>,*56*<*2,1;*57*,*58*>,*59*<*1,2;*60*;*61*>,*62*<*2,3;*63*,*64*;*65*,*66*;*67*,*68*>,*69*>;,*70*<*18,1;*71*,*72*,*73*,*74*,*75*,*76*,*77*,*78*<*1,3;*79*<*2,5;*80*,*81*;*82*,*83*;*84*,*85*;*86*,*87*;*88*,<*1,1;*89*>>;*90*<*2,2;*91*,*92*;*93*,*94*>;*95*<*2,1;*96*,*97*>>,*98*,*99*,*100*,*101*,*102*,*103*<*2,2;*104*,*105*;*106*,*107*>,*108*<*2,1;*109*,*110*>,*111*<*1,2;*112*;*113*>,*114*<*2,3;*115*,*116*;*117*,*118*;*119*,*120*>,*121*>>"
let l=["preset","LOADED","instructions:","if top left cell is not \"LOADED\", preset will be regenerated","if \"num\" cell is \"#\" then that row will be ignored.","If name is empty it is autofilled with id. ","in mix, empty icon disables icon, empty item_cat disables the item, empty recipe_cat disables the recipe","categories","images","num","img","id","used_hex","items","num","category","img","id","icon","row","name","data","machine","drain","number","type","burner","usage","number","speed","number","fuelCategories","none","fuel","category","name","value","number","belt","speed","number","notes","recipes","num","category","img","id","icon","row","name","time","in","1","input","2","input2","out","1","output","producers","id_1","id_2","data","isMining","true","cost","100","burner","4","notes","mix","num","id","icon","used_hex","item_cat","row","name","data","machine","drain","number","type","burner","usage","number","speed","number","fuelCategories","none","fuel","category","name","value","number","belt","speed","number","recipe_cat","icon","row","name","time","in","1","input","2","input2","out","1","output","producers","id_1","id_2","data","isMining","true","cost","100","burner","4","notes"]
ts_write_from_preset(s,l)