import print_f
import python
import navigation
import parse
import selection
import modify

def main():
    let l_l=[]
    factorio(l_l)
    write_log(l_l)
main()

//for now obsolete since the https://mods.factorio.com/mod/factoryplanner exists
//but gives an example of how python interface would be used
def factorio(l_l):
    let api_location="scripts\\factorio"//doesn't need the entire path, as subprocess starts from the TS directory
    var err=is_single_cell()
    if(err):
        print_f_and_status(l_l,err)
        return
    ts_goto_selection()
    while(ts_has_parent()):
        if starts_with(ts_get_text(),"FACTORIO_FILE="):
            break
        ts_goto_parent()
    if not ts_has_parent():
        print_f_and_status(l_l,"\"FACTORIO_FILE=\" parent cell not found")
        return
    err=ts_write_from_preset("<*1,1;",[])
    if err:
        print_f(l_l,err)
        return
    var i=0
    if ts_get_text()=="SAVE":
        i=1
    else:
        ts_set_text("LOAD")
    ts_goto_parent()
    connect_python(l_l,"{api_location}\\factorio.py",i)
