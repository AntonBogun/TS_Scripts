import print_f
import navigation
import parse
import selection
import modify
// ts_goto_selection()
// ts_set_text("hello")
// find first cell with EVAL text




def main():
    var x,y=subgrid_size()
    ts_goto_child(0)
    let exec_code=[]::string
    find_eval(x,y)
    build_recursive_code(exec_code)
    exec_code.push("\"0\"")
    ts_goto_selection()
    let exec_text=concat_string(exec_code,"\n")
    let a,b=compile_run_code(exec_text,[])
    ts_set_status_message("return {a} ; message {b}")

def find_eval(x,y):
    for(y)j:
        for(x)i:
            move_in_grid(i,j)
            if(ts_get_text()=="EVAL"):
                if(not has_subgrid()):
                    ts_set_status_message("no subgrid")
                    return from main
                else:
                    ts_goto_child(0)
                    return
    ts_set_status_message("no eval found")
    return from main
def build_recursive_code(exec_code,depth=0)->void:
    let x,y=current_grid_size()
    for(y)j:
        for(x)i:
            move_in_grid(i,j)
            if(length(ts_get_text())):
                exec_code.push(repeat_string(" ",depth)+ts_get_text())
            if(has_subgrid()):
                ts_goto_child(0)
                build_recursive_code(exec_code,depth+1)
                ts_goto_parent()
    return

main()
