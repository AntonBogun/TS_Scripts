def is_selecting_single_cell(): //success?, fail string    
    if(ts_has_selection()):
        let size,pos=ts_selection()
        if(not (size.x==0 or size.y==0)):
            if(size.x+size.y==2):
                return 1, ""
            else:
                return 0, "Select a single cell"
        else:
            return 0, "Border selection"
    else:
        return 0, "No selection"

def border_selection(): //selection must exist -> bool
    let size,pos=ts_selection()
    return not(size.x==0 or size.y==0)

def delete_subgrid()://of current cell
    let size=ts_num_columns_rows()
    if size.x==0 or size.y==0:
        return
    ts_delete(xy_i{0,0},size)


//NOTE: currently impossible to backtrack to start, as there is no way to know 
//current position in the grid
//todo::change to use lists?
//todo::return to origin upon fail
def try_goto_path(path): //[int] -> fail step (or 0), fail string
    //(0)=up, (1,x,y)=down, (2,x,y)=move to cell, (3)=root, (4)=selection
    //todo::(5,x,y)=relative move
    let l=length(path)
    var i=0
    var step=0
    while(i<l):
        step+=1
        let option=path[i]
        i+=1
        switch option:
            case 0:
                if(not ts_has_parent()):
                    return step, "No parent"
                ts_goto_parent()
            case 1:
                if(l-i<2):
                    return step, "Not enough arguments"
                let col,row=path[i],path[i+1]
                i+=2
                if(ts_num_children()==0):
                    return step, "No children"
                let grid_cap=ts_num_columns_rows()
                if(col>=grid_cap.x or row>=grid_cap.y):
                    return step, "Cell out of range"
                ts_goto_column_row(col,row)
            case 2:
                if(not ts_has_parent()):
                    return step, "Can not move in root"
                ts_goto_parent()
                if(l-i<2):
                    return step, "Not enough arguments"
                let col,row=path[i],path[i+1]
                i+=2
                let grid_cap=ts_num_columns_rows()
                if(col>=grid_cap.x or row>=grid_cap.y):
                    return step, "Cell out of range"
                ts_goto_column_row(col,row)
            case 3:
                ts_goto_root()
            case 4:
                if(not ts_has_selection()):
                    return step, "No selection"
                ts_goto_selection()
            // case 5:
            //     if(l-i<2):
            //         return step, "Not enough arguments"
            //     let r_col,r_row=path[i],path[i+1]
            //     let cur_pos=ts_current_column_row()
            //     i+=2
            //     if(not ts_has_parent()):
            //         return step, "Can not move relative in root"
            //     ts_goto_parent()
            //     let grid_cap=ts_num_columns_rows()
            //     ts_goto_column_row((cur_pos.x+r_col)%grid_cap.x,(cur_pos.y+r_row)%grid_cap.y)
            default:
                return step, "Invalid option"
    return 0, ""


def parse_path(path): //string -> [int] (empty if invalid), fail string
    //..=up, d(x,y)=down, m(x,y)=move to cell, r=root, s=selection
    //todo::r(x,y)=relative move
    let cmnds=tokenize(path,"/"," ")
    let out=[]
    for(cmnds) s:
        if(s==".."):
            push(out,0)
        elif(s=="r"):
            push(out,3)
        elif(s=="s"):
            push(out,4)
        elif(length(s)>1):
            let cmnd=substring(s,0,1)
            let arg=tokenize(substring(s,1,-1),","," ")
            if(cmnd=="d"):
                if(length(arg)!=2):
                    return [],"Incorrect number of arguments for down"
                push(out,1)
                push(out,string_to_int(arg[0]))
                push(out,string_to_int(arg[1]))
            elif(cmnd=="m"):
                if(length(arg)!=2):
                    return [],"Incorrect number of arguments for move"
                push(out,2)
                push(out,string_to_int(arg[0]))
                push(out,string_to_int(arg[1]))
            // elif(cmnd=="r"):
            //     if(length(arg)!=2):
            //         return [],"Incorrect number of arguments for relative move"
            //     push(out,5)
            //     push(out,string_to_int(arg[0]))
            //     push(out,string_to_int(arg[1]))
            else:
                return [],"Invalid command"
        else:
            return [],"Invalid option"
    return out,""

def parse_path(path): //string -> [int] (empty if invalid), fail string
    //..=up, d(x,y)=down, m(x,y)=move to cell, r=root, s=selection
    //todo::r(x,y)=relative move
    let cmnds=tokenize(path,"/"," ")
    let out=[]
    for(cmnds) s:
        if(s==".."):
            push(out,0)
        elif(s=="r"):
            push(out,3)
        elif(s=="s"):
            push(out,4)
        elif(length(s)>1):
            let cmnd=substring(s,0,1)
            let arg=tokenize(substring(s,1,-1),","," ")
            if(cmnd=="d"):
                if(length(arg)!=2):
                    return [],"Incorrect number of arguments for down"
                push(out,1)
                push(out,string_to_int(arg[0]))
                push(out,string_to_int(arg[1]))
            elif(cmnd=="m"):
                if(length(arg)!=2):
                    return [],"Incorrect number of arguments for move"
                push(out,2)
                push(out,string_to_int(arg[0]))
                push(out,string_to_int(arg[1]))
            // elif(cmnd=="r"):
            //     if(length(arg)!=2):
            //         return [],"Incorrect number of arguments for relative move"
            //     push(out,5)
            //     push(out,string_to_int(arg[0]))
            //     push(out,string_to_int(arg[1]))
            else:
                return [],"Invalid command"
        else:
            return [],"Invalid option"
    return out,""

//ts_set_status_message("var is {var}")
//[]!=[]
//f=f() is safe?
let is_selecting_single_cell, fail_string=is_selecting_single_cell()
if(is_selecting_single_cell):
    ts_goto_selection()
    let text=ts_get_text()
    let path,fail_string1=parse_path(text)
    if(length(path)>0):
        let fail,fail_string2=try_goto_path(path)
        if(not fail):
            ts_set_status_message("{ts_get_text()}")
            // ts_set_text("overwrite")
            // delete_subgrid()
        else:
            ts_set_status_message("Goto path failed: {fail_string2}")
    else:
        ts_set_status_message("Parse path failed: {fail_string1}")
else:
    ts_set_status_message(fail_string)