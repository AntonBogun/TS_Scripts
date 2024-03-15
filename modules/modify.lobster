//lobster
import parse
import navigation
def delete_subgrid() -> void://of current cell
    let size=ts_num_columns_rows()
    if size.x==0 or size.y==0:
        return
    ts_delete(xy_i{0,0},size)

def append_row_no_return() -> void:
    ts_goto_parent()
    let size=ts_num_columns_rows()
    ts_insert_row(size.y)

def append_row(posx:int,posy:int) -> void:
    append_row_no_return()
    ts_goto_column_row(posx,posy)

def append_column_no_return() -> void:
    ts_goto_parent()
    let size=ts_num_columns_rows()
    ts_insert_column(size.x)

def append_column(posx:int,posy:int) -> void:
    append_column_no_return()
    ts_goto_column_row(posx,posy)

//since ts_delete_column doesn't exist, we have to delete all cells in the column
def delete_subgrid_column(posx:int) -> void:
    let size=ts_num_columns_rows()
    if size.x==0 or size.y==0:
        return
    ts_delete(xy_i{posx,0},xy_i{1,size.y})

def delete_subgrid_row(posy:int) -> void:
    let size=ts_num_columns_rows()
    if size.x==0 or size.y==0:
        return
    ts_delete(xy_i{0,posy},xy_i{size.x,1})

def resize_subgrid(x:int,y:int) -> void:
    let size=ts_num_columns_rows()
    if size.x==0 or size.y==0:
        ts_create_grid(x,y)
        return
    if x>=size.x:
        for (x-size.x) i:
            ts_insert_column(size.x)
    else:
        for (size.x-x) i:
            delete_subgrid_column(size.x-1-i)
    if y>=size.y:
        for (y-size.y) i:
            ts_insert_row(size.y)
    else:
        for (size.y-y) i:
            delete_subgrid_row(size.y-1-i)

def resize_subgrid_at_least(x:int,y:int) -> void:
    let size=ts_num_columns_rows()
    if size.x==0 or size.y==0:
        ts_create_grid(x,y)
        return
    if x>size.x:
        for (x-size.x) i:
            ts_insert_column(size.x)
    if y>size.y:
        for (y-size.y) i:
            ts_insert_row(size.y)

//write in the current grid
//syntax: *n*... - use text in n index from str_list (n must only be number chars)
//,... - go to next column
//;... - jump to beginning of next row
//(n)...- jump to n indexed column in the current row
//[n]... - jump to n indexed row, keep current column
//<n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is overwritten
//<?n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is resized accordingly if exists
//<*n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is set at least to size (n,m)
//#n#... - use text in n index from str_list for setting image
//%... - remove image
//every character other than starting characters ('*', ',', ';', '(', '[', '<', '>','#','%')
//are ignored within "..." regions
//lim_x,lim_y = size limit in the initial grid
//typical usage:
// let err=ts_write_from_preset
// if err:
//    print_f(l_l,err)
//    return
//! notice: this does not reset the current cell to the top left corner
def ts_write_from_preset(cmnds_raw:string,str_list:[string],lim_x:int=-1,lim_y:int=-1) -> string?:
    if lim_x==-1 and lim_y==-1:
        lim_x,lim_y=current_grid_size()
    let cmnds=string_to_unicode(cmnds_raw)
    var col,row=current_pos()
    var i=0
    while i<length(cmnds):
        let c=cmnds[i]
        if c=='*':
            let n,l_n,err=escaped_num(cmnds,'*','*',i)
            if err: return err
            i+=l_n+2//skip '}'
            if n>=length(str_list):
                return "string list index out of bounds: {n}"
            let s=str_list[n]
            ts_set_text(s)
        elif c==',':
            col+=1
            if col>=lim_x:
                return "column limit exceeded on char: {i}"
            move_in_grid(col,row)
            i+=1
        elif c==';':
            row+=1
            if row>=lim_y:
                return "row limit exceeded on char: {i}"
            col=0
            move_in_grid(col,row)
            i+=1
        elif c=='(':
            let n,l_n,err=escaped_num(cmnds,'(',')',i)
            if err: return err
            i+=l_n+2//skip ')'
            if n>=lim_x:
                return "column limit exceeded on char: {i}"
            col=n
            move_in_grid(col,row)
        elif c=='[':
            let n,l_n,err=escaped_num(cmnds,'[',']',i)
            if err: return err
            i+=l_n+2//skip ']'
            if n>=lim_y:
                return "row limit exceeded on char: {i}"
            row=n
            move_in_grid(col,row)
        elif c=='<':
            if length(cmnds) <= i+1:
                return "unexpected end of string at \'<\'"

            var overwrite=switch cmnds[i+1]:
                case '?': 0
                case '*': 2
                default: 1
            if overwrite!=1:
                i+=1

            let n,l_n,err=escaped_num(cmnds,cmnds[i],',',i)
            if err: return err
            i+=l_n+1//end on ','

            let m,l_m,err2=escaped_num(cmnds,',',';',i)
            if err2: return err2
            i+=l_m+2//skip ';'
            lim_x,lim_y=n,m
            col,row=0,0
            if overwrite==0:
                resize_subgrid(n,m)
            elif overwrite==2:
                resize_subgrid_at_least(n,m)
                lim_x,lim_y=subgrid_size()
            else:
                delete_subgrid()
                ts_create_grid(n,m)
            ts_goto_child(0)
        elif c=='>':
            i+=1//skip '>'
            // if length(limlist)==0:
            //     return "unmatched \'>\' on char: {i}"
            // lim_y,lim_x=limlist.pop(),limlist.pop()
            // row,col=poslist.pop(),poslist.pop()
            ts_goto_parent()
            lim_x,lim_y=current_grid_size()
            col,row=current_pos()
        elif c=='#':
            let n,l_n,err=escaped_num(cmnds,'#','#',i)
            if err: return err
            i+=l_n+2//skip '}'
            if n>=length(str_list):
                return "string list index out of bounds: {n}"
            let s=str_list[n]
            ts_load_image_from_file(s)
        elif c=='%':
            ts_clear_image_in_cell()
            i+=1
        else:
            i+=1
    return nil

private let assert0=[1,2]
private let assert2,assert1=assert0.pop(),assert0.pop()
assert(assert1==1 and assert2==2)//!in case order of multi argument evaluation is changed