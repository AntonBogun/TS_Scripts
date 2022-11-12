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



//write in the current grid
//syntax: *n*... - use text in n index from str_list (n must only be number chars)
//,... - go to next column
//;... - jump to beginning of next row
//(n)...- jump to n indexed column in the current row
//[n]... - jump to n indexed row, keep current column
//<n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is overwritten
//every character other than starting characters ('*', ',', ';', '(', '[', '<', '>')
//are ignored within "..." regions
//lim_x,lim_y = size limit in the initial grid
def ts_write_from_preset(cmnds_raw:string,str_list:[string],lim_x:int,lim_y:int) -> string?:
    let cmnds=string_to_unicode(cmnds_raw)
    ts_goto_parent()
    ts_goto_child(0)
    var col,row=0,0
    let limlist=[]//limits and positions lists are used as a stack for nested subgrids
    let poslist=[]
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
            let n,l_n,err=escaped_num(cmnds,'<',',',i)
            if err: return err
            i+=l_n+1//end on ','
            let m,l_m,err2=escaped_num(cmnds,',',';',i)
            if err2: return err2
            i+=l_m+2//skip ';'
            limlist.append_into([lim_x,lim_y])
            poslist.append_into([col,row])
            lim_x,lim_y=n,m
            col,row=0,0
            delete_subgrid()
            ts_create_grid(n,m)
            ts_goto_child(0)
        elif c=='>':
            i+=1//skip '>'
            if length(limlist)==0:
                return "unmatched \'>\' on char: {i}"
            lim_y,lim_x=limlist.pop(),limlist.pop()
            row,col=poslist.pop(),poslist.pop()
            ts_goto_parent()
        else:
            i+=1
    return nil

private let assert0=[1,2]
private let assert2,assert1=assert0.pop(),assert0.pop()
assert(assert1==1 and assert2==2)//!in case order of multi argument evaluation is changed