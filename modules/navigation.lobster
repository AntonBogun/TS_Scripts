import parse
def move_in_selection(x:int,y:int) -> void:
    ts_goto_parent()
    let size,pos=ts_selection()
    ts_goto_column_row(pos[0]+x,pos[1]+y)

def move_in_grid(x:int,y:int) -> void:
    ts_goto_parent()
    ts_goto_column_row(x,y)

def goto_subcell(x:int,y:int) -> void:
    ts_goto_column_row(x,y)

//implied start at root
def foreach_cell(lambda) -> void: //lambda based on current cell
    for(ts_num_children()) n:
        ts_goto_child(n)
        lambda()
        ts_goto_parent()
def foreach_cell_depth(lambda) -> void: //lambda based on current cell
    for(ts_num_children()) n:
        ts_goto_child(n)
        lambda()
        if(ts_num_children()):
            foreach_cell_depth(lambda)
        ts_goto_parent()

def current_grid_size() -> int, int:
    if (not ts_has_parent()):
        return 1,1
    let curr_pos=ts_current_pos()
    ts_goto_parent()
    let size=ts_num_columns_rows()
    ts_goto_column_row(curr_pos[0],curr_pos[1])
    return size[0],size[1]

def current_pos() -> int, int:
    let _pos=ts_current_pos()
    return _pos[0],_pos[1]

def subgrid_size() -> int, int:
    let size=ts_num_columns_rows()
    return size[0],size[1]

def has_subgrid() -> bool:
    return ts_num_children()!=0


//+3,4; to move to 3rd,4th cell in subgrid
//=3,4; to move to 3rd,4th cell in current grid
//- to move to parent
//! to move to root
//> to move to selection
def ts_goto(cmnds_raw:string,toerr:int=1,lim_x:int=-1,lim_y:int=-1) -> string?:
    let cmnds=string_to_unicode(cmnds_raw)
    if lim_x==-1 or lim_y==-1:
        lim_x,lim_y=current_grid_size()
    var i=0
    while i<length(cmnds):
        let c=cmnds[i]
        if c=='+':
            let n,l_n,err=escaped_num(cmnds,'+',',',i)
            if err: return err
            i+=l_n+1//skip '+'
            let n2,l_n2,err2=escaped_num(cmnds,',',';',i)
            if err2: return err2
            i+=l_n2+2//skip ';'
            if not has_subgrid() and toerr:
                return "no subgrid to move in on char: {i}"
            lim_x,lim_y=current_grid_size()
            if n>=lim_x or n2>=lim_y and toerr:
                return "column/row limit exceeded on char: {i}"
            ts_goto_column_row(n,n2)
        elif c=='-':
            if not ts_has_parent() and toerr:
                return "no parent to move to on char: {i}"
            ts_goto_parent()
            lim_x,lim_y=current_grid_size()
            i+=1
        elif c=='=':
            let n,l_n,err=escaped_num(cmnds,'=',',',i)
            if err: return err
            i+=l_n+1//skip '+'
            let n2,l_n2,err2=escaped_num(cmnds,',',';',i)
            if err2: return err2
            i+=l_n2+2//skip ';'
            if n>=lim_x or n2>=lim_y and toerr:
                return "column/row limit exceeded on char: {i}"
            move_in_grid(n,n2)
        elif c=='!':
            ts_goto_root()
            lim_x,lim_y=current_grid_size()
            i+=1
        elif c=='>':
            if not ts_has_selection() and toerr:
                return "no selection to move in on char: {i}"
            ts_goto_selection()
            i+=1
        else:
            i+=1
    return nil