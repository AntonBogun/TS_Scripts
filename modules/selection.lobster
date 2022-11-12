//success, fail string?
def is_single_cell() -> string?: //error?
    if(ts_has_selection()):
        let size,pos=ts_selection()
        if(not (size.x==0 or size.y==0)):
            if(size.x+size.y==2):
                return nil
            else:
                return "Error: Select a single cell"
        else:
            return "Error: Border selection"
    else:
        return "Error: No selection"

//false if selection doesn't exist
def is_border_selection() -> int:
    let size,pos=ts_selection()
    return not(size.x==0 or size.y==0)

def is_nxm_selection(n:int,m:int) -> string?: //error?
    if(ts_has_selection()):
        let size,pos=ts_selection()
        if(size.x==n and size.y==m):
            return nil
        else:
            return "Error: Select {n}x{m} cell(s)"
    else:
        return "Error: No selection"

def is_top_left_selection() -> string?: //error?
    if(ts_has_selection()):
        let size,pos=ts_selection()
        if(pos.x==0 and pos.y==0):
            return nil
        else:
            return "Error: Select top left cell"
    else:
        return "Error: No selection"

// def read_selection_rel_pos_int(pos:xy_i,x:int,y:int) -> int, int:
//     ts_goto_parent()
//     ts_goto_column_row(pos.x+x,pos.y+y)
//     return string_to_int(ts_get_text())

def selection_pos_x() -> int:
    if(ts_has_selection()):
        let size,pos=ts_selection()
        return pos.x
    else:
        return 0

def selection_pos_y() -> int:
    if(ts_has_selection()):
        let size,pos=ts_selection()
        return pos.y
    else:
        return 0

def selection_size_x() -> int:
    if(ts_has_selection()):
        let size,pos=ts_selection()
        return size.x
    else:
        return 0

def selection_size_y() -> int:
    if(ts_has_selection()):
        let size,pos=ts_selection()
        return size.y
    else:
        return 0

def is_grid_at_least_nxm(n:int,m:int) -> string?: //error?
    if(ts_has_selection()):
        ts_goto_selection()
        ts_goto_parent()
        let size=ts_num_columns_rows()
        if(size.x>=n and size.y>=m):
            return nil
        else:
            return "Error: Grid is not at least {n}x{m}"
    else:
        return "Error: No selection"