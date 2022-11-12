def move_in_selection(x:int,y:int) -> void:
    ts_goto_parent()
    let size,pos=ts_selection()
    ts_goto_column_row(pos[0]+x,pos[1]+y)

def move_in_grid(x:int,y:int) -> void:
    ts_goto_parent()
    ts_goto_column_row(x,y)

//implied start at root
def foreach_cell(lambda) -> void: //lambda based on current cell
    for(ts_num_children()) n:
        ts_goto_child(n)
        lambda()
        if(ts_num_children()):
            foreach_cell(lambda)
        ts_goto_parent()
