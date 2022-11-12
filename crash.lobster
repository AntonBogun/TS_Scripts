def is_selecting_single_cell(): //success?, fail string    
    if(ts_has_selection()):
        let size,pos=ts_selection()
        if(size.x+size.y!=1):
            if(size.x+size.y==2):
                ts_goto_selection()
                return 1, ""
            else:
                return 0, "Select a single cell"
        else:
            return 0, "Border selection"
    else:
        return 0, "No selection"



let is_selecting_single_cell, fail_string=is_selecting_single_cell()
if(is_selecting_single_cell):
    let num,success=string_to_int(ts_get_text())
    if(success):
        for(num) i:
            ts_goto_parent()
        let grid_size=ts_num_columns_rows()
        ts_delete(xy_i{0,0},grid_size)
    else:
        ts_set_status_message("not a number")
else:
    ts_set_status_message(fail_string)