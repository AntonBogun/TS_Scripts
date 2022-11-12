//implied start at root
def foreach_cell(lambda) -> void: //lambda based on current cell
    for(ts_num_children()) n:
        ts_goto_child(n)
        lambda()
        if(ts_num_children()):
            foreach_cell(lambda)
        ts_goto_parent()

let gray=30.0/255.0
let d_grid=160.0/255.0 //default grid color
ts_set_background_color(xyzw_f{gray,gray,gray,0.0})
ts_set_border_color(xyzw_f{d_grid,d_grid,d_grid,0.0})
def do_things():
    ts_set_text_color(xyzw_f{1.0,1.0,1.0,0.0})
    ts_set_border_color(xyzw_f{d_grid,d_grid,d_grid,0.0})
    //ts_set_cell_color(xyzw_f{0.0,0.0,0.0,0.0})

foreach_cell(fn(): do_things())

ts_set_status_message("done stuff")