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
        if starts_with(ts_get_text(),"USE_FACTORIO="):
            break
        ts_goto_parent()
    if not ts_has_parent():
        print_f_and_status(l_l,"\"USE_FACTORIO=\" parent cell not found")
        return
    let file_dir=substring(ts_get_text(),length("USE_FACTORIO="),-1)
    if length(file_dir)==0:
        print_f_and_status(l_l,"No file directory specified")
        return
    print_f(l_l,"File directory: "+file_dir)
    var to_write_preset=1
    if has_subgrid():
        ts_goto_child(0)
        to_write_preset= not (ts_get_text()=="LOADED")
        ts_goto_parent()
    if to_write_preset:
        let preset="<*2,6;*0*,*1*<*1,4;*2*;*3*;*4*;*5*>;,*6*<*4,1;*7*,*8*,*9*,*10*>;,*11*<*4,1;*12*,*13*,*14*,*15*>;,*16*<*9,1;*17*,*18*,*19*,*20*,*21*,*22*,*23*,*24*<*1,3;*25*<*2,5;*26*,*27*;*28*,*29*;*30*,*31*;*32*,*33*;*34*,<*1,1;*35*>>;*36*<*2,2;*37*,*38*;*39*,*40*>;*41*<*2,1;*42*,*43*>>,*44*>;,*45*<*13,1;*46*,*47*,*48*,*49*,*50*,*51*,*52*,*53*,*54*<*2,2;*55*,*56*;*57*,*58*>,*59*<*2,1;*60*,*61*>,*62*<*1,2;*63*;*64*>,*65*<*2,3;*66*,*67*;*68*,*69*;*70*,*71*>,*72*>;,*73*<*19,1;*74*,*75*,*76*,*77*,*78*,*79*,*80*,*81*,*82*<*1,3;*83*<*2,5;*84*,*85*;*86*,*87*;*88*,*89*;*90*,*91*;*92*,<*1,1;*93*>>;*94*<*2,2;*95*,*96*;*97*,*98*>;*99*<*2,1;*100*,*101*>>,*102*,*103*,*104*,*105*,*106*,*107*<*2,2;*108*,*109*;*110*,*111*>,*112*<*2,1;*113*,*114*>,*115*<*1,2;*116*;*117*>,*118*<*2,3;*119*,*120*;*121*,*122*;*123*,*124*>,*125*>>"
        let strlist=["LOADED","instructions:","if top left cell is not \"LOADED\", preset will be regenerated","if \"num\" cell is \"#\" then that row will be ignored.","If name is empty it is autofilled with id. ","in mix, empty or \"#\" icon disables icon, empty item_cat disables the item, empty recipe_cat disables the recipe","categories","num","id","name","icon","images","num","img","id","used_hex","items","num","category","img","id","icon","row","name","data","machine","drain","number","type","burner","usage","number","speed","number","fuelCategories","none","fuel","category","name","value","number","belt","speed","number","notes","recipes","num","category","img","id","icon","row","name","time","in","1","input","2","input2","out","1","output","producers","id_1","id_2","data","isMining","true","cost","100","burner","4","notes","mix","num","id","icon","used_hex","item_cat","iconid","row","name","data","machine","drain","number","type","burner","usage","number","speed","number","fuelCategories","none","fuel","category","name","value","number","belt","speed","number","recipe_cat","iconid","row","name","time","in","1","input","2","input2","out","1","output","producers","id_1","id_2","data","isMining","true","cost","100","burner","4","notes"]
        err=ts_write_from_preset(preset,strlist)
        if err:
            print_f(l_l,err)
            return
    else:
        //scan through cells and dump images in "images" or "mix" cells
        //dump images in images section
        let top_len=ts_num_children()
        for (top_len) i:
            ts_goto_child(i)

            //set num 
            let top_text=ts_get_text()
            if (top_text=="images" or top_text=="mix" or top_text=="items" or top_text=="recipes" or top_text=="categories") and has_subgrid():
                let x,y=subgrid_size()
                let req_size=if (top_text=="images" or top_text=="categories"): 4 else: if top_text=="mix": 18 else: if top_text=="items": 8 else: 12
                if x<req_size:
                    print_f_and_status(l_l,"Not enough columns in {top_text} section: {x} < {req_size}")
                    return
                var count=0
                ts_goto_child(0)
                for (y-1) __y:
                    let _y=__y+1
                    move_in_grid(0,_y)
                    let t=ts_get_text()
                    if t!="#":
                        count+=1
                        ts_set_text("{count}")
                ts_goto_parent()

            if (top_text=="images" or top_text=="mix") and has_subgrid():
                let x,y=subgrid_size()
                ts_goto_child(0)
                if top_text=="images":
                    for (y-1) __y:
                        let _y=__y+1
                        move_in_grid(0,_y)
                        if ts_get_text()!="#":
                            move_in_grid(2,_y)
                            let _id=strip(ts_get_text())
                            if length(_id)>0:
                                let _file=file_dir+"\\"+_id+".png"
                                move_in_grid(1,_y)
                                if not ts_has_image():
                                    move_in_grid(0,_y)
                                    print_f_and_status(l_l,"No image found in images grid {i} at row {ts_get_text()}")
                                    return
                                ts_save_image_to_file(_file)
                            else:
                                move_in_grid(0,_y)
                                print_f_and_status(l_l,"No id found in images grid {i} at row {ts_get_text()}")
                                return
                else:
                    for (y-1) __y:
                        let _y=__y+1
                        move_in_grid(0,_y)
                        if ts_get_text()!="#":
                            move_in_grid(2,_y)
                            if not ts_has_image():
                                ts_set_text("#")
                            if ts_get_text()!="#":
                                move_in_grid(1,_y)
                                let _id=strip(ts_get_text())
                                if length(_id)>0:
                                    let _file=file_dir+"\\"+_id+".png"
                                    move_in_grid(2,_y)
                                    if not ts_has_image():
                                        move_in_grid(0,_y)
                                        print_f_and_status(l_l,"No image found in images grid {i} at row {ts_get_text()}")
                                        return
                                    ts_save_image_to_file(_file)
                                else:
                                    move_in_grid(0,_y)
                                    print_f_and_status(l_l,"No id found in images grid {i} at row {ts_get_text()}")
                                    return
                ts_goto_parent()
            
            ts_goto_parent()

        //dump the entire grid
        // let lines=[]::string
        // dump_cell_data(lines,5) //just enough to reach fuelCategories: [none]
        // let printoutfile=file_dir+"\\printout.txt"
        // let toprintout=concat_string(lines,"")
        // let success=write_lines(printoutfile,[toprintout])
        // if not success and length(toprintout):
        //     print_f_and_status(l_l,"Could not write to file {encode_string(printoutfile)}")
        //     return

        var i=0
        connect_python(l_l,"{api_location}\\factorio.py",i)



// def factorio(l_l):
//     let api_location="scripts\\factorio"//doesn't need the entire path, as subprocess starts from the TS directory
//     var err=is_single_cell()
//     if(err):
//         print_f_and_status(l_l,err)
//         return
//     ts_goto_selection()
//     while(ts_has_parent()):
//         if starts_with(ts_get_text(),"FACTORIO_FILE="):
//             break
//         ts_goto_parent()
//     if not ts_has_parent():
//         print_f_and_status(l_l,"\"FACTORIO_FILE=\" parent cell not found")
//         return
//     err=ts_write_from_preset("<*1,1;",[])
//     if err:
//         print_f(l_l,err)
//         return
//     var i=0
//     if ts_get_text()=="SAVE":
//         i=1
//     else:
//         ts_set_text("LOAD")
//     ts_goto_parent()
//     connect_python(l_l,"{api_location}\\factorio.py",i)