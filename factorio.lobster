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
    let strlist=["categories","id","name","icon"]//3
    strlist.append_into(["images","img","id","used_hex"])//7
    strlist.append_into(["items","category","img","id","icon","row","name","data"])//15
    strlist.append_into(["machine","drain","number","type","burner","usage","number","fuelCategories","none"])//24
    strlist.append_into(["fuel","category","name","value","number","belt","speed","number","notes"])//33
    strlist.append_into(["recipes","category","img","id","icon","row","name","time"])//41
    strlist.append_into(["in","1","input","out","1","output","producers","id_1","data","isMining","true","cost","100","notes"])//55
    var preset="<*2,3;*0*<*3,1;*1*,*2*,*3*>,*4*<*3,1;*5*,*6*,*7*>[1]"
    preset+="*8* <*8,1;*9*cat,*10*img,*11*id,*12*icon,*13*row,*14*name,*15*data"
    preset+="<"
    preset+="*1,3; *16* <*2,5; FORGOT speed-number *17*,*18*;*19*,*20*;*21*,*22*;*31*,*22* speed-number here;*23*,<*1,1;*24*> >;"
    preset+="*25* <*2,2;*26*,*27*;*28*,*29*> ;   *30* <*2,1;*31*,*32*> "
    preset+=">,"
    preset+=" *33*notes >"
    preset+="[2]*34*  <*12,1; *35*cat,*36*img,*37*id,*38*icon,*39*row,*40*name,*41*time, "
    preset+="*42* in <*2,1;*43*,*44*>, *45* out <*2,1; *46*,*47*>, *48*producers <*1,1;*49*>, *50*data <*2,2; *51*,*52*;*53*,*54*>, *55* >"

    err=ts_write_from_preset(preset,strlist)
    if err:
        print_f(l_l,err)
        return
    //dump images in images section
    move_in_grid(1,0)
    var x,y=subgrid_size()
    ts_goto_child(0)
    for (y-1) __y:
        let _y=__y+1
        move_in_grid(1,_y)
        let _id=strip(ts_get_text())
        if length(_id)>0:
            let _file=file_dir+"\\"+_id+".png"
            move_in_grid(0,_y)
            ts_save_image_to_file(_file)
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