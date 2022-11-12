import std
import print_f
import selection
import navigation
import modify
import bugs

def main():
    let l_l=[]
    write_bookmarks(l_l)
    write_log(l_l)

//requires 2x1 selection, left is bookmark location (request if empty), right is write location
//!actually relies on the top being a folder
def write_bookmarks(l_l):
    let python="scripts\\python"//doesn't need the entire path, as subprocess starts from the TS directory
    var error=is_nxm_selection(2,1)
    if error:
        print_f(l_l,error)
        return
    // error=is_top_left_selection()
    // if error:
    //     print_f(l_l,error)
    //     return
    ts_goto_selection()//top left corner
    var bookmarkdir=ts_get_text()
    if length(bookmarkdir)==0:
        bookmarkdir=ts_get_filename_from_user(0)
        ts_set_text(bookmarkdir)
    let status, buffer = launch_subprocess(["python", python+"\\bookmarkparse.py",bookmarkdir])
    move_in_selection(1,0)
    delete_subgrid()
    ts_create_grid(2,1)
    ts_goto_child(0)
    print_f(l_l,"status: {status}")
    print_f(l_l,"buffer: {buffer}")
    // print_f(l_l,["output","output"])
    // print_f(l_l,read_file(buffer))
    // print_f(l_l,"output")
    let data=read_file(buffer)
    if data == nil:
        print_f(l_l,"error: buffer read error")
        return
    if data:
        //!replace with tokenize(data,"\n","\r") when bug goes away
        let cmnds=map(tokenize(data,"\n",""),fn(x):strip_whitespace(x,"\r"))//carriage return in windows
        // print_f(l_l,cmnds)
        var state=0 //0=expect name in quotes or ], 1=expect [ or url
        let positions=[selection_pos_y()] //stack of current vertical positions in grids
        for(cmnds) c:
            if state==0:
                if c=="]":
                    positions.pop()
                    ts_goto_parent()
                    positions[length(positions)-1]+=1
                elif (c.substring(0,1)=="\"" //"
                    and 
                    // (c.substring(length(c)-2,2)=="\"\r" or //bug in tokenize
                    // c.substring(length(c)-1,1)=="\""
                    // )):
                    c.substring(length(c)-1,1)=="\"" //"
                    ):
                    if top(positions)!=0:
                        append_row(0,top(positions))
                    let name=c.substring(1,length(c)-2)
                    move_in_grid(0,top(positions))
                    ts_set_text(name)
                    state=1
                else:
                    print_f(l_l,"Error: expected name in quotes but got {c}, "+
                    "beginning = {string_to_unicode(c.substring(0,1))}, "+
                    "end = {string_to_unicode(c.substring(length(c)-1,1))}")
                    return
            else:
                if c=="[":
                    state=0
                    positions.push(0)
                    ts_create_grid(2,1)
                    ts_goto_child(0)
                else:
                    state=0
                    move_in_grid(1,top(positions))
                    ts_set_text(c)//url
                    positions[length(positions)-1]+=1
                





        return
main()

// ts_goto_selection()
// ts_set_text(ts_get_filename())
// print(tokenize(",a,,",",",""))
// let c="\"hello\""
// print(c.substring(0,1)=="\"" and c.substring(length(c)-1,1)=="\"")
// print(c.substring(c.length-1,1))

