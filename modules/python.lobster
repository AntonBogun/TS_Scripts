import print_f
import parse
import file
import navigation
import modify



//opcodes:
//setout,filename
//getdata,depth
//call
//setinfo,i
//goto,goto_str
//write,preset,write_str*
//exit
def connect_python(l_l,filename:string,i:int=0):
    var tocall=1
    var printoutfile=""
    var callnum=0
    while tocall:
        let toprintout=[]
        tocall=0
        var status, buffer = launch_subprocess(["python", filename, "-i", string(i)])
        _py_if_err_print_return(l_l,status,"Python script returned error status {status}: {buffer}",i,-1,callnum,-1,buffer)
        let instructions = read_lines(buffer)
        if instructions:
            var instlen=length(instructions)
            var j=0
            while j<instlen:
                //!this was j=instlen-1, not fun to debug
                if j==instlen-1 and length(instructions[j])==0:
                    break //ignore trailing newline
                let inst,err0=decode_csv(instructions[j])
                _py_if_err_print_return(l_l,err0,i,j,callnum,instlen,buffer)
                let op=inst[0]
                if op=="setout":
                    printoutfile=inst[1]
                elif op=="getdata":
                    let depth,err=decode_int(inst[1])
                    _py_if_err_print_return(l_l,err,i,j,callnum,instlen,buffer)                    
                    let lines=[]::string
                    dump_cell_data(lines,depth)
                    toprintout.push(concat_string(lines,""))
                elif op=="call":
                    tocall=1
                    break
                elif op=="setinfo":
                    let _i,err=decode_int(inst[1])
                    _py_if_err_print_return(l_l,err,i,j,callnum,instlen,buffer)
                    i=_i
                elif op=="goto":
                    _py_if_err_print_return(l_l,ts_goto(inst[1]),i,j,callnum,instlen,buffer)
                elif op=="write":
                    _py_if_err_print_return(l_l,ts_write_from_preset(inst[1],safe_slice(inst,2,length(inst))),i,j,callnum,instlen,buffer)
                elif op=="exit":
                    tocall=0
                    break
                else:
                    _py_if_err_print_return(l_l,"Unknown instruction {op}",i,j,callnum,instlen,buffer)
                j+=1
            if length(printoutfile):
                let success=write_lines(printoutfile,toprintout)
                if not success and length(toprintout):
                    _py_if_err_print_return(l_l,"Could not write to file {encode_string(printoutfile)}",i,j,callnum,instlen,buffer)
            callnum+=1
            if callnum>1000:
                _py_if_err_print_return(l_l,"Too many calls",i,-1,callnum,-1,buffer)
        else:
            _py_if_err_print_return(l_l,"Couldn\'t get instructions",i,-1,callnum,-1,buffer)


def _py_if_err_print_return(l_l,err,ret,i,j,callnum,instlen,buffer):
    if err:
        print_f(l_l,"---Error occured with info {i} at op {j} of call {callnum} of length {instlen} with buffer {buffer}")
        print_f(l_l,ret)
        return from connect_python

def _py_if_err_print_return(l_l,err,i,j,callnum,instlen,buffer):
    if err:
        print_f(l_l,"---Error occured with info {i} at op {j} of call {callnum} of length {instlen} with buffer {buffer}")
        print_f(l_l,err)
        return from connect_python



// recrusivedepth=-1 to infinitely recurse
def dump_cell_data(lines,recrusivedepth:int=0,top=1)->void:
    lines.push("[")
    if not recrusivedepth:
        lines.push("{encode_string(ts_get_text())}")
    else:
        lines.push(encode_string(ts_get_text()))
        if has_subgrid():
            let x,y=subgrid_size()
            lines.push(",{x},{y},[")
            ts_goto_child(0)
            for(y) j:
                for(x) i:
                    if i or j:
                        lines.push(",")
                    move_in_grid(i,j)
                    dump_cell_data(lines,recrusivedepth-1,0)
            ts_goto_parent()
            lines.push("]")
    lines.push("]")



// let a=fn(f): f()
// def f():
//     print("hello")
//     print("world")
// a(fn(): print("hello") == print("world"))
// a(fn: f())

// def a():
//     let c=3
//     def b():
//         print(c)
//     b()
// a()