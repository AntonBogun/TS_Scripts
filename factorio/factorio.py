import argparse
import os
import json
def escape_csv(s):
    # \ as escape character, escape " and , and \
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace(",", "\\,")

# Create the argument parser
parser = argparse.ArgumentParser()

# Add the integer argument
parser.add_argument('-i', '--integer', type=int, help='info')

# Parse the arguments
args = parser.parse_args()

class Cell:
    def __init__(self, text="", has_grid=False, gridsize=(0,0)):
        self.text=text
        self.has_grid=has_grid
        self.gridsize=gridsize
        self.grid=[Cell() for i in range(gridsize[0]*gridsize[1])]#stored row-wise
        self.has_image=False
        self.imagepath=""
    def recursive_build(self, data):
        if len(data)==0:
            raise Exception("data is empty")
        if len(data)==1:
            self.text=data[0]
            return
        if len(data)!=4:
            raise Exception(f"invalid data length: {len(data)}")
        self.text=data[0]
        self.has_grid=True
        self.gridsize=(data[1],data[2])
        if len(data[3])!=data[1]*data[2]:
            raise Exception(f"invalid grid data length: {len(data[3])}")
        self.grid=[Cell() for i in range(data[1]*data[2])]
        for i in range(data[1]*data[2]):
            self.grid[i].recursive_build(data[3][i])
    
    
    def add_image(self, path):
        self.has_image=True
        self.imagepath=path
    def clear_image(self):
        self.has_image=False
        self.imagepath=""
    
    def get_row(self, index):
        if not self.has_grid:
            raise Exception("cell has no grid")
        if index<0 or index>=self.gridsize[1]:
            raise Exception("index out of range")
        return [self.grid[index*self.gridsize[0]+i] for i in range(self.gridsize[1])]
    #write in the current grid
    #syntax: *n*... - use text in n index from str_list (n must only be number chars)
    #,... - go to next column
    #;... - jump to beginning of next row
    #(n)...- jump to n indexed column in the current row
    #[n]... - jump to n indexed row, keep current column
    #<n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is overwritten
    #<?n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is resized accordingly if exists
    #<*n,m;...> - make subgrid of size (n,m) in current cell, previous subgrid is set at least to size (n,m)
    ##n#... - use text in n index from str_list for setting image
    #%... - remove image
    def to_write(self):
        strings=[]
        commands=[]
        if not self.has_image:
            commands.append("%")#no need to put % in recursive since it overwrites anyway
        def inner_recurse(cell):
            if len(cell.text)!=0:
                strings.append(cell.text)
                commands.append(f"*{len(strings)-1}*")
            if cell.has_image:
                strings.append(cell.imagepath)
                commands.append(f"#{len(strings)-1}#")
            if cell.has_grid:
                commands.append(f"<{cell.gridsize[0]},{cell.gridsize[1]};")
                for i in range(cell.gridsize[1]):
                    for j in range(cell.gridsize[0]):
                        inner_recurse(cell.grid[i*cell.gridsize[0]+j])
                        if j!=cell.gridsize[0]-1:
                            commands.append(",")
                    if i!=cell.gridsize[1]-1:
                        commands.append(";")
                commands.append(">")
        inner_recurse(self)
        return ("".join(commands),strings)

#get current path
script_dir = os.path.dirname(os.path.realpath(__file__))
communicate_file = os.path.join(script_dir, "communicate.txt")
out_file = os.path.join(script_dir, "out.txt")
temp_file = os.path.join(script_dir, "temp.txt")
#! must use newline="\n" to avoid windows newline
#clear communicate_file
with open(communicate_file, "w",newline="\n") as f:
    if args.integer==2:
        # with open(out_file, "r") as f2:
        lines=open(out_file, "r").readlines()
        last_i=int(open(temp_file, "r").read())
        data=json.loads(lines[0])
        # open(temp_file, 'w').write(str(data))
        tree=Cell()
        tree.recursive_build(data)
        to_write=tree.to_write()
        open(temp_file, 'w').write(str(to_write))
        f.write("exit")
        print(communicate_file,end="")
    else:
        f.write(f"setout,{escape_csv(out_file)}\nsetinfo,2\n")
        if args.integer==1:
            f.write("getdata,0\n")
        elif args.integer==0:
            f.write("getdata,-1\n")
        f.write("call")
        open(temp_file, 'w').write(str(args.integer))
        print(communicate_file,end="")
