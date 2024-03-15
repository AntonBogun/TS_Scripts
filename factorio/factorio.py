import argparse
import os
from collections import namedtuple

def escape_csv(s):
    # \ as escape character, escape " and , and \
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace(",", "\\,")


def weighted_alpha_distance(color1, color2, alpha):
    """Calculate weighted Euclidean distance between two colors, taking alpha into account."""
    return ((color1 - color2) ** 2).sum(axis=-1) * alpha

def update_color(rgb, mask, alpha):
    """Update color based on weighted alpha mean of assigned pixels."""
    if np.sum(mask * alpha) == 0:  # Avoid division by zero
        return rgb[mask].mean(axis=0)  # Fallback to simple mean if all alphas are zero
    return np.average(rgb, axis=0, weights=mask * alpha)


def do_the_colors(image_path):
    # Load the image
    img = Image.open(image_path+".png").convert('RGBA')  # Ensure image is in RGBA format
    img.resize((64, 64)).save(image_path+"_64.png")
    pixels = np.array(img).reshape(-1, 4).astype(np.uint32)
    rgb = pixels[..., :3]
    alpha = pixels[..., 3]
    
    average_color = update_color(rgb, np.ones_like(alpha), alpha)

    A=update_color(rgb, np.arange(len(rgb)) % 2 == 0, alpha)
    B=update_color(rgb, np.arange(len(rgb)) % 2 == 1, alpha)
    for i in range(10):
        bit_mask = weighted_alpha_distance(rgb, A, alpha) < weighted_alpha_distance(rgb, B, alpha)
        _A = update_color(rgb, bit_mask, alpha)
        _B = update_color(rgb, ~bit_mask, alpha)
        if np.all(A == _A) and np.all(B == _B):
            break
        A, B = _A, _B
    # print("took", i, "iterations")
    #sort A and B by distance from 0,0,0
    A_dist = np.linalg.norm(A)
    B_dist = np.linalg.norm(B)
    if A_dist < B_dist:
        A, B = B, A
    
    
    return (average_color, A, B)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*tuple(map(int, rgb)))

def is_rg_hex_valid(s):
    if len(s)!=7:
        return False
    if s[0]!='#':
        return False
    for i in range(1,7,2):
        try:
            if int(s[i:i+2], 16) not in range(256):
                return False
        except:
            return False
    return True

def collect_images(_f):
    import math
    len_f=len(_f)
    n=math.ceil(len_f**0.5)
    img_size=64
    out_img=Image.new("RGBA", (n*img_size,n*img_size))

    #i.e. "-512px 0px"
    offsets=[f"{-(i%n)*img_size}px {-(i//n)*img_size}px" for i in range(len_f)]
    for i, _file in enumerate(_f):
        img=Image.open(_file+"_64.png")
        assert img.size==(64,64), f"image {_file} is not 64x64"
        row=i//n
        col=i%n
        out_img.paste(img, (col*img_size,row*img_size))
    out_img.save("icons.webp", "WEBP")
    return offsets
        

#=====
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
        self.gridsize=namedtuple("GridSize", ["x", "y"])(gridsize[0],gridsize[1])
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
        # self.gridsize=(data[1],data[2])
        self.gridsize=namedtuple("GridSize", ["x", "y"])(data[1],data[2])
        if len(data[3])!=data[1]*data[2]:
            raise Exception(f"invalid grid data length: {len(data[3])}")
        self.grid=[Cell() for i in range(data[1]*data[2])]
        for i in range(data[1]*data[2]):
            self.grid[i].recursive_build(data[3][i])
    def add_grid(self, size):
        self.has_grid=True
        self.gridsize=namedtuple("GridSize", ["x", "y"])(size[0],size[1])
        self.grid=[Cell() for i in range(size[0]*size[1])]
    
    def add_image(self, path):
        self.has_image=True
        self.imagepath=path
    def clear_image(self):
        self.has_image=False
        self.imagepath=""
    
    def get_row(self, index):
        if not self.has_grid:
            raise Exception("cell has no grid")
        if index<0 or index>=self.gridsize.y:
            raise Exception("index out of range")
        return [self.grid[index*self.gridsize.x+i] for i in range(self.gridsize.x)]
    def get_column(self, index):
        if not self.has_grid:
            raise Exception("cell has no grid")
        if index<0 or index>=self.gridsize.x:
            raise Exception("index out of range")
        return [self.grid[i*self.gridsize.x+index] for i in range(self.gridsize.y)]
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
    def to_write(self, lenient=False):
        strings=[]
        commands=[]
        def inner_recurse(cell):
            if len(cell.text)!=0 and lenient:
                strings.append(cell.text)
                commands.append(f"*{len(strings)-1}*")
            if cell.has_image:
                strings.append(cell.imagepath)
                commands.append(f"#{len(strings)-1}#")
            elif not cell.has_image and not lenient:
                commands.append("%")
            if cell.has_grid:
                if lenient:
                    commands.append(f"<*{cell.gridsize.x},{cell.gridsize.y};")
                else:
                    commands.append(f"<{cell.gridsize.x},{cell.gridsize.y};")
                for i in range(cell.gridsize.y):
                    for j in range(cell.gridsize.x):
                        inner_recurse(cell.grid[i*cell.gridsize.x+j])
                        if j!=cell.gridsize.x-1:
                            commands.append(",")
                    if i!=cell.gridsize.y-1:
                        commands.append(";")
                commands.append(">")
        inner_recurse(self)
        return ("".join(commands),strings)

    def __str__(self):
        return f"Cell({self.text},{self.has_grid},({self.gridsize.x},{self.gridsize.y}))"
    def __getitem__(self, index):
        if not self.has_grid:
            raise Exception("cell has no grid")
        if type(index)==int:
            if index<0 or index>=self.gridsize.x*self.gridsize.y:
                raise Exception("index out of range")
            return self.grid[index]
        if type(index)==slice:
            # if index.start==None:
            #     index=range(0,index.stop,index.step)
            # if index.stop==None:
            #     index=range(index.start,self.gridsize[0]*self.gridsize[1],index.step)
            # if index.step==None:
            #     index=range(index.start,index.stop,1)
            _s=index.start if index.start!=None else 0
            _e=index.stop if index.stop!=None else self.gridsize.x*self.gridsize.y
            _st=index.step if index.step!=None else 1
            return [self.grid[i] for i in range(_s,_e,_st)]
        if type(index)==tuple:
            if len(index)!=2:
                raise Exception("invalid index length")
            if index[0]<0 or index[0]>=self.gridsize.x:
                raise Exception("index out of range")
            if index[1]<0 or index[1]>=self.gridsize.y:
                raise Exception("index out of range")
            return self.grid[index[1]*self.gridsize.x+index[0]]
        raise Exception("invalid index type")

[(i,v) for i,v in enumerate(zip([1,2,3],[4,5,6],[7,8,9]))]
def do_the_parse_and_write(_f,_h,tree):
    out_dict={"categories":[],"icons":[],"items":[],"recipes":[], "limitations":{}, "defaults":{}}
    hash_dict={"items":[], "beacons":[], "belts":[], "fuels":[], "wagons":[], "machines":[], "modules":[], "recipes":[], "technologies":[]}
    
    curr=tree[(0,0)]
    out_dict["categories"]=[{"id":_id.text,"name":_name.text,"icon":_icon.text} for (_id,_name,_icon) in zip(curr.get_column(0)[1:],curr.get_column(1)[1:], curr.get_column(2)[1:])]
    hash_dict["categories"]=[_id.text for _id in curr.get_column(0)[1:]]
    
    out_dict["icons"]=[{"id":_id,"color":_color,"position":_pos} for (_id,_color,_pos) in zip(_f,_h,collect_images(_f))]

    curr=tree[(1,1)]#category,img,id,icon,row,name,data
    first_belt=None
    first_fuel=None
    for i in range(1,curr.gridsize.y):
        try:
            _row=int(curr[(4,i)].text)
            if _row<0:
                raise Exception("row is negative")
        except:
            raise Exception("invalid row: "+curr[(4,i)].text+" at index "+str(i))            
        add={"category":curr[(0,i)].text,"id":curr[(2,i)].text,"name":curr[(5,i)].text, "row":_row, "stack":1}
        if len(curr[(3,i)].text):#~ implicit icon from id is allowed
            add["icon"]=curr[(3,i)].text

        hash_dict["items"].append(add["id"])

        for j in range(len(curr[(6,i)].grid)):
            curr_type=curr[(6,i)][j]
            curr_type_dict={}
            assert curr_type.text in ["machine","fuel","belt"], f"invalid type: {curr_type.text}"

            if curr_type.text=="belt":#~ need to add to defaults
                first_belt=first_belt or add["id"]
                hash_dict["belts"].append(add["id"])
            if curr_type.text=="fuel":
                first_fuel=first_fuel or add["id"]
                hash_dict["fuels"].append(add["id"])
            if curr_type.text=="machine":
                hash_dict["machines"].append(add["id"])

            for k in range(curr_type.gridsize.y):
                _key,_value=curr_type.get_row(k)
                if _key.text=="fuelCategories" and curr_type.text=="machine":
                    assert len(_value.grid)>0, "fuelCategories is empty"
                    curr_type_dict[_key.text]=[v.text for v in _value.grid]
                else:
                    curr_type_dict[_key.text]=_value.text
            add[curr_type.text]=curr_type_dict
        out_dict["items"].append(add)

    curr=tree[(1,2)]#category,img,id,icon,row,name,time,in[],out[],producers[],data
    for i in range(1,curr.gridsize.y):
        try:
            _row=int(curr[(4,i)].text)
            if _row<0:
                raise Exception("row is negative")
        except:
            raise Exception("invalid row: "+curr[(4,i)].text+" at index "+str(i))            
        add={"category":curr[(0,i)].text,"id":curr[(2,i)].text,"name":curr[(5,i)].text, "row":_row, "time":curr[(6,i)].text, "in":{}, "out":{}, "producers":[]}
        if len(curr[(3,i)].text):#~ implicit icon from id is allowed
            add["icon"]=curr[(3,i)].text
        
        hash_dict["recipes"].append(add["id"])

        for j in range(curr[(7,i)].gridsize.y):
            _num,_id=curr[(7,i)].get_row(j)
            add["in"][_id.text]=_num.text
        for j in range(curr[(8,i)].gridsize.y):
            _num,_id=curr[(8,i)].get_row(j)
            add["out"][_id.text]=_num.text
        add["producers"]=[v.text for v in curr[(9,i)].grid]
        for j in range(curr[(10,i)].gridsize.y):
            _key,_value=curr[(10,i)].get_row(j)
            if _key.text=="isMining":
                assert _value.text in ["true","false"], f"invalid isMining: {_value.text}"
                add[_key.text]=_value.text.capitalize()=="true"
            else:
                add[_key.text]=_value.text

        out_dict["recipes"].append(add)

    out_dict["defaults"]={"fuel":first_fuel,"minBelt":first_belt,"excludedRecipes":[],"maxMachineRank":[],"minMachineRank":[],"modIds":[],"moduleRank":[]}
    json.dump(out_dict, open("data.json", "w"))
    json.dump(hash_dict, open("hash.json", "w"))





    # categories=[v for i,v in enumerate(zip(curr.get_column(0),curr.get_column(1),curr.get_column(2))) if i>0]
    # _i_cat=


#opcodes:
#setout,filename
#getdata,depth
#call
#setinfo,i
#goto,goto_str
#write,preset,write_str*
#exit
def add_command(f, command, data):
    if len(data)==0:
        f.write(command+"\n")
    else:
        f.write(command+","+",".join([escape_csv(s) for s in data])+"\n")



#get current path
script_dir = os.path.dirname(os.path.realpath(__file__))
communicate_file = os.path.join(script_dir, "communicate.txt")
out_file = os.path.join(script_dir, "out.txt")
temp_file = os.path.join(script_dir, "temp.txt")
temp2_file = os.path.join(script_dir, "temp2.txt")
curr_dir=os.getcwd()
#! must use newline="\n" to avoid windows newline
#clear communicate_file
with open(communicate_file, "w",newline="\n") as outf:
    
    if args.integer==0:
        add_command(outf, "setout", [out_file])
        add_command(outf, "getdata,-1", [])
        add_command(outf, "setinfo", ["1"])
        add_command(outf, "call", [])
        print(communicate_file,end="")
    if args.integer==1:
        import json
        data=open(out_file, "r").read()
        open(temp2_file, 'w').write(data)

        import numpy as np
        from PIL import Image
        data=json.loads(data)
        tree=Cell()
        tree.recursive_build(data)
        new_dir=tree.text[len("USE_FACTORIO="):]
        assert os.path.exists(new_dir), f"directory {new_dir} does not exist"
        os.chdir(new_dir)
        # [str(i) for i in tree[(1,0)][:]]
        _t=[c.text for c in tree[(1,0)].get_column(2)]
        _f=[c.text for c in tree[(1,0)].get_column(1)]
        _h=[]
        assert len(_t)==len(_f), "lengths of _t and _f are not equal"
        out_tree=Cell()
        out_tree.add_grid(tree.gridsize)
        out_tree[(1,0)].add_grid(tree[(1,0)].gridsize)
        for i in range(1,len(_t)):
            t=0
            try:
                t=int(_t[i])
                if t<1 or t>3:
                    t=0
            except:
                t=0
            do_process=t==0
            t=1 if t==0 else t
            #assert file exists
            # assert os.path.exists(_f[i]+".png"), f"file {_f[i]}.png does not exist"
            if not os.path.exists(_f[i]+".png"):
                continue
            do_process = do_process or not os.path.exists(_f[i]+"_64.png")
            if do_process:
                average_color, A, B = [rgb_to_hex(k) for k in do_the_colors(_f[i])]
                out_cell=out_tree[(1,0)][(2,i)]
                out_cell.text=str(t)
                out_cell.add_grid((1,3))
                out_cell[(0,0)].text=average_color
                out_cell[(0,1)].text=A
                out_cell[(0,2)].text=B
                _h.append([average_color, A, B][t-1])
            else:
                try:
                    _h.append(tree[(1,0)][(2,i)][(0,t-1)].text)
                    if not is_rg_hex_valid(_h[-1]):
                        _h[-1]="#ffffff"
                except:
                    _h.append("#ffffff")
        out_str,strings=out_tree.to_write(lenient=True)
        add_command(outf, "write", [out_str]+strings)
        add_command(outf, "exit", [])
        #filter out non-existing files
        _f=[_f[i] for i in range(1,len(_f)) if os.path.exists(_f[i]+".png")]
        do_the_parse_and_write(_f,_h,tree)

        os.chdir(curr_dir)
        open(temp_file, 'w').write((out_str+"\n"+",".join(strings)))

        print(communicate_file,end="")
#0 -> get_data,-1; set 2 -> (read,save to temp) exit 


# with open(communicate_file, "w",newline="\n") as f:
#     if args.integer==2:
#         # with open(out_file, "r") as f2:
#         lines=open(out_file, "r").readlines()
#         last_i=int(open(temp_file, "r").read())
#         data=json.loads(lines[0])
#         # open(temp_file, 'w').write(str(data))
#         tree=Cell()
#         tree.recursive_build(data)
#         to_write=tree.to_write()
#         open(temp_file, 'w').write(str(to_write))
#         f.write("exit")
#         print(communicate_file,end="")
#     else:
#         f.write(f"setout,{escape_csv(out_file)}\nsetinfo,2\n")
#         if args.integer==1:
#             f.write("getdata,0\n")
#         elif args.integer==0:
#             f.write("getdata,-1\n")
#         f.write("call")
#         open(temp_file, 'w').write(str(args.integer))
#         print(communicate_file,end="")
