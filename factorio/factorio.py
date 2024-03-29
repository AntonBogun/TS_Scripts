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
    def is_empty(self):
        return len(self.text)==0 and not self.has_grid and not self.has_image
    def to_write(self, lenient=True):
        strings={}
        str_count=0
        commands=[]
        def inner_recurse(cell):
            nonlocal str_count
            if len(cell.text)!=0 or not lenient:
                if cell.text not in strings:
                    strings[cell.text]=str_count
                    str_count+=1
                commands.append(f"*{strings[cell.text]}*")
            if cell.has_image:
                if cell.imagepath not in strings:
                    strings[cell.imagepath]=str_count
                    str_count+=1
                commands.append(f"#{strings[cell.imagepath]}#")
            elif not cell.has_image and not lenient:
                commands.append("%")
            if cell.has_grid:
                if lenient:
                    commands.append(f"<*{cell.gridsize.x},{cell.gridsize.y};")
                else:
                    commands.append(f"<{cell.gridsize.x},{cell.gridsize.y};")
                last_pos=(0,0)
                for i in range(cell.gridsize.y):
                    for j in range(cell.gridsize.x):
                        if lenient and cell.grid[i*cell.gridsize.x+j].is_empty():
                            continue
                        off=j+i*cell.gridsize.x-(last_pos[0]+last_pos[1]*cell.gridsize.x)
                        off_x=j-last_pos[0]
                        off_y=i-last_pos[1]
                        if off<=6:#!not the most efficient but reasonable
                            if off_y==0:
                                if off_x!=0:
                                    commands.append(","*off_x)
                            else:
                                commands.append(";"*off_y)
                                if j!=0:
                                    commands.append(","*j)
                        else:
                            if off_x!=0:
                                commands.append(f"({j})")
                            if off_y!=0:
                                commands.append(f"[{i}]")
                        last_pos=(j,i)#!forgor to update
                        inner_recurse(cell.grid[i*cell.gridsize.x+j])
                commands.append(">")
        inner_recurse(self)
        # return ("".join(commands),strings)
        str_sorted_list=[None]*str_count
        for k,v in strings.items():
            str_sorted_list[v]=k
        return ("".join(commands),str_sorted_list)

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
    def __setitem__(self, index, value):
        if not self.has_grid:
            raise Exception("cell has no grid")
        if type(index)==int:
            if index<0 or index>=self.gridsize.x*self.gridsize.y:
                raise Exception("index out of range")
            self.grid[index]=value
            return
        if type(index)==slice:
            raise Exception("slice not supported")
        if type(index)==tuple:
            if len(index)!=2:
                raise Exception("invalid index length")
            if index[0]<0 or index[0]>=self.gridsize.x:
                raise Exception("index out of range")
            if index[1]<0 or index[1]>=self.gridsize.y:
                raise Exception("index out of range")
            self.grid[index[1]*self.gridsize.x+index[0]]=value
            return
        raise Exception("invalid index type")

# [(i,v) for i,v in enumerate(zip([1,2,3],[4,5,6],[7,8,9]))]



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






#===========
def do_images_cell(cell, pos, is_mix):
    _nums=[c.text for c in cell.get_column(0)]
    _t=[c.text for c in cell.get_column(3)]
    _f=[c.text for c in cell.get_column(1 if is_mix else 2)]
    if is_mix:
        _icon=[c.text for c in cell.get_column(2)]
    _h=[]
    assert len(_t)==len(_f)==len(_nums), f"lengths of _t and _f and _nums are not equal at images {pos}"
    out_tree=Cell()
    out_tree.add_grid(cell.gridsize)
    for i in range(1,len(_t)):
        if _nums[i]=="#":
            continue
        if is_mix and _icon[i]=="#":
            continue
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
        #assert no space in file name
        # assert " " not in _f[i], f"file {repr(_f[i])} contains space at row {_nums[i]} at images {pos}"
        assert len(_f[i])>0, f"file is empty at row {_nums[i]} at images {pos}"#!also covers for checking in items and in recipes
        assert os.path.exists(_f[i]+".png"), f"file {_f[i]}.png does not exist at row {_nums[i]} at images {pos}"
        do_process = do_process or not os.path.exists(_f[i]+"_64.png")
        if do_process:
            average_color, A, B = [rgb_to_hex(k) for k in do_the_colors(_f[i])]
            out_cell=out_tree[(3,i)]
            out_cell.text=str(t)
            out_cell.add_grid((1,3))
            out_cell[(0,0)].text=average_color
            out_cell[(0,1)].text=A
            out_cell[(0,2)].text=B
            _h.append([average_color, A, B][t-1])
        else:
            try:
                _h.append(cell[(3,i)][(0,t-1)].text)
                if not is_rg_hex_valid(_h[-1]):
                    _h[-1]="#ffffff"
            except:
                _h.append("#ffffff")
    if is_mix:
        _f=[_f[i] for i in range(1,len(_f)) if _nums[i]!="#" and _icon[i]!="#"]
    else:
        _f=[_f[i] for i in range(1,len(_f)) if _nums[i]!="#"]

    return (out_tree,_f,_h)

def do_items_cell(cell, hash_dict, pos, out, is_mix):
    # out=[]
    first_belt=None
    first_fuel=None
    x_arr=[i for i in range(8)] if not is_mix else [0,4,2,1,5,6,7,8]
    for i in range(1,cell.gridsize.y):
        num=cell[(x_arr[0],i)].text
        if num=="#":
            continue
        if is_mix and len(cell[(x_arr[1],i)].text)==0:#category
            continue
        try:
            _row=int(cell[(x_arr[5],i)].text)
            if _row<0:
                raise Exception(f"row is negative at row {num} at items {pos}")
        except:
            raise Exception("invalid row value: "+cell[(x_arr[5],i)].text+f" at row {num} at items {pos}")
        
        add={"category":cell[(x_arr[1],i)].text,"id":cell[(x_arr[3],i)].text,"name":cell[(x_arr[6],i)].text, "row":_row, "stack":1}


        if len(add["name"])==0:
            add["name"]=add["id"]
        
        if len(cell[(x_arr[4],i)].text):#~ implicit icon from id is allowed
            add["icon"]=cell[(x_arr[4],i)].text
        
        if add["id"] in hash_dict["items"]:
            raise Exception(f"item {add['id']} already exists at row {num} at items {pos}")
        hash_dict["items"].add(add["id"])

        for j in range(len(cell[(x_arr[7],i)].grid)):
            curr_type=cell[(x_arr[7],i)][j]
            curr_type_dict={}
            assert curr_type.text in ["machine","fuel","belt"], f"invalid type: {curr_type.text} at row {num} at items {pos}"
            if curr_type.text=="belt":#~ need to add to defaults
                first_belt=first_belt or add["id"]
                if add["id"] in hash_dict["belts"]:
                    raise Exception(f"belt {add['id']} already exists at row {num} at items {pos}")
                hash_dict["belts"].add(add["id"])
            if curr_type.text=="fuel":
                first_fuel=first_fuel or add["id"]
                if add["id"] in hash_dict["fuels"]:
                    raise Exception(f"fuel {add['id']} already exists at row {num} at items {pos}")
                hash_dict["fuels"].add(add["id"])
            if curr_type.text=="machine":
                if add["id"] in hash_dict["machines"]:
                    raise Exception(f"machine {add['id']} already exists at row {num} at items {pos}")
                hash_dict["machines"].add(add["id"])

            for k in range(curr_type.gridsize.y):
                _key,_value=curr_type.get_row(k)
                if _key.text=="fuelCategories" and curr_type.text=="machine":
                    assert len(_value.grid)>0, f"fuelCategories is empty at row {num} at items {pos}"
                    curr_type_dict[_key.text]=[v.text for v in _value.grid]
                else:
                    curr_type_dict[_key.text]=_value.text
                    if _key.text=="category" and curr_type.text=="fuel":
                        hash_dict["fuelCategories"].add(_value.text)
            add[curr_type.text]=curr_type_dict
        out.append(add)
    # return out
    return first_belt, first_fuel

def do_recipes_cell(cell, hash_dict, pos, out, is_mix):
    x_arr=[i for i in range(12)] if not is_mix else [0,9,2,1,10,11,12,13,14,15,16,17]
    for i in range(1,cell.gridsize.y):
        num=cell[(x_arr[0],i)].text
        if num=="#":
            continue
        if is_mix and len(cell[(x_arr[1],i)].text)==0:
            continue
        try:
            _row=int(cell[(x_arr[5],i)].text)
            if _row<0:
                raise Exception("row is negative")
        except:
            # raise Exception("invalid row: "+cell[(4,i)].text+" at index "+str(i))    
            raise Exception(f"invalid row value: {cell[(x_arr[5],i)].text} at row {i} at recipes {pos}")        
        add={"category":cell[(x_arr[1],i)].text,"id":cell[(x_arr[3],i)].text,"name":cell[(x_arr[6],i)].text, "row":_row, "time":cell[(x_arr[7],i)].text, "in":{}, "out":{}, "producers":[]}

        if len(add["name"])==0:
            add["name"]=add["id"]

        if len(cell[(x_arr[4],i)].text):#~ implicit icon from id is allowed
            add["icon"]=cell[(x_arr[4],i)].text
            assert " " not in add["icon"], f"icon {repr(add['icon'])} contains space at row {num} at recipes {pos}"
        if add["id"] in hash_dict["recipes"]:
            raise Exception(f"recipe {add['id']} already exists at row {num} at recipes {pos}")        
        hash_dict["recipes"].add(add["id"])

        for j in range(cell[(x_arr[8],i)].gridsize.y):
            _num,_id=cell[(x_arr[8],i)].get_row(j)
            add["in"][_id.text]=_num.text
            # assert " " not in _id.text, f"item {repr(_id.text)} contains space at row {num} at recipes {pos}"
        for j in range(cell[(x_arr[9],i)].gridsize.y):
            _num,_id=cell[(x_arr[9],i)].get_row(j)
            add["out"][_id.text]=_num.text
            # assert " " not in _id.text, f"item {repr(_id.text)} contains space at row {num} at recipes {pos}"

        add["producers"]=[v.text for v in cell[(x_arr[10],i)].grid]
        # for j in range(len(add["producers"])):
        #     assert " " not in add["producers"][j], f"producer {repr(add['producers'][j])} contains space at row {num} at recipes {pos}"
        for j in range(cell[(x_arr[11],i)].gridsize.y):
            _key,_value=cell[(x_arr[11],i)].get_row(j)
            if _key.text=="isMining":
                assert _value.text.lower() in ["true","false"], f"invalid isMining: {_value.text} at row {num} at recipes {pos}"
                add[_key.text]=_value.text.lower()=="true"
            else:
                add[_key.text]=_value.text

        out.append(add)

def do_the_parse_and_write(tree):
    out_dict={"categories":[],"icons":[],"items":[],"recipes":[], "limitations":{}, "defaults":{}}
    # hash_dict={"items":[], "beacons":[], "belts":[], "fuels":[], "wagons":[], "machines":[], "modules":[], "recipes":[], "technologies":[]}
    hash_dict={"items":set(), "beacons":[], "belts":set(), "fuels":set(), "wagons":[], "machines":set(), "modules":[], "recipes":set(), "technologies":[], "categories":set(), "icons":set(), "fuelCategories":set()}
    out_tree=Cell()
    out_tree.add_grid(tree.gridsize)
    _f=[]
    _h=[]
    first_belt=None
    first_fuel=None

    for pos in range(len(tree.grid)):


        curr=tree[pos]
        if curr.text=="categories":
            proof_set=hash_dict["categories"]
            for i,d in ((_num.text,{"id":_id.text,"name":_name.text if len(_name.text) else _id.text ,"icon":_icon.text
                                  }) for (_num,_id,_name,_icon) in zip(curr.get_column(0)[1:],curr.get_column(1)[1:],curr.get_column(2)[1:], curr.get_column(3)[1:]) if _num.text!="#"):
                if d["id"] in proof_set:
                    raise Exception(f"category {d['id']} already exists at row {i} at categories {pos}")
                if len(d["id"])==0:
                    raise Exception(f"category id is empty at row {i} at categories {pos}")
                # proof_dict_cat[d["id"]]=i+proof_l
                proof_set.add(d["id"])
                out_dict["categories"].append(d)
        elif curr.text=="images":
            proof_set=hash_dict["icons"]
            out_tree[pos],_f0,_h0=do_images_cell(curr, pos, is_mix=False)
            for i in range(len(_f0)):
                if _f0[i] in proof_set:
                    raise Exception(f"icon {_f0[i]} already exists at row {curr[(0,i+1)].text} at images {pos}")
                # proof_dict_icon[_f0[i]]=i+proof_l
                proof_set.add(_f0[i])
            _f.extend(_f0)
            _h.extend(_h0)
        elif curr.text=="items":
            _first_belt, _first_fuel=do_items_cell(curr, hash_dict, pos, out_dict["items"], is_mix=False)
            first_belt=first_belt or _first_belt
            first_fuel=first_fuel or _first_fuel
        elif curr.text=="recipes":
            do_recipes_cell(curr, hash_dict, pos, out_dict["recipes"], is_mix=False)
        elif curr.text=="mix":
            proof_set=hash_dict["icons"]
            out_tree[pos],_f0,_h0=do_images_cell(curr, pos, is_mix=True)
            for i in range(len(_f0)):
                if _f0[i] in proof_set:
                    raise Exception(f"icon {_f0[i]} already exists at row {curr[(0,i+1)].text} at mix {pos}")
                # proof_dict_icon[_f0[i]]=i+proof_l
                proof_set.add(_f0[i])
            _f.extend(_f0)
            _h.extend(_h0)
            _first_belt, _first_fuel=do_items_cell(curr, hash_dict, pos, out_dict["items"], is_mix=True)
            first_belt=first_belt or _first_belt
            first_fuel=first_fuel or _first_fuel
            do_recipes_cell(curr, hash_dict, pos, out_dict["recipes"], is_mix=True)




    # hash_dict["categories"]=[_id.text for _id in curr.get_column(0)[1:]]

    out_dict["icons"]=[{"id":_id,"color":_color,"position":_pos} for (_id,_color,_pos) in zip(_f,_h,collect_images(_f))]
    
    #perform checks
    #category icons -> check exist
    #item icons, item categories -> check exist; check fuelCategories
    #recipe icons, recipe categories, recipe items, recipe producers -> check exist
    for i in out_dict["categories"]:
        assert i["icon"] in hash_dict["icons"], f"category icon {repr(i['icon'])} at category {repr(i['id'])} does not exist in icons"
    for i in out_dict["items"]:
        if "icon" in i:
            assert i["icon"] in hash_dict["icons"], f"item icon {repr(i['icon'])} at item {repr(i['id'])} does not exist in icons"
        else:
            assert i["id"] in hash_dict["icons"], f"implicit item icon {repr(i['id'])} does not exist in icons"
        assert i["category"] in hash_dict["categories"], f"item category {repr(i['category'])} at item {repr(i['id'])} does not exist in categories"
        if "machine" in i and "fuelCategories" in i["machine"]:
            for j in i["machine"]["fuelCategories"]:
                assert j in hash_dict["fuelCategories"], f"fuelCategory {repr(j)} at item {repr(i['id'])} does not exist in fuelCategories"
    for i in out_dict["recipes"]:
        if "icon" in i:
            assert i["icon"] in hash_dict["icons"], f"recipe icon {repr(i['icon'])} at recipe {repr(i['id'])} does not exist in icons"
        else:
            assert (i["id"] in hash_dict["icons"] or i["id"] in hash_dict["items"]), f"implicit recipe icon {repr(i['id'])} does not exist in icons or items"
        assert i["category"] in hash_dict["categories"], f"recipe category {repr(i['category'])} at recipe {repr(i['id'])} does not exist in categories"
        for j in i["in"]:
            assert j in hash_dict["items"], f"recipe item {repr(j)} at recipe {repr(i['id'])} does not exist in items"
        for j in i["out"]:
            assert j in hash_dict["items"], f"recipe item {repr(j)} at recipe {repr(i['id'])} does not exist in items"
        for j in i["producers"]:
            assert j in hash_dict["machines"], f"recipe producer {repr(j)} at recipe {repr(i['id'])} does not exist in machines"

    out_dict["defaults"]={"fuel":first_fuel,"minBelt":first_belt,"excludedRecipes":[],"maxMachineRank":[],"minMachineRank":[],"modIds":[],"moduleRank":[]}
    #fix hash_dict
    # hash_dict={"items":[], "beacons":[], "belts":[], "fuels":[], "wagons":[], "machines":[], "modules":[], "recipes":[], "technologies":[]}
    # hash_dict={"items":set(), "beacons":[], "belts":set(), "fuels":set(), "wagons":[], "machines":set(), "modules":[], "recipes":set(), "technologies":[], "categories":set(), "icons":set(), "fuelCategories":set()}

    for i in ["items","belts","fuels","machines","recipes"]:
        hash_dict[i]=list(hash_dict[i])
    for i in ["categories","icons","fuelCategories"]:
        del hash_dict[i]


    json.dump(out_dict, open("data.json", "w"))
    json.dump(hash_dict, open("hash.json", "w"))

    return out_tree



    # categories=[v for i,v in enumerate(zip(curr.get_column(0),curr.get_column(1),curr.get_column(2))) if i>0]
    # _i_cat=


#============
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

        # out_tree=Cell()
        # out_tree.add_grid(tree.gridsize)
        # out_tree[(1,0)].add_grid(tree[(1,0)].gridsize)

        # [str(i) for i in tree[(1,0)][:]]

        #filter out non-existing files
        out_tree=do_the_parse_and_write(tree)
        
        out_str,strings=out_tree.to_write(lenient=True)
        add_command(outf, "write", [out_str]+strings)
        add_command(outf, "exit", [])
        
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
