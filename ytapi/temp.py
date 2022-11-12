from collections import deque
#find differences in terms of ranges of offsets
#([(start,length,newstart)...], [(delstart,dellength)...], [(unknownstart,unknownlength)...])
#[0,1,2]<>[2,0,1] = ([(0,2,1),(2,1,0)],[],[])
#[0,1,2]<>[0,3,4] = ([],[(1,2)],[(1,2)])
#[0,1,2]<>[0,1,2,3] = ([],[],[(3,1)])
#[0,1,2,3]<>[0,1,2] = ([],[(3,1)],[])
#[0,1,2,3,4,5,6]<>[0,3,4,7,2,1,5] = ([(1,1,5),(2,1,4),(3,2,1)],[(6,1)],[(3,1)])
def find_differences(a,b):
    state=None#0=same,1=diff,2=unknown
    range_start=None; last=None
    diff=[]; dell=[]; unknown=[]
    #setup b index dictionary
    b_index={}
    for i in range(len(b)):
        if b[i] not in b_index:
            b_index[b[i]]=deque()
        b_index[b[i]].append(i)
    #find differences
    for i in range(len(a)):
        if a[i] in b_index:
            b_i=b_index[a[i]].popleft()
            if len(b_index[a[i]])==0:
                del b_index[a[i]]
            if state==None: range_start=i
            elif state==0 and b_i!=i: range_start=i
            elif state==1 and b_i!=last+1:
                diff.append((range_start,i-range_start,last-i+range_start+1))
                range_start=i
            elif state==2:
                dell.append((range_start,i-range_start))
                range_start=i
            state=int(b_i!=i)
            last=b_i
        else:
            if state==None: range_start=i
            elif state==0: range_start=i
            elif state==1:
                diff.append((range_start,i-range_start,last-i+range_start+1))
                range_start=i
            state=2
    if state==1:
        diff.append((range_start,len(a)-range_start,last-len(a)+range_start+1))
    elif state==2:
        dell.append((range_start,len(a)-range_start))
    #find unknown
    sorted_b_index=[[[x[0],y] for y in x[1]] for x in b_index.items()]
    sorted_b_index=[x for y in sorted_b_index for x in y]
    sorted_b_index.sort(key=lambda x:x[1])
    start_range=None; last=None; llen=0
    for i in range(len(sorted_b_index)):
        if last==None:
            start_range=sorted_b_index[i][1]
            llen=0
        elif last!=sorted_b_index[i][1]-1:
            unknown.append((start_range,llen))
            start_range=sorted_b_index[i][1]
            llen=0
        last=sorted_b_index[i][1]
        llen+=1
    if start_range!=None:
        unknown.append((start_range,llen))
    return (diff,dell,unknown)

# a=[0,1,2,3,4,5,6]
# b=[0,3,4,7,2,1,5]
a=[0,0,2,3,1,1]
b=[5,0,1,4,0,2,4,4]
# print(find_differences(a,b))
# print(find_differences([1,2,3],[]))
print(find_differences([1,2,3,4,5],[1,2,6,3,4,5]))

