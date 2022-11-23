from ytmusicapi import YTMusic
import json,csv,argparse,os
from collections import deque

current_file_path=__file__
last_slash=current_file_path.rfind("\\")
directory=current_file_path[:last_slash+1]
#make sure headers_auth.json exists
if not os.path.exists(directory+"headers_auth.json"):
    raise Exception("headers_auth.json not found")
ytmusic=YTMusic(directory+'headers_auth.json')



def actually_in(a,b):
    return a in b and b[a] is not None



#requests the full playlist. does not handle errors.
def request_playlist(playlistID):
    #!never use trackCount when full track list is known
    size=ytmusic.get_playlist(playlistID)["trackCount"]
    out=ytmusic.get_playlist(playlistID,limit=size)
    config={
        "videoId":1,
        "title":1,
        "artists":1,
        "album":1,
        "likeStatus":0,
        "thumbnails":0,
        "isAvailable":0,
        "isExplicit":0,
        "duration":1,
        "duration_seconds":0,
        "setVideoId":1,#id of the track in the playlist, needed as there can be duplicates
        "feedbackTokens":0
    }
    out["tracks"]=filter_config(out["tracks"],config)
    return out

#loads playlist from a file
def load_playlist(filename):
    with open(filename,"r", encoding="utf-8") as f:
        return json.load(f)

#saves playlist to a file
def save_playlist(playlist,filename):
    with open(filename,"w", encoding="utf-8") as f:
        json.dump(playlist,f,indent=2,ensure_ascii=False)

#requests the playlist and checks if it matches to stored playlist according to config.
#raises an exception if not. newplaylist is a global that can be accessed from outside to get the new playlist.
def assert_up_to_date(playlistID,playlist,config:dict):
    global newplaylist
    if playlistID!=playlist["id"]:
        raise Exception("playlistID does not match: {} != {}".format(playlistID,playlist["id"]))
    newplaylist=request_playlist(playlistID)
    if len(newplaylist["tracks"])!=len(playlist["tracks"]):
        raise Exception("playlist length changed: {} != {}"
        .format(len(newplaylist["tracks"]),len(playlist["tracks"])))
    values_to_check=[k for k in config if config[k]]
    for i in range(len(playlist["tracks"])):
        for k in values_to_check:
            if newplaylist["tracks"][i][k]!=playlist["tracks"][i][k]:
                raise Exception("playlist value {} changed: {}!={} in index {}"
                .format(k,newplaylist["tracks"][i][k],playlist["tracks"][i][k],i))

#moves an item in index old to in front of index new
def move_to_infront(l:list,old:int,new:int):
    if old==new or new==old+1:#old==new cannot be done, new==old+1 does not change anything
        return
    if old not in range(len(l)) or new not in range(len(l)):
        return
    if new<old:
        l.insert(new,l.pop(old))
    else:
        l.insert(new-1,l.pop(old))#edge cases


#moves a track to the front of another track
def move_in_playlist(playlist,origin:int,target:int):
    tracks=playlist["tracks"]
    # if origin==target or target==origin+1:
    #origin==target cannot be done, target==origin+1 does not change anything
    if target in range(origin,origin+2):
        return
    if origin not in range(len(tracks)) or target not in range(len(tracks)):
        ltracks=len(tracks)
        raise Exception("can not move from {} to {} in playlist length {}".format(origin,target,ltracks))
    # print((tracks[origin]["setVideoId"],tracks[target]["setVideoId"]))
    status=ytmusic.edit_playlist(playlist["id"],
    moveItem=(tracks[origin]["setVideoId"],tracks[target]["setVideoId"]))
    if status!="STATUS_SUCCEEDED":
        raise Exception("move_in_playlist failed: {}".format(status))
    # print(status)#!remove later
    move_to_infront(tracks,origin,target)

#move "length" tracks from origin to target
def move_in_playlist_range(playlist,origin:int,target:int,length:int):
    tracks=playlist["tracks"]
    if target in range(origin,origin+length+1):#movement impossible/does not change anything
        return
    if origin not in range(len(tracks)) or target not in range(len(tracks)):
        ltracks=len(tracks)
        raise Exception("can not move from {} to {} in playlist length {}".format(origin,target,ltracks))
    if origin+length>len(tracks):
        raise Exception("can not move {} items from {} to {} in playlist length {}".format(length,origin,target,ltracks))
    for i in range(length):
        #tracks are consumed top to bottom, and always inserted before the same target
        #however when target is above origin, the origin shifts along with
        try:
            move_in_playlist(playlist,origin+i*(target<origin),target+i*(target<origin))
        except Exception as e:
            raise Exception("{} move_range failed: {}".format(i,e))


def delete_unavailable(playlist):
    toremove=[]
    for i in range(len(playlist["tracks"])):
        if not playlist["tracks"][i]["isAvailable"]:
            toremove.append(i)
    if len(toremove)==0:
        return
    # print("{}".format(",".join([playlist["tracks"][v]["setVideoId"] for v in toremove])))
    status=ytmusic.remove_playlist_items(playlist["id"],[playlist["tracks"][v] for v in toremove])
    if status!="STATUS_SUCCEEDED":
        raise Exception("delete_unavailable failed: {}".format(status))
    # print(status)#!remove later
    for v in toremove[::-1]:
        del playlist["tracks"][v]





IGNORE_UNAVAILABLE=0 #1 to enable
# unavailable not allowed by default as they can mess up the movement
def filter_config(tracks,config):
    newtracks=[]
    for t in tracks:
        if not t["isAvailable"]:
            if IGNORE_UNAVAILABLE:
                continue
            raise Exception("Track {} is not available, please delete all unavailable tracks before using the script or enable IGNORE_UNAVAILABLE mode (on your own risk)".format(t["setVideoId"]))
        newtracks.append({k:t[k] for k in config if config[k] and actually_in(k,t)})
    return newtracks

def dump_playlist_treesheets(playlist,filename):
    with open(filename,"w",encoding="utf-8") as f:
        # json.dump(playlist,f,indent=2,ensure_ascii=False)
        # return
        writer=csv.writer(f, quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n",quotechar=None)
        writer.writerow([playlist["title"]])
        for t in playlist["tracks"]:
            writer.writerow([
                t["title"],
                ",".join([a["name"] for a in t["artists"]]),
                # t["artists"][0]["name"]+(" and others" if len(t["artists"])>1 else ""),
                # "-" if not t["album"] else t["album"]["name"],
                "-" if not actually_in("album",t) else t["album"]["name"],
                t["duration"],
                t["videoId"],
                t["setVideoId"]
                ])

def load_playlist_treesheets(filename):
    with open(filename,"r",encoding="utf-8") as f:
        reader=csv.reader(f, quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n",quotechar=None)
        title=next(reader)[0]
        tracks=[]
        for row in reader:
            tracks.append({
                "title":row[0],
                "artists":[{"name":row[1]}],
                "album":{"name":row[2]},
                "duration":row[3],
                "videoId":row[4],
                "setVideoId":row[5]
                })
        return {"title":title,"tracks":tracks}

# {
#     "title": "...",//relevant
#     "playlistId": "...",//relevant
#     "thumbnails": [...],//irrelevant until lobster can import pictures into treesheets
#     "description": "...",//relevant
#     "count": "...",//relevant,can be null
#     "author": [//can be null
#       {
#         "name": "...",//relevant
#         "id": "..."
#       }
#     ]
#   },

def dump_playlist_list_treesheets(playlists,filename):
    with open(filename,"w",encoding="utf-8") as f:
        # json.dump(playlist,f,indent=2,ensure_ascii=False)
        # return
        writer=csv.writer(f, quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n",quotechar=None)
        for p in playlists:
            writer.writerow([
                p["title"],
                p["description"],
                "-" if not actually_in("count", p) else p["count"],
                "-" if not actually_in("author", p) else ",".join([a["name"] for a in p["author"]]),
                p["playlistId"]
                # "-" if not p["count"] else p["count"],
                # "-" if not p["author"] else ",".join([a["name"] for a in p["author"]])
                ])


#find differences in terms of ranges of offsets
#([(start,length,newstart)...], [(delstart,dellength)...], [(unknownstart,unknownlength)...])
#[0,1,2]<>[2,0,1] = ([(0,2,1),(2,1,0)],[],[])
#[0,1,2]<>[0,3,4] = ([],[(1,2)],[(1,2)])
#[0,1,2]<>[0,1,2,3] = ([],[],[(3,1)])
#[0,1,2,3]<>[0,1,2] = ([],[(3,1)],[])
#[0,1,2,3,4,5,6]<>[0,3,4,7,2,1,5] = ([(1,1,5),(2,1,4),(3,2,1), (5, 1, 6)],[(6,1)],[(3,1)])
def find_differences(a,b):
    state=None#previous comparison state, 0=same,1=diff,2=deleted
    rs=None; last=None #range start index, last range index
    diff=[]; dell=[]; unknown=[]
    #setup b index dictionary (value:indexes)
    b_index={}
    for i in range(len(b)):
        if b[i] not in b_index:
            b_index[b[i]]=deque()
        b_index[b[i]].append(i)
    #find differences
    for i in range(len(a)):
        if a[i] in b_index:
            b_i=b_index[a[i]].popleft()
            if len(b_index[a[i]])==0: del b_index[a[i]]#not sure does anything
            if state==None: rs=i #init rs=0
            elif state==0 and b_i!=i: rs=i#beginning of diff
            elif state==1 and b_i!=last+1:#end of diff?
                diff.append((rs,i-rs,last-i+rs+1))#append diff
                rs=i
            elif state==2:
                dell.append((rs,i-rs))#end of deleted, append
                rs=i
            state=int(b_i!=i)#fancy way of saying if b_i!=i: state=1 else: state=0
            last=b_i
        else:
            if state==None: rs=i #rs=0
            elif state==0: rs=i #beginning of deleted
            elif state==1: #end of diff
                diff.append((rs,i-rs,last-i+rs+1))#append diff
                rs=i
            state=2
    #end cleanup
    if state==1:
        diff.append((rs,len(a)-rs,last-len(a)+rs+1))
    elif state==2:
        dell.append((rs,len(a)-rs))
    #find unknown (note that items in b that were in a are no longer in b_index)
    sorted_b_index=[[[x[0],y] for y in x[1]] for x in b_index.items()]#unpack b_index
    sorted_b_index=[x for y in sorted_b_index for x in y]#flatten
    sorted_b_index.sort(key=lambda x:x[1])#sort by index
    rs=None; last=None; llen=0#range start index, last range index, length of range (didn't feel like last-i+rs+1?)
    for i in range(len(sorted_b_index)):
        if last==None:
            rs=sorted_b_index[i][1]
            llen=0
        elif last!=sorted_b_index[i][1]-1:#end of range?
            unknown.append((rs,llen))
            rs=sorted_b_index[i][1]
            llen=0
        last=sorted_b_index[i][1]
        llen+=1
    #end cleanup
    if rs!=None:
        unknown.append((rs,llen))
    return (diff,dell,unknown)



if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Loads playlists from library, tracks from a playlist and moves tracks in a playlist")
    #toggle playlists, set "playlists" to True and ignore playlist_id, load all playlists from library
    parser.add_argument("-p","--playlists",action="store_true",help="load all playlists from library")
    #not needed as the script will not be called if playlist is just being loaded
    # #toggle load from file, set "load" to True and load from *PLAYLIST_ID*.txt
    # parser.add_argument("-l","--load",action="store_true",help="load playlist from file")
    #input playlist id
    parser.add_argument("-i","--playlist_id",help="playlist id")
    parser.add_argument("-m","--move",action="store_true",help="move tracks in playlist")
    parser.add_argument("-r","--range",help="movement range, format: origin,target,length")
    parser.add_argument("-d","--diff",action="store_true",help="find differences between two playlists")
    args=parser.parse_args()
    # args=parser.parse_args(["-i","...","-d"])#!DEBUG

    if args.playlists:
        file=directory+"playlists.csv"
        dump_playlist_list_treesheets(ytmusic.get_library_playlists(100),file)
        print(file,end="")
        exit(0)
    #!not actually sure if playlist_id will always be a valid file name
    file=directory+args.playlist_id+".csv"
    if args.move:
        playlist=load_playlist_treesheets(file)
        playlist["id"]=args.playlist_id
        move_range=[int(v) for v in args.range.split(",")]
        move_in_playlist_range(playlist,move_range[0],move_range[1],move_range[2])
        dump_playlist_treesheets(playlist,file)
        print(file,end="")
    else:
        if args.diff:#rename old playlist to temp.txt if exists (using os.rename), store new playlist in place, store differences of lines in diff.txt
            if not os.path.exists(file):
                print("File does not exist {}".format(file))
                exit(1)
            request=request_playlist(args.playlist_id)
            #=can be commented starting here
            if os.path.exists(directory+"temp.txt"):
                os.remove(directory+"temp.txt")
            os.rename(file,directory+"temp.txt")
            dump_playlist_treesheets(request,file)
            #=ending here to debug without requesting playlist
            with open(file,"r",encoding="utf-8") as new_f:
                with open(directory+"temp.txt","r",encoding="utf-8") as old_f:
                    new_lines=new_f.readlines()
                    old_lines=old_f.readlines()
                    diffs=find_differences(old_lines,new_lines)
                    with open(directory+"diff.txt","w",encoding="utf-8") as f:
                        diff=diffs[0]; dell=diffs[1]; unknown=diffs[2]
                        f.write("(1 based indexes, add 1 to convert to lines in the file)\ndifferences:\n(old start-old inclusive end) -> (new start-new inclusive end)\n")
                        for i in range(len(diff)):
                            f.write("({}-{}) -> ({}-{})\n".format(diff[i][0],diff[i][0]+diff[i][1]-1,diff[i][2],diff[i][2]+diff[i][1]-1))
                        f.write("\ndeleted (old):\nstart-inclusive end\nNOTE: if youtube changes track length by a second the video would be marked as deleted and unknown at the same time\n")
                        for i in range(len(dell)):
                            f.write("{}-{}\n".format(dell[i][0],dell[i][0]+dell[i][1]-1))
                        f.write("\nunknown (new):\nstart-inclusive end\n")
                        for i in range(len(unknown)):
                            f.write("{}-{}\n".format(unknown[i][0],unknown[i][0]+unknown[i][1]-1))
                    print("Stored differences in {}, old playlist in {}".format(directory+"diff.txt",directory+"temp.txt"),end="")
        else:
            dump_playlist_treesheets(request_playlist(args.playlist_id),file)
            print(file,end="")
    # print(file,end="")#~no point because in case of load, the file should already be known
    # # print(ytmusic.get_library_playlists())
    # playlist="..."
    
    # # playlist="..."
    # RESET=1
    # LOADFILE=directory+"saved.txt"
    # TEMPFILE=directory+"temp.txt"
    # out={}
    # if RESET:
    #     out=request_playlist(playlist)
    # else:
    #     out=load_playlist(LOADFILE)
    # dump_playlist_treesheets(out,TEMPFILE)
    # print(TEMPFILE,end="")
    
    # with open(TEMPFILE,"w",encoding="utf-8") as f:
    #     json.dump(ytmusic.get_library_playlists(),f,indent=2,ensure_ascii=False)
    # dump_playlist_list_treesheets(ytmusic.get_library_playlists(100),TEMPFILE)

    # delete_unavailable(out)
    # move_in_playlist_range(out,1,7,3)
    # try:
    #     assert_up_to_date(playlist,out,{"videoId":1,"setVideoId":1,"isAvailable":1})
    #     # save_playlist(newplaylist,TEMPFILE)
    # except Exception as e:
    #     print(e)
    #     save_playlist(newplaylist,LOADFILE)
    # else:
    #     save_playlist(out,LOADFILE)


# totallen=0
# for t in out["tracks"]:
#     if "duration_seconds" not in t:
#         # print(t)
#         continue
#     totallen+=t["duration_seconds"]
# print(totallen)










