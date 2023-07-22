import std
import print_f
import selection
import navigation
import modify
import bugs
import file
import parse


def main():
    let l_l=[]
    ytapi(l_l)
    write_log(l_l)
main()



def ytapi(l_l):
    let api_location="scripts\\ytapi"//doesn't need the entire path, as subprocess starts from the TS directory
    if_err_print_return(l_l,is_grid_at_least_nxm(4,4))
    ts_goto_selection()//top left corner
    if ts_num_children()>0://moving
        let origin,no_err=string_to_int(ts_get_text())
        if_err_print_return(l_l,not no_err,"origin is not an integer: {ts_get_text()}")

        ts_goto_child(0)
        let target,no_err2=string_to_int(ts_get_text())
        if_err_print_return(l_l,not no_err2,"target is not an integer: {ts_get_text()}")
        
        ts_goto_parent()
        let len_y=selection_size_y()
        
        ts_goto_parent()
        move_in_grid(0,0)//status
        if_err_print_return(l_l,ts_get_text()!="LOADED","Playlist not set up correctly (ensure LOADED)")
        
        move_in_grid(1,0)
        let playlist_id=ts_get_text()

        move_in_grid(3,3)
        let size=ts_num_columns_rows()
        
        ts_goto_child(0)
        if_err_print_return(l_l,target<0 or target>=size.y-1,"Invalid target: {target}")
        if_err_print_return(l_l,target>=origin and target<origin+len_y,"Target within range of origin and length.")
        if_err_print_return(l_l,target==origin+len_y,"Moving does not change anything.")
        
        let end_pos=target-if target>origin: len_y else: 0
        print_f(l_l,"Moving tracks {origin}-{origin+len_y-1} in front of {target}, will become {end_pos}-{end_pos+len_y-1}")
        
        let status, buffer = launch_subprocess(["python", "{api_location}\\ytapitest.py","-i",playlist_id,"-m","-r","{origin},{target},{len_y}"])
        if_err_print_return(l_l,status,"Python script returned error status {status}: {buffer}")

        let result=read_lines("{buffer}")//returns file name in buffer
        if result:
            let len=length(result)
            for(len-1) i:
                let row,err=decode_csv(result[i+1])
                if_err_print_return(l_l,err,"Error decoding track {i}: {err}")
                if_err_print_return(l_l,length(row)!=6,"Invalid row length at track {i}: {row}")
                let preset5="[{i+1}]*0*,*1*,*2*,*3*,*4*,*5*"
                let strlist5=["{i}",row[0],row[1],row[2],row[3],"https://music.youtube.com/watch?v={row[4]}&list={playlist_id}"]
                if_err_print_return(l_l,ts_write_from_preset(preset5,strlist5,6,len+1))
            print_f(l_l,"Tracks moved successfully.")
            ts_goto_selection()
            delete_subgrid()
            return
    move_in_grid(0,0)//status
    if ts_get_text()!="LOADED":
        if_err_print_return(l_l,ts_get_text()!="","For reloading the preset, ensure top left cell is empty.")
        var preset="*0* preset done, *1* id, *2* action;"
        preset+="*3* preset info, *4* id describe,*5* action describe"
        preset+="<2,4;*6* load playlists,*7*;"
        preset+="*8* load playlist from id,*9*;"
        preset+="*10* load playlist from file,*11*;"
        preset+="*15* diff playlist,*16*>"
        preset+=";*12* auth describe,*13* log info,*14* moving info"
        let strlist=["LOADED","PLAYLIST_ID","LIBRARY_REQUEST_PLAYLISTS"]
        strlist.append_into(["In order to reload this preset, please clear text in the top left cell and run the script on any cell without a subgrid. The script determines a move action if the top left cell of a selection has a subgrid, so run the script when you are sure you\'ve done everything correctly (or make sure you can CTRL+Z)","Playlist ID, used by actions REQUEST_PLAYLIST and LOAD_FILE_PLAYLIST.","Action to perform when running the script with a loaded preset."])
        strlist.append_into(["LIBRARY_REQUEST_PLAYLISTS","Request playlists from library and saves to temporary file (/scripts/ytapi/playlists.csv). Authentication must be present.","REQUEST_PLAYLIST","Request songs in playlist. Will save the data to file (/scripts/ytapi/*PLAYLIST_ID*.csv). Large playlists may require waiting for some time.","LOAD_FILE_PLAYLIST","Load playlist tracks from file (/scripts/ytapi/*PLAYLIST_ID*.csv). Faster than requesting playlist, but may be outdated. Moving tracks in outdated playlist can result in unexpected track order."])
        strlist.append_into(["Please follow instructions at https://ytmusicapi.readthedocs.io/en/latest/setup.html to get authentication headers data, and put it in the file *INSTALL_DIRECTORY*/scripts/ytapi/headers_auth.json. Using Firefox is highly recommended.","Errors will be attempted to be logged under *CURRENT_TREESHEET_FILE_DIRECTORY*/log.txt. It is recommended to check the log file after each run.","Moving tracks in playlist is done by selecting a range of rows with index columns, and then creating a 1x1 subgrid in the index column on the top row in selection, and writing the insertion index in the cell. The subgrid is automatically removed. Example: range (1-5) with selected index of 9 will be moved before 9, therefore becoming (4-8). Note that the correct playlist ID must be present in the preset. Authentication must be present. Additionally, the time of the request is linear to the number being moved, so moving a large number of tracks may take a long time."])
        strlist.append_into(["REQUEST_AND_DIFF_CURRENT_PLAYLIST","Request playlist and diff with current one in file (/scripts/ytapi/*PLAYLIST_ID*.csv), saves the diff in a file, along with old playlist. See log file for more info."])
        if_err_print_return(l_l,ts_write_from_preset(preset,strlist,3,3))
        return
    move_in_grid(2,0)//action
    let action=ts_get_text()
    if action=="LIBRARY_REQUEST_PLAYLISTS":
        print_f(l_l,"Requesting playlists from library.")
        let status, buffer = launch_subprocess(["python", "{api_location}\\ytapitest.py","-p"])
        if_err_print_return(l_l,status,"Python script returned error status {status}: {buffer}")
        let result=read_lines(buffer)
        if result:
            let len=length(result)
            let preset2="[3](3)*0* playlist count <?5,{len+1};*1* title, *2* description, *3* count, *4* author, *5* id"
            let strlist2=["Playlists: {len}","Title","Description","Amount of Tracks","Author","Playlist ID"]
            if_err_print_return(l_l,ts_write_from_preset(preset2,strlist2,4,4))
            for(len) i:
                let row,err=decode_csv(result[i])
                if_err_print_return(l_l,err,"Error decoding playlist at row {i}: {err}")
                if_err_print_return(l_l,length(row)!=5,"Invalid row length at row {i}: {row}")
                let preset3="[{i+1}]*0*,*1*,*2*,*3*,*4*"
                if_err_print_return(l_l,ts_write_from_preset(preset3,row,5,len+1))
            print_f(l_l,"Playlists requested successfully.")
        else:
            print_f(l_l,"Error reading playlist data from file: {buffer}")
            return
    elif action=="REQUEST_PLAYLIST" or action=="LOAD_FILE_PLAYLIST" or action=="REQUEST_AND_DIFF_CURRENT_PLAYLIST":
        move_in_grid(1,0)//playlist_id
        let playlist_id=ts_get_text()
        if action=="REQUEST_PLAYLIST":
            print_f(l_l,"Requesting playlist {playlist_id}")
            let status, buffer = launch_subprocess(["python", "{api_location}\\ytapitest.py","-i",playlist_id])
            if_err_print_return(l_l,status,"Python script returned error status {status}: {buffer}")
        elif action=="REQUEST_AND_DIFF_CURRENT_PLAYLIST":
            print_f(l_l,"Requesting playlist {playlist_id} and diffing with current one.")
            let status, buffer = launch_subprocess(["python", "{api_location}\\ytapitest.py","-i",playlist_id,"-d"])
            if_err_print_return(l_l,status,"Python script returned error status {status}: {buffer}")
            print_f(l_l,"Python run successfully: {buffer}")
        //!FIX WHEN THERE IS BUILT IN WAY TO GET INSTALL PATH
        let status, buffer = launch_subprocess(["python", "{api_location}\\location.py"])
        if_err_print_return(l_l,status,"Python script returned error status {status}: {buffer}")
        let result=read_lines("{buffer}\\{playlist_id}.csv")
        if result:
            let len=length(result)
            let preset3="[3](3)*0* track count <?6,{len};*1* index, *2* title, *3* artists, *4* album name, *5* duration, *6* link"
            let strlist3=["Playlist name: {result[0]}; Tracks: {len-1}","Index","Title","Artists","Album Name","Duration","Link to Track in Playlist Queue"]
            //the name of the playlist is in the first row of the file, therefore the length of the playlist is one less than the length of the file
            //the grid also thus needs to be the same length as the file, top row replaced by description row
            if_err_print_return(l_l,ts_write_from_preset(preset3,strlist3,4,4))
            for(len-1) i:
                let row,err=decode_csv(result[i+1])
                if_err_print_return(l_l,err,"Error decoding track {i}: {err}")
                //index is inferred, however also contains setVideoID, which is irrelevant
                if_err_print_return(l_l,length(row)!=6,"Invalid row length at track {i}: {row}")
                let preset4="[{i+1}]*0*,*1*,*2*,*3*,*4*,*5*"
                let strlist4=["{i}",row[0],row[1],row[2],row[3],"https://music.youtube.com/watch?v={row[4]}&list={playlist_id}"]
                if_err_print_return(l_l,ts_write_from_preset(preset4,strlist4,6,len+1))
            print_f(l_l,"Playlist loaded successfully.")
        else:
            print_f(l_l,"Error reading playlist data from file: {buffer}\\{playlist_id}.csv")
            return
    else:
        print_f(l_l,"Invalid action: {action}")
        return


def if_err_print_return(l_l,err,ret):
    if err:
        print_f(l_l,ret)
        return from ytapi

def if_err_print_return(l_l,err):
    if err:
        print_f(l_l,err)
        return from ytapi