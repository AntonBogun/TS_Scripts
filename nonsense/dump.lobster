def parse_encoded_string(str):
    //python's encode("utf-8") returns a byte string converted to text, so we need to decode it
    let bytes = str//string_to_unicode(str)//not needed because no non-ascii should be present
    let i=0
    let real = []::int
    while i<len(bytes):
        let c = bytes[i]
        if c==92://backslash
            let curr=0
            c = bytes[i+1]
            if c==120://x
                let hex = (bytes[i+2]-48)*16+(bytes[i+3]-48)
                let l=0
                if hex>=240:
                    curr = (hex-240)<<18
                    l=3
                elif hex>=224:
                    curr = (hex-224)<<12
                    l=2
                elif hex>=192:
                    curr = (hex-192)<<6
                    l=1
                else:
                    return "Invalid encoding at position "+i
                let j=0
                while(j<l):
                    if(bytes[i+4+j*4]!=92 or bytes[i+4+j*4+1]!=120 or 
                    bytes[i+4+j*4+2]<128 or bytes[i+4+j*4+2]>=192 or 
                    bytes[i+4+j*4+3]<128 or bytes[i+4+j*4+3]>=192):
                        return "Invalid encoding at position "+i
                    curr = curr + ((bytes[i+4+j*4+2]-48)*16+(bytes[i+4+j*4+3]-48)-