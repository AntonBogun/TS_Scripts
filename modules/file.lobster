def read_lines(file:string) -> [string]?:
    let result=read_file(file)
    if result:
        return tokenize(result,"\n","")
    return nil

def write_lines(file:string, lines:[string]) -> int:
    return write_file(file,concat_string(lines,"\n"),0,1)//!fix when relative/full is inferred