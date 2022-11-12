l="ф".encode('utf-8')
print(str(l)[2:-1])
print(" ".join([bin(x)[2:] for x in l]))
print(0b10001000100)


print(0b11100000)