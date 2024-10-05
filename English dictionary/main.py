f = open("word.txt")
que = input("你想查找甚麼單字\n")

#引用

a = f.readline()
a = a.split()

g = True

while g == True:  
  if a[0] == "00" or a[0] == que:
   g = False
  else:
   a = f.readline()
   a = a.split()
    
else:
  if a[0] == que:
    print(a[0].strip("'"),",",a[1].strip("'"))
  else:
    print("not found any about%s"%que)
    
f.close()
