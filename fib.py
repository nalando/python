import time
x = 1
y = 1
print(x,end=",")
print(y,end=",")
while True:
    y = x + y
    print(y,end=",")
    time.sleep(1)
    x = x + y
    print(x,end=",")
    time.sleep(1)