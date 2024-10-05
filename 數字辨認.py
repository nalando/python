import time
n = 0
d = False
#定義函數數值判斷
def DEF (n):
    try:
        float (n)
        return True
    except:
        return False
while d == False :
    n=input("數字辨認器，我會辨認你所給是否大於0，請輸入大於0的數字:\n")
    time.sleep(0.5)
    print("請稍後，系統正在計算.")
    time.sleep(0.5)
    if DEF(n) == True:
        #>0
        if float(n) > 0:
           print(n,">0")
           d = True
           break
        else:
            print("請稍後，系統正在計算..")
            time.sleep(0.5)
        #=0
        if float(n) == 0:
            print(n,"=0")
            
        else:
            print("請稍後，系統正在計算...")
            time.sleep(0.5)    
        #<0
        if float(n) < 0:
            print(n,"<0")
            
    else:
        print("輸入錯誤，請再試一次\n")


    
    