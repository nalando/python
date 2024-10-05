import numpy as np
import matplotlib.pyplot as plt

def A(x):
    if x<8 and 6<=x:
        return (x-6)/2
    elif x==8:
        return 1
    elif 8<x<=10:
        return -(x-10)/2
    else:
        return 0
X = np.linspace(6,10,5)
y = np.array([A(i) for i in X])
plt.plot(X,y,c= 'blue',label=r'$ \alpha_{A4}$')

def B(x):
    if 8<=x<10:
        return (x-8)/2
    elif x==10:
        return 1
    elif 10<x<=12:
        return -(x-12)/2
    else:
        return 0
X = np.linspace(8,12,5)
y = np.array([B(i) for i in X])
plt.plot(X,y,c= 'red',label=r'$ \alpha_{A5}$')

def C(x):
    if 0<=x<2:
        return x/2
    elif x==2:
        return 1
    elif 2<x<=4:
        return -(x-4)/2
    else:
        return 0
X = np.linspace(0,4,5)
y = np.array([C(i) for i in X])
plt.plot(X,y,c= 'gray',label=r'$ \alpha_{A1}$')

def D(x):
    if x<4 and 2<=x:
        return (x-2)/2
    elif x==4:
        return 1
    elif 4<x<=6:
        return -(x-6)/2
    else:
        return 0
X = np.linspace(2,6,5)
y = np.array([D(i) for i in X])
plt.plot(X,y,c= 'green',label=r'$ \alpha_{A2}$')

def E(x):
    if x<6 and 4<=x:
        return (x-4)/2
    elif x==6:
        return 1
    elif 6<x<=8:
        return -(x-8)/2
    else:
        return 0
X = np.linspace(4,8,5)
y = np.array([E(i) for i in X])
plt.plot(X,y,c= 'purple',label=r'$ \alpha_{A3}$')

plt.ylim(0,2)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [2,3,4,0,1]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])



plt.show()