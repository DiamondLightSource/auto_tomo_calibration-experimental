def f(a,b,c,d):
    print a,b,c,d
    
f(b=2,a=1,d=4,c=3)

myDict = {'b':2, 'a':1, 'd':4, 'c':3}

f(**myDict)