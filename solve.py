import numpy as np
import math
import itertools

Test = [1, 2, 3, 4]
def check(x, y):
    if x > y:
        return False
    return True

x = itertools.permutations(Test, 4)
trainsleft = []
x = list(x)
print(x[0])
print(len(x))
for i in x:
    if i[0] == 1:
        trainsleft.append(1)
    elif i[1] == 1:
        trainsleft.append(2)
    elif i[2] == 1:
        if check(i[0], i[1]):
            trainsleft.append(2)
        else:
            trainsleft.append(3)
    elif i[3] == 1:
        if check(i[0], i[1]):
            pass
        #check for three cases,
        #if i[4] == 1: check 4 cases
            
        
    
    
    
    
print(trainsleft)

    
    
    
