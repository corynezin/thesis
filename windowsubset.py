def getWindowSubset(M,C,n):
    C = min(C,M)
    n1 = n - C//2
    n2 = n1 + C - 1
    if n1 >= 1 and n2 <= M:
        i1 = n1; i2 = n2
        print(1)
    elif n1 < 1 and n2 <= M:
        i1 = 1; i2 = C
        print(2)
    elif n1 >= 1 and n2 >= M:
        i1 = M - C + 1; i2 = M
        print(3)
    else:
        i1  = 1; i2 = M
        print(4)
    return i1,i2

M = 11
C = 9
for n in range(1,M+1):
    print(n,getWindowSubset(M,C,n))
    
