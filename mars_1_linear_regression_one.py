# This example demonstrates Linear Regression 
# demo for one variable 
# Validating for formula for linearRegression one
# Variable

X=[]
C=[]

# Generating Test Data
for i in range(50,60):
    X.append((i*9/5)+32)
    C.append(i)

print (X)
print (C)

a,b,c,d=0,0,0,0
x,y=0,0

# Core
for i,j in zip(X,C):
    a+=(i*j)
    x+=j
    y+=i
    c+=(j*j)

a=len(X)*a
b=x*y
c=len(X)*c
d=x*x

m=((a-b)/(c-d))
c=(1/(len(X)))*(y-(m*x))

# m,c represents constants in y=mx+c
print ("predicted m : ",m," c : ",c)
print ("actual m : 1.8 c : 32.0")

