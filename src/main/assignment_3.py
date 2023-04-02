"""
Created on Sun Apr  2 14:29:58 2023

@author: Janny
"""

import numpy as np
#1

def f(x,y):
    return x- (y*y)


def euler(x0,y0,xn,n):
    
    h = (xn-x0)/n
    
    for i in range(n):
        slope = f(x0, y0)
        ans = y0 + h * slope
        y0 = ans
        x0 = x0+h
    
    print(ans)
    print()

x0 = 0
xn = 2
y0 = 1
step = 10

euler(x0,y0,xn,step)

# 2

def f(x,y):
    return x-(y*y)

def rk4(x0,y0,xn,n):
    
    h = (xn-x0)/n
    
    for i in range(n):
        k1 = h * (f(x0, y0))
        k2 = h * (f((x0+h/2), (y0+k1/2)))
        k3 = h * (f((x0+h/2), (y0+k2/2)))
        k4 = h * (f((x0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        ans2 = y0 + k
        y0 = ans2
        x0 = x0+h
    
    print(ans2)
    print()
    
x0 = 0
xn = 2
y0 = 1
step = 10

rk4(x0,y0,xn,step)

#3

mat1 = [[2, -1, 1, 6],
        [1, 3, 1, 0],
        [-1, 5, 4, -3]]

for i in range(len(mat1)):

    max_row = i
    for j in range(i+1, len(mat1)):
        if abs(mat1[j][i]) > abs(mat1[max_row][i]):
            max_row = j

    mat1[i], mat1[max_row] = mat1[max_row], mat1[i]
    pivot = mat1[i][i]
    
    for j in range(i, len(mat1[i])):
        mat1[i][j] /= pivot

    for j in range(len(mat1)):
        if j != i:
            factor = mat1[j][i]
            
            for k in range(i, len(mat1[i])):
                mat1[j][k] -= factor * mat1[i][k]

x = [row[-1] for row in mat1]

print(x)
print()


#4

def L_U(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]
    
    return L, U
def mat_det(matrix):
    L, U = L_U(matrix)
    det = np.prod(np.diag(U))
    return det

mat4 = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])
det = mat_det(mat4)
L, U = L_U(mat4)

print("%.5f" % det)
print()
print(L)
print()
print(U)
print()

#5

mat5 = [[9,0,5,2,1],
        [3,9,1,2,1],
        [4,2,3,12,2],
        [3,2,4,0,8]]

def diag_dom(mat5):
    for i, row in enumerate(mat5):
        s = sum(abs(v) for j, v in enumerate(row) if i != j)
        if s > abs(row[i]):
            return False
    return True
    
print(diag_dom(mat5))

print()

#6

# Creating numpy array
mat6 = np.array([[2,2,1],[2,3,0],[1,0,2]])

# Check all eigen value
pos = np.all(np.linalg.eigvals(mat6) > 0)

print(pos)
