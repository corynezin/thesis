import sys, os, re

file_list = []

with open('./log') as f:
    for line in f:
        m = 'Processing'
        L = len(m)
        if line[0:L] == m:
            filename = line[L+1:-1]
            print( filename )
            file_list.append(filename)

print('Number of files: ',len(file_list))
