# Name: Cory Nezin
# Date: 01/13/2018
# Goal: Enhance plotting utilities of PyPlot

import matplotlib.pyplot as plt

def vstem(y_arr, x_arr,c,label=None):
    # Plot each horizontal line:
    for (x,y) in zip(x_arr, y_arr):
        plt.plot( [0,x], [y,y], c+'-')

    plt.plot(x_arr, y_arr, c+'o',label=label)
    plt.plot([0,0], [y_arr.min(), y_arr.max()], 'k-')
