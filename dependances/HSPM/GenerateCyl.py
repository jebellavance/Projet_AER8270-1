from Vector3 import Vector3
from sourcePanel import sourcePanel
from math import *
import numpy as np

def GenerateCylinder(radius=1.0, N=20, outputFile='cylinder.dat'):
    # allocate arrays
    x      = np.zeros(N)
    y      = np.zeros(N)
    theta  = np.zeros(N)
    dtheta = 2.*pi/(N-1)
    listOfPanels = []

    #create coodinates
    x[0] = radius
    for i in range(1,N):
        theta[i] = theta[i-1]-dtheta
        x[i]     = radius*cos(theta[i])
        y[i]     = radius*sin(theta[i])
        # build list of panels
        p1 = Vector3(x[i-1], 0.0, y[i-1])
        p2 = Vector3(x[i]  , 0.0, y[i]  )
        listOfPanels.append(sourcePanel(p1,p2))
    #last panel
    # p1 = Vector3(x[N-1], 0.0, y[N-1])
    # p2 = Vector3(x[0]  , 0.0, y[0]  )
    # listOfPanels.append(sourcePanel(p1,p2))

    # write to file
    with open(outputFile,'w') as fout:
        for i in range(N):
            fout.write("{:.6e} {:.6e}\n".format(x[i],y[i]))
        # fout.write("{:.6e} {:.6e}\n".format(x[0],y[0]))

    return listOfPanels

GenerateCylinder()
