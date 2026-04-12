from Vector3 import Vector3
from sourcePanel import sourcePanel
from math import *
import numpy as np

def ReadPoints(inputFile):
    data = np.loadtxt(inputFile, dtype=float, usecols = (0,1))
    listOfPanels = []
    for i in range(0,len(data)-1):
        pt     = data[i,:]
        ptNext = data[i+1,:]
        p1 = Vector3(pt[0],     0.0, pt[1])
        p2 = Vector3(ptNext[0], 0.0, ptNext[1])
        listOfPanels.append(sourcePanel(p1,p2))
    return listOfPanels


def GenerateNACA4digit(maxCamber=0.0, positionOfMaxCamber=0.0, thickness=12.0, pointsPerSurface=50):
    M = maxCamber*0.01;
    P = positionOfMaxCamber*0.1;
    XX = thickness*0.01;
    a0 = 0.2969;
    a1 = -0.126;
    a2 = -0.3516;
    a3 = 0.2843;
    a4 = -0.1036;
    dxpi = pi/(pointsPerSurface-1);
    X = -dxpi;

    xu = np.zeros(pointsPerSurface)
    yu = np.zeros(pointsPerSurface)
    xl = np.zeros(pointsPerSurface)
    yl = np.zeros(pointsPerSurface)

    Xn = np.zeros(pointsPerSurface * 2 -1)
    Yn = np.zeros(pointsPerSurface * 2 -1)

    for i in range(0,pointsPerSurface):

        X += dxpi;
        x = (1.0-cos(X))/2.0;

        if (x<P):
            dydc = 2.0*M*(P-x)/(P*P);
            yn = M*(2.0*P*x-x*x)/(P*P);
        else:
            dydc = 2.0*M*(P-x)/((1.0-P)*(1.0-P));
            yn = M*(1.0-2.0*P+2.0*P*x-x*x)/((1.0-P)*(1.0-P));

        yt = XX/0.2*(a0*sqrt(x)+a1*x+a2*x*x+a3*pow(x,3)+a4*pow(x,4));
        theta = atan(dydc);

        xu = x-yt*sin(theta);
        xl = x+yt*sin(theta);
        yu = yn+yt*cos(theta);
        yl = yn-yt*cos(theta);

        if (i == 0):
            Xn[i+pointsPerSurface-1-i*2] = xl;
            Yn[i+pointsPerSurface-1-i*2] = yl;

        else:
            Xn[i+pointsPerSurface-1-i*2] = xl;
            Yn[i+pointsPerSurface-1-i*2] = yl;
            Xn[i+pointsPerSurface-1] = xu;
            Yn[i+pointsPerSurface-1] = yu;

    listOfPanels = []

    for i in range(0,len(Xn)-1):
        p1 = Vector3(Xn[i],   0.0, Yn[i])
        p2 = Vector3(Xn[i+1], 0.0, Yn[i+1])
        listOfPanels.append(sourcePanel(p1,p2))

    return listOfPanels
