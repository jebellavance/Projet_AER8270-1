from Vector3 import Vector3
from sourcePanel import sourcePanel
import numpy as np
import geometryGenerator
from math import *

class HSPM(object):
    """docstring for HSPM"""
    def __init__(self, listOfPanels = None, alphaRange = [5.0], referencePoint=[0.0,0.0,0.0]):
        super(HSPM, self).__init__()
        self.size = len(listOfPanels)
        self.listOfPanels = listOfPanels
        self.A = np.zeros((self.size+1,self.size+1))
        self.RHS = np.zeros(self.size+1)
        self.alphaRange = alphaRange
        self.singularityStrenghts = np.zeros(self.size+1)
        self.CP = np.zeros(self.size)
        self.deltaCPvalarezo = []
        self.CL = []
        self.CD = []
        self.CM = []
        self.liftAxis = Vector3(0.0,0.0,1.0)
        self.flowAxis = Vector3(1.0,0.0,0.0)
        self.referencePoint = Vector3(referencePoint[0], referencePoint[1], referencePoint[2])

    def computeInfluenceMatrix(self):
        self.A *= 0.0
        # Computing source induced velocity influence between the sources on themselves
        for i in range(self.size):
            for j in range(self.size):

                influence,B = self.listOfPanels[j].influence(self.listOfPanels[i],i,j)
                # Filling the influence matrix of the linear system
                self.A[i,j] = influence
                self.A[i,-1] += B

                if (i == 0 or i == (self.size - 1)):
                    self.A[-1,j] -= B
                    self.A[-1,-1] += self.A[i,j]

    def computeRHS(self,alpha):
        # Computing the right-hand side vector of linear system (Freestream)
        for i in range(self.size):
            panel = self.listOfPanels[i]
            self.RHS[i] = panel.sinTheta() * cos(alpha) - panel.cosTheta() * sin(alpha)

        self.RHS[-1]  = - (self.listOfPanels[0].cosTheta() + self.listOfPanels[-1].cosTheta()) * cos(alpha)
        self.RHS[-1] -=   (self.listOfPanels[0].sinTheta() + self.listOfPanels[-1].sinTheta()) * sin(alpha)

    def solve(self):
        self.singularityStrenghts = np.linalg.solve(self.A, self.RHS)

    def computeCPandVtang(self):
        # Computing the total velocity on each panel to compute CP
        for i in range(self.size):
            vTang = self.flowAxis.dot(self.listOfPanels[i].dl().Normalized())
            for j in range(self.size):

                influence,B = self.listOfPanels[j].influence(self.listOfPanels[i],i,j)
                vTang += self.singularityStrenghts[-1] * influence - self.singularityStrenghts[j] * B

            self.CP[i] = self.listOfPanels[i].setCP(vTang)

        cpMin = np.min(self.CP)
        cpTE = (self.CP[0] + self.CP[-1]) * 0.5
        self.deltaCPvalarezo.append(abs(cpTE - cpMin))

    def computeForces(self):
        self.CL.append(0.0)
        self.CD.append(0.0)
        self.CM.append(0.0)

        for i,panel in enumerate(self.listOfPanels):
            force = panel.getForceVector()
            self.CL[-1] += force.dot(self.liftAxis)
            self.CD[-1] += force.dot(self.flowAxis)
            self.CM[-1] += force.crossProduct(self.referencePoint-panel.collocationPoint())[1]

    def updateFlow(self, alpha):
        # Setting lift and drag axis for new aangle of attack
        self.liftAxis = Vector3(-sin(alpha), 0.0, cos(alpha))
        self.flowAxis = Vector3( cos(alpha), 0.0, sin(alpha))

    def writeCP(self,outputFile):
        fid = open(outputFile, 'w')
        fid.write("VARIABLES= \"X\",\"Y\",\"Z\",\"CP\"\n")
        for i,panel in enumerate(self.listOfPanels):
            cp = self.CP[i]
            if (i == 0):
                cp += self.CP[-1]
            else:
                cp += self.CP[i-1]

            cp *= 0.5

            fid.write("%.4lf %.4lf %.4lf %.4lf\n" % (panel.p1[0], panel.p1[1], panel.p1[2], cp))
            if (i == self.size-1):
                fid.write("%.4lf %.4lf %.4lf %.4lf\n" % (panel.p2[0], panel.p2[1], panel.p2[2], (self.CP[-1]+self.CP[0])*0.5))

        fid.close()

    def findAlphaMaxClMax(self, valarezoCriterion=None):
        if (len(self.alphaRange) < 2):
            print('Error: Cannot find stall with only one solution!')
            exit()

        if (valarezoCriterion > np.max(self.deltaCPvalarezo)):
            print('Warning: Cannot find CL max, not enough alpha solution!')

        if (valarezoCriterion == None):
            print('Warning: No valarezo criterion given...')
            return None, None

        alphaMax = np.interp(valarezoCriterion, self.deltaCPvalarezo, self.alphaRange)
        clMax = np.interp(valarezoCriterion, self.deltaCPvalarezo, self.CL)

        return alphaMax, clMax

    def checkIfStall_valarezo(self, deltaCPcriterion):

        if (deltaCPcriterion <= np.max(self.deltaCPvalarezo)):
            return True
        else:
            return False

    def getUpperVtangential(self):
        offset = 3
        stagnationFacetId = np.argmax(self.CP[offset:-offset]) + offset
        pointsCoordinate = []
        Vtang = []

        # Test for node with velocity nearest 0
        Vtang1 = 0.5 * (self.listOfPanels[stagnationFacetId-1].Vtangentielle + self.listOfPanels[stagnationFacetId].Vtangentielle)
        Vtang2 = 0.5 * (self.listOfPanels[stagnationFacetId+1].Vtangentielle + self.listOfPanels[stagnationFacetId].Vtangentielle)

        if (abs(Vtang2) < abs(Vtang1)):
            # Stagnation point is p1 from facet, but p2 is closer to 0 velocity
            stagnationFacetId += 1

       

        for i in range(stagnationFacetId,self.size):
            pointsCoordinate.append(self.listOfPanels[i].p1)
            Vtang.append(abs(0.5 * (self.listOfPanels[i].Vtangentielle + self.listOfPanels[i-1].Vtangentielle)))
            

        pointsCoordinate.append(self.listOfPanels[-1].p2)
        Vtang.append(abs(0.5 * (self.listOfPanels[0].Vtangentielle + self.listOfPanels[-1].Vtangentielle)))

        # Forcing stagnation velocity to 0.0 to avoid errors due to sign change
        Vtang[0] = 0.0

        return pointsCoordinate,Vtang

    def getLowerVtangential(self):
        offset = 3
        stagnationFacetId = np.argmax(self.CP[offset:-offset]) + offset
        pointsCoordinate = []
        Vtang = []

        # Test for node with velocity nearest 0
        Vtang1 = 0.5 * (self.listOfPanels[stagnationFacetId-1].Vtangentielle + self.listOfPanels[stagnationFacetId].Vtangentielle)
        Vtang2 = 0.5 * (self.listOfPanels[stagnationFacetId+1].Vtangentielle + self.listOfPanels[stagnationFacetId].Vtangentielle)

        if (abs(Vtang2) < abs(Vtang1)):
            # Stagnation point is p1 from facet, but p2 is closer to 0 velocity
            stagnationFacetId += 1

       

        pointsCoordinate.append(self.listOfPanels[0].p1)
        Vtang.append(abs(0.5 * (self.listOfPanels[0].Vtangentielle + self.listOfPanels[-1].Vtangentielle)))

        for i in range(1,stagnationFacetId+1):
            pointsCoordinate.append(self.listOfPanels[i].p1)
            Vtang.append(abs(0.5 * (self.listOfPanels[i].Vtangentielle + self.listOfPanels[i-1].Vtangentielle)))
            

        # Forcing stagnation velocity to 0.0 to avoid errors due to sign change
        Vtang[-1] = 0.0

        # Returns reversed arrays
        return pointsCoordinate[::-1],Vtang[::-1]

    def run(self):
        for alpha in self.alphaRange:
            self.updateFlow(alpha * pi / 180.0)
            self.computeInfluenceMatrix()
            self.computeRHS(alpha * pi / 180.0)
            self.solve()
            self.computeCPandVtang()
            self.computeForces()
            self.writeCP("CPsol_A%.2lf.dat" % (alpha))

            print('Alpha= %.2lf CL= %.4lf CD= %.4lf CM= %.6lf' % (alpha, self.CL[-1], self.CD[-1], self.CM[-1]))

if __name__ == '__main__':

    # Creating the geometry and source panels
    panels = geometryGenerator.GenerateNACA4digit(maxCamber=0.0,
                                                   positionOfMaxCamber=0.0,
                                                   thickness=12.0,
                                                   pointsPerSurface=25)

    # Instantiating HSPM class to compute the pressure solution on the given geometry
    prob = HSPM(listOfPanels = panels,
                alphaRange = [0.0,5.0,10.0,20.0,25.0],
                referencePoint=[0.25,0.0,0.0])
    # Solving...
    prob.run()
