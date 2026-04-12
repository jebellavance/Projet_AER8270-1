import numpy as np
from Vector3 import Vector3
from vortexRing import vortexRing as panel
from math import *
import sys
from scipy import linalg

class VLM:
	def __init__(self, ni=5, nj=10, chordRoot=1.0, chordTip=1.0, twistRoot=0.0, twistTip=0.0, span=5.0, sweep=30.0, Sref = 1.0, referencePoint=[0.0,0.0,0.0], wingType=1, alphaRange = [0.0]):

		self.size = ni * nj

		self.A   = np.zeros((self.size,self.size))
		self.rhs = np.zeros(self.size)
		self.inducedDownwash = [np.zeros((self.size,self.size)), np.zeros((self.size,self.size)), np.zeros((self.size,self.size))]
		
		self.nw = 1
		self.panels     = []
		self.wakePanels = []
		self.ni = ni
		self.nj = nj

		self.gamma   = np.zeros(self.size)
		self.gammaij = np.zeros(self.size)

		self.liftAxis = Vector3(0.0,0.0,1.0)

		self.Sref      = Sref
		self.chordRoot = chordRoot
		self.chordTip  = chordTip
		self.cavg      = 0.5 * (chordRoot + chordTip)
		self.twistRoot = twistRoot
		self.twistTip  = twistTip
		self.span      = span
		self.sweep     = sweep * pi / 180.0
		self.referencePoint = Vector3(referencePoint[0],
			                          referencePoint[1],
			                          referencePoint[2])
		self.wingType = wingType

		self.CL = []
		self.CD = []
		self.CM = []
		self.spanLoad = []

		self.alphaRange = alphaRange
		self.Ufree = Vector3(1.0,0.0,0.0)
		self.rho = 1.0

	def calcA(self):

		self.A *= 0.0
		self.inducedDownwash[0] *= 0.0
		self.inducedDownwash[1] *= 0.0
		self.inducedDownwash[2] *= 0.0

		for j in range(0,self.nj):
			for i in range(0,self.ni):

				ia = self.ni*j + i
				panel = self.panels[ia]
				collocationPoint = panel.center
				normal = panel.normal
				
				for j2 in range(0,self.nj):
					for i2 in range(0,self.ni):
						ia2 = self.ni*j2+i2

						panel2 = self.panels[ia2]

						u = panel2.influence(collocationPoint, Sym=True)
						downWash = panel2.influence(collocationPoint, Sym=True, boundInfluence=False)

						# Ajout de l'influence du sillage
						if (i2 == self.ni-1):
							for n in range(0,self.nw):
								iaw = self.nw*j2 + n
								wakePanel = self.wakePanels[iaw]
								u += wakePanel.influence(collocationPoint, Sym=True)
								downWash += wakePanel.influence(collocationPoint, Sym=True, boundInfluence=False)

						self.A[ia,ia2] += u.dot(normal)

						self.inducedDownwash[0][ia,ia2] += downWash[0]
						self.inducedDownwash[1][ia,ia2] += downWash[1]
						self.inducedDownwash[2][ia,ia2] += downWash[2]

	def calcRHS(self):
		for i,r in enumerate(self.rhs):
			self.rhs[i] = -self.Ufree.dot(self.panels[i].normal)


	def solve(self):
		self.gamma = np.linalg.solve(self.A,self.rhs)


	def postProcess(self):
		for j in range(0,self.nj):
			for i in range(0,self.ni):
				ia = self.ni*j + i
				if (i == 0):
					self.gammaij[ia] = self.gamma[ia]
				else:
					iam = ia - 1
					self.gammaij[ia] = self.gamma[ia] - self.gamma[iam]

	def computeForcesAndMoment(self):
		self.CL.append(0.0)
		self.CM.append(0.0)
		self.CD.append(0.0)

		inducedDownwashX = np.dot(self.inducedDownwash[0],self.gamma)
		inducedDownwashY = np.dot(self.inducedDownwash[1],self.gamma)
		inducedDownwashZ = np.dot(self.inducedDownwash[2],self.gamma)

		for index,panel in enumerate(self.panels):
			force = self.Ufree.crossProduct(panel.dl()) * self.rho * self.gammaij[index]

			distToRefrence = self.referencePoint - panel.forceActingPoint()
			moment = force.crossProduct(distToRefrence)

			downWash = Vector3(inducedDownwashX[index], inducedDownwashY[index], inducedDownwashZ[index])

			self.CL[-1] += force.dot(self.liftAxis)
			self.CM[-1] += moment[1]
			self.CD[-1] -= self.rho * downWash.dot(self.liftAxis) * self.gammaij[index] * panel.dy()

		self.CL[-1] /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * self.Sref)
		self.CD[-1] /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * self.Sref)
		self.CM[-1] /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * self.Sref * self.cavg)

	def writeSpanload(self,outputfile):
		ypos = np.zeros(self.nj)
		cl_sec = np.zeros(self.nj) 
		for j in range(self.nj):
			area = 0.0
			cl = 0.0
			for i in range(self.ni):
				ia = self.ni*j + i
				panel = self.panels[ia]
				area += panel.Area() 
				force = self.Ufree.crossProduct(panel.dl()) * self.rho * self.gammaij[ia]
				cl += force.dot(self.liftAxis)

			cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * area)

			ypos[j] = self.panels[self.ni * j].forceActingPoint()[1]
			cl_sec[j] = cl

		fid = open(outputfile, 'w')
		fid.write("VARIABLES= \"Y\",\"Cl\"\n")
		for i,y in enumerate(ypos):
			fid.write("%.4lf %.4lf\n" % (y, cl_sec[i]))
		fid.close()
        
	
	def writeSolution(self,outputfile):
		out = 'Variables=\"X\",\"Y\",\"Z\",\"GAMMA\"\n'
		out += 'ZONE T=\"WING\" i=%d,j=%d,k=1, ZONETYPE=Ordered\nDATAPACKING=BLOCK\nVARLOCATION=([4]=CELLCENTERED)\n'%(self.ni+1,self.nj+1)

		for j in range(0,self.nj):
			for i in range(0,self.ni):
				ia = self.ni*j + i
				pan = self.panels[ia]
				out += '%lf '%(pan.p1.x)
			out += '%lf '%(pan.p2.x)

		for i in range(0,self.ni):
			ia = self.ni*j + i
			pan = self.panels[ia]
			out += '%lf '%(pan.p4.x)
		out += '%lf\n'%(pan.p3.x)

		for j in range(0,self.nj):
			for i in range(0,self.ni):
				ia = self.ni*j + i
				pan = self.panels[ia]
				out += '%lf '%(pan.p1.y)
			out += '%lf '%(pan.p2.y)

		for i in range(0,self.ni):
			ia = self.ni*j + i
			pan = self.panels[ia]
			out += '%lf '%(pan.p4.y)
		out += '%lf\n'%(pan.p3.y)

		for j in range(0,self.nj):
			for i in range(0,self.ni):
				ia = self.ni*j + i
				pan = self.panels[ia]
				out += '%lf '%(pan.p1.z)
			out += '%lf '%(pan.p2.z)

		for i in range(0,self.ni):
			ia = self.ni*j + i
			pan = self.panels[ia]
			out += '%lf '%(pan.p4.z)
		out += '%lf\n'%(pan.p3.z)

		for j in range(0,self.nj):
			for i in range(0,self.ni):
				ia = self.ni*j + i
				pan = self.panels[ia]
				out += '%lf '%(self.gamma[ia])


		f = open(outputfile,'w')
		f.write(out)
		f.close()


	def initializeWing(self):
		dy = self.span/float(self.nj)
		y = 0.0
		yNext = y + dy
		for j in range(self.nj):
			eta     = y / self.span
			etaNext = yNext / self.span

			twist = (1.0 - eta) * self.twistRoot + eta * self.twistTip			
			twistNext = (1.0 - etaNext) * self.twistRoot + etaNext * self.twistTip

			chord = Vector3((1.0 - eta) * self.chordRoot + eta * self.chordTip, 0.0, 0.0).rotate(0.0,twist,0.0)
			chordNext = Vector3((1.0 - etaNext) * self.chordRoot + etaNext * self.chordTip, 0.0, 0.0).rotate(0.0,twistNext, 0.0)
            
			pt = Vector3(tan(self.sweep) * y, y, 0.0)
			ptNext = Vector3(tan(self.sweep) * yNext, yNext, 0.0)

			ds = chord / float(self.ni)
			dsNext = chordNext / float(self.ni)

			for i in range(self.ni):
				p1 = pt
				p4 = ptNext

				pt = pt + ds
				ptNext += dsNext

				p2 = pt
				p3 = ptNext

				self.panels.append(panel(p1,p2,p3,p4))

			y += dy
			yNext += dy
            
	def initializeWingElliptic(self):
		theta_i = np.linspace(0.5*np.pi, np.pi, self.nj+1)
		y_i = -self.span*np.cos(theta_i)
		for j in range(self.nj):
			y     = y_i[j]
			yNext = y_i[j+1]
			eta     = y / self.span
			etaNext = yNext / self.span

			twist = (1.0 - eta) * self.twistRoot + eta * self.twistTip
			twistNext = (1.0 - etaNext) * self.twistRoot + etaNext * self.twistTip

			chord = Vector3(np.sqrt(1.0 - eta*eta*.995) * self.chordRoot, 0.0, 0.0).rotate(0.0,twist,0.0)
			chordNext = Vector3(np.sqrt(1.0 - etaNext*etaNext*.995) * self.chordRoot, 0.0, 0.0).rotate(0.0,twistNext,0.0)

			pt = Vector3(tan(self.sweep) * y, y, 0.0) + (Vector3(self.chordRoot,0.0,0.0)-chord)*0.5
			ptNext = Vector3(tan(self.sweep) * yNext, yNext, 0.0) + (Vector3(self.chordRoot,0.0,0.0)-chordNext)*0.5

			ds = chord / float(self.ni)
			dsNext = chordNext / float(self.ni)

			for i in range(self.ni):
				p1 = pt
				p4 = ptNext

				pt = pt + ds
				ptNext += dsNext

				p2 = pt
				p3 = ptNext

				self.panels.append(panel(p1,p2,p3,p4))

	def initializeWingCosine(self):
       
		theta_i = np.linspace(0.5*np.pi, np.pi, self.nj+1)
		y_i = -self.span*np.cos(theta_i)
		dy = self.span/float(self.nj)

		for j in range(self.nj):
			y     = y_i[j]
			yNext = y_i[j+1]
			eta     = y / self.span
			etaNext = yNext / self.span

			twist = (1.0 - eta) * self.twistRoot + eta * self.twistTip
			twistNext = (1.0 - etaNext) * self.twistRoot + etaNext * self.twistTip

			chord = Vector3((1.0 - eta) * self.chordRoot + eta * self.chordTip, 0.0, 0.0).rotate(0.0,twist,0.0)
			chordNext = Vector3((1.0 - etaNext) * self.chordRoot + etaNext * self.chordTip, 0.0, 0.0).rotate(0.0,twistNext,0.0)

			pt = Vector3(tan(self.sweep) * y, y, 0.0)
			ptNext = Vector3(tan(self.sweep) * yNext, yNext, 0.0)

			ds = chord / float(self.ni)
			dsNext = chordNext / float(self.ni)

			for i in range(self.ni):
				p1 = pt
				p4 = ptNext

				pt = pt + ds
				ptNext += dsNext

				p2 = pt
				p3 = ptNext

				self.panels.append(panel(p1,p2,p3,p4))


	def initializeWake(self):
		i = self.ni-1
		for j in range(0,self.nj):
			ia = self.ni*j + i
			pan = self.panels[ia]
			p1 = pan.p2
			p4 = pan.p3
			p2 = p1 + self.Ufree * 100.0 * self.chordRoot
			p3 = p4 + self.Ufree * 100.0 * self.chordRoot
			self.wakePanels.append(panel(p1,p2,p3,p4))

	def updateWake(self):
		i = self.ni-1
		for j in range(0,self.nj):
			ia = self.ni*j + i
			pan = self.panels[ia]
			p1 = pan.p2
			p4 = pan.p3
			p2 = p1 + self.Ufree * 100.0 * self.chordRoot
			p3 = p4 + self.Ufree * 100.0 * self.chordRoot
			self.wakePanels[j] = panel(p1,p2,p3,p4)


	def updateFreeStream(self,alpha):
		self.Ufree = Vector3(cos(alpha * pi / 180.0), 0.0, sin(alpha * pi / 180.0))
		self.liftAxis = Vector3(-sin(alpha * pi / 180.0), 0.0, cos(alpha * pi / 180.0))

	def run(self):

		if self.wingType == 1:
			self.initializeWing()
		elif self.wingType == 2:
 			self.initializeWingCosine()
		elif self.wingType == 3:
			self.initializeWingElliptic()
		else:
			print("Wrong Input, defaulting to regular wing discretization")
			self.initializeWing()

		
		self.initializeWake()

		for alpha in self.alphaRange:
			self.updateFreeStream(alpha)
			self.updateWake()
			self.calcA()
			self.calcRHS()
			self.solve()
			self.postProcess()
			self.computeForcesAndMoment()
			self.writeSpanload('Spanload_A%.2lf.dat' % alpha)
			self.writeSolution('3D_sol_A%.2lf.dat' % alpha)

			print('Alpha= %.2lf CL= %.3lf CD= %.4lf CM= %.4lf' % (alpha, self.CL[-1], self.CD[-1], self.CM[-1]))


if __name__ == '__main__':
	prob = VLM(ni=5,
		       nj=20,
		       chordRoot=1.0,
		       chordTip=0.3,
		       twistRoot=0.0,
		       twistTip=0.0,
		       span=5.0,
		       sweep=0.0,
		       Sref =3.250,
		       referencePoint=[0.25,0.0,0.0],
		       wingType=1,
		       alphaRange = [0.0,5.0, 10.0, 20.0])
	prob.run()
