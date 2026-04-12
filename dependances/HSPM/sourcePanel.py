from math import*
import numpy as np
from Vector3 import Vector3

class sourcePanel:
	"""docstring for panel"""
	def __init__(self, p1,p2):
		self.p1 = p1
		self.p2 = p2
		self.CP = 0.0
		self.Vtangentielle = 0.0

	def setCP(self, vTang):
		self.Vtangentielle = vTang
		self.CP = 1.0 - vTang**2
		return self.CP

	def getForceVector(self):
		return self.normal() * -self.CP * self.area()

	#returning edge vector
	def dl(self):
		return self.p2 - self.p1

	def dx(self):
		return self.dl()[0]

	def dz(self):
		return self.dl()[2]

	def normal(self):
		return (self.dl().crossProduct(Vector3(0.0,1.0,0.0))).Normalized()

	def collocationPoint(self):
		return (self.p1 + self.p2) * 0.5

	def area(self):
		return self.dl().Magnitude()

	def sinTheta(self):
		dl = self.dl()
		return dl[2] / dl.Magnitude()

	def cosTheta(self):
		dl = self.dl()
		return dl[0] / dl.Magnitude()

	# Computing the influence of this panel on referencePanel
	def influence(self, referencePanel, i, j):
		flog = 0.0
		ftan = pi

		r1 = referencePanel.collocationPoint() - self.p1
		r2 = referencePanel.collocationPoint() - self.p2

		if (j != i):
			flog = 0.5 * log ( ( r2[0]**2 + r2[2]**2) / ( r1[0]**2 + r1[2]**2 ) )
			ftan = atan2(r2[2] * r1[0] - r2[0] * r1[2], r2[0] * r1[0] + r2[2] * r1[2])

		ctimtj = referencePanel.cosTheta() * self.cosTheta() + referencePanel.sinTheta() * self.sinTheta()
		stimtj = referencePanel.sinTheta() * self.cosTheta() - referencePanel.cosTheta() * self.sinTheta()

		influence = 0.5 / pi * (ftan * ctimtj + flog * stimtj)
		B = 0.5 / pi * (flog * ctimtj - ftan * stimtj);

		return influence,B
