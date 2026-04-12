from math import*
import numpy as np
from Vector3 import Vector3

class vortexRing:
	"""docstring for panel"""
	def __init__(self, p1,p2,p3,p4):
		self.p1 = p1
		self.p2 = p2
		self.p3 = p3
		self.p4 = p4
		self.gamma = []
		self.gammaij = []

		self.area = self.Area()
		self.normal = self.Normal()
		self.center = self.collocation()

	def dl(self):
		return self.p4 - self.p1

	def forceActingPoint(self):
		return (self.p1 + self.p4) * 0.5

	def dy(self):
		return (self.p4.y - self.p1.y)

	def Normal(self):
		return ((self.p2-self.p4).crossProduct(self.p3-self.p1)).Normalized()

	def collocation(self):
		return (self.p1 + self.p2 + self.p3 +self.p4)*0.25

	def Area(self):
		b = self.p3-self.p1
		f = self.p2-self.p1
		e = self.p4-self.p1
		s1 = f.crossProduct(b)
		s2 = b.crossProduct(e)
		return 0.5 * (s1.Magnitude() + s2.Magnitude())

	def influence(self,collocationPoint,Sym=True,boundInfluence=True):
		SYM = 0
		u = Vector3(0.0,0.0,0.0)
		rcut = 1.0e-12
		
		if (Sym): SYM=1
		for sym in range(0,SYM+1):
			x = collocationPoint.x
			y = collocationPoint.y
			if (sym): y *= -1.0
			z = collocationPoint.z

			if (boundInfluence):
				edges = [0,1,2,3]
			else:
				edges = [1,3]

			for i in edges:
				if (i == 0):
					x1 = self.p1.x
					y1 = self.p1.y
					z1 = self.p1.z
					x2 = self.p4.x
					y2 = self.p4.y
					z2 = self.p4.z
				elif (i == 1):
					x1 = self.p4.x 
					x2 = self.p3.x
					y1 = self.p4.y 
					y2 = self.p3.y
					z1 = self.p4.z 
					z2 = self.p3.z
				elif (i == 2):
					x1 = self.p3.x 
					x2 = self.p2.x
					y1 = self.p3.y 
					y2 = self.p2.y
					z1 = self.p3.z 
					z2 = self.p2.z
				elif (i == 3):
					x1 = self.p2.x 
					x2 = self.p1.x
					y1 = self.p2.y 
					y2 = self.p1.y
					z1 = self.p2.z 
					z2 = self.p1.z
				r1r2x =   (y-y1)*(z-z2) - (z-z1)*(y-y2)
				r1r2y = -((x-x1)*(z-z2) - (z-z1)*(x-x2))
				r1r2z =   (x-x1)*(y-y2) - (y-y1)*(x-x2)

				r1 = sqrt((x-x1)**2+(y-y1)**2+(z-z1)**2)
				r2 = sqrt((x-x2)**2+(y-y2)**2+(z-z2)**2)
				r0 = Vector3(x2-x1,y2-y1,z2-z1)

				r1 = sqrt((x-x1)**2+(y-y1)**2+(z-z1)**2)
				r2 = sqrt((x-x2)**2+(y-y2)**2+(z-z2)**2)
				r0 = Vector3(x2-x1,y2-y1,z2-z1)

		        # //calculation of (r1 x r2)^2
				square = ((r1r2x)**2+(r1r2y)**2+(r1r2z)**2)

				if ((r1<rcut) or (r2<rcut) or (square<rcut)):
					pass
				else:
					r0r1 = (x2-x1)*(x-x1)+(y2-y1)*(y-y1)+(z2-z1)*(z-z1)
					r0r2 = (x2-x1)*(x-x2)+(y2-y1)*(y-y2)+(z2-z1)*(z-z2)
					coef = 1.0/(4.0*pi*square) * (r0r1/r1 - r0r2/r2)

					u.x += r1r2x * coef
					if (sym):
						u.y -= r1r2y * coef	
					else:
						u.y += r1r2y * coef
					u.z += r1r2z * coef
		return u

