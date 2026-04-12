from math import *
import numpy as np
class Vector3:
	def __init__(self,x,y,z):
		self.x = x
		self.y = y
		self.z = z
	def __add__(self,other):
		x = self.x+other.x
		y = self.y+other.y
		z = self.z+other.z
		return Vector3(x,y,z)
	def __sub__(self,other):
		x = self.x-other.x
		y = self.y-other.y
		z = self.z-other.z
		return Vector3(x,y,z)
	def __mul__(self,value):
		x = self.x*value
		y = self.y*value
		z = self.z*value
		return Vector3(x,y,z)
	def __truediv__(self,value):
		x = self.x/value
		y = self.y/value
		z = self.z/value
		return Vector3(x,y,z)
	def __str__(self):
		return 'X\tY\tZ\n%f\t%f\t%f'%(self.x,self.y,self.z)
	def __getitem__(self,index):
		if (index == 0):
			return self.x
		elif (index == 1):
			return self.y
		elif (index == 2):
			return self.z
		else:
			print('Error: Index out of range for Vector3')
			exit()

	def dot(self,other):
		x = self.x*other.x
		y = self.y*other.y
		z = self.z*other.z
		return x+y+z

	def div(self,other):
		x = self.x/other.x
		y = self.y/other.y
		z = self.z/other.z
		return x+y+z

	def crossProduct(self,other):
		x = self.y*other.z-self.z*other.y
		y = self.z*other.x-self.x*other.z
		z = self.x*other.y-self.y*other.x
		return Vector3(x,y,z)

	def Magnitude(self):
		return sqrt(self.x**2 + self.y**2 + self.z**2)

	def Normalized(self):
		mag = self.Magnitude()
		return Vector3(self.x/mag,self.y/mag,self.z/mag)

	def array(self):
		a = np.zeros(3)
		a[0] = self.x
		a[1] = self.y
		a[2] = self.z
		return a

	def rotate(self,theta,phi,psi):
		rotationEuler = np.zeros((3,3))
		C_theta = cos(theta*pi/180.0)
		C_phi = cos(phi*pi/180.0)
		C_psi = cos(psi*pi/180.0)
		S_theta = sin(theta*pi/180.0)
		S_phi = sin(phi*pi/180.0)
		S_psi = sin(psi*pi/180.0)
		rotationEuler[0,0] = C_psi*C_theta 
		rotationEuler[0,1] = C_psi*S_theta*S_phi-S_psi*C_phi 
		rotationEuler[0,2] = C_psi*S_theta*C_phi+S_psi*S_phi
		rotationEuler[1,0] = S_psi*C_theta 
		rotationEuler[1,1] = S_psi*S_theta*S_phi+C_psi*C_phi 
		rotationEuler[1,2] = S_psi*S_theta*C_phi-C_psi*S_phi
		rotationEuler[2,0] = -S_theta      
		rotationEuler[2,1] = C_theta*S_phi                   
		rotationEuler[2,2] = C_theta*C_phi
		pt = np.zeros(3)
		pt[0] = self.x
		pt[1] = self.y
		pt[2] = self.z
		pt = rotationEuler.dot(pt)
		return Vector3(pt[0],pt[1],pt[2])

	def move(self,dx,dy,dz):
		pt = Vector3(self.x,self.y,self.z)
		ds = Vector3(dx,dy,dz)
		return (pt + ds)

	def rotateMove(self,theta,phi,psi,axis,dx,dy,dz):
		rotationEuler = np.zeros((3,3))
		C_theta = cos(theta*pi/180.0)
		C_phi = cos(phi*pi/180.0)
		C_psi = cos(psi*pi/180.0)
		S_theta = sin(theta*pi/180.0)
		S_phi = sin(phi*pi/180.0)
		S_psi = sin(psi*pi/180.0)
		rotationEuler[0,0] = C_psi*C_theta 
		rotationEuler[0,1] = C_psi*S_theta*S_phi-S_psi*C_phi 
		rotationEuler[0,2] = C_psi*S_theta*C_phi+S_psi*S_phi
		rotationEuler[1,0] = S_psi*C_theta 
		rotationEuler[1,1] = S_psi*S_theta*S_phi+C_psi*C_phi 
		rotationEuler[1,2] = S_psi*S_theta*C_phi-C_psi*S_phi
		rotationEuler[2,0] = -S_theta      
		rotationEuler[2,1] = C_theta*S_phi                   
		rotationEuler[2,2] = C_theta*C_phi
		pt = np.zeros(3)
		pt[0] = self.x - axis.x
		pt[1] = self.y - axis.y
		pt[2] = self.z - axis.z
		pt = rotationEuler.dot(pt)
		pts = Vector3(pt[0] + axis.x,pt[1] + axis.y,pt[2] + axis.z)
		ds = Vector3(dx,dy,dz)
		return (pts + ds)
