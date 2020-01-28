import numpy as np
import sys

def range2d(a,b=None):
	if b is None:
		a,b = a

	for x in range(a):
		for y in range(b):
			yield x,y

def circleArea(r):
	if r < 0:
		return np.zeros((0,0), dtype=bool)
	square = np.zeros((2*r+1, 2*r+1), dtype=bool)

	for x, y in range2d(2*r+1, 2*r+1):
		if np.hypot(x-r,y-r) <= r + .5:
			square[x,y] = True
	
	return square

def circularSection(r):
	smallerSquare = np.pad(circleArea(r-1), 1, mode="constant")
	square = circleArea(r)
	return np.logical_xor(smallerSquare, square)

for i in range(0,11):
	area = circularSection(i)
	print(f"# r={i}\n[", end="")
	for x,y in range2d(np.shape(area)):
		if y == 0: print()
		print("cɔ" if area[x,y] else "  ", end="")
	print(f"],")