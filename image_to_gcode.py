#!/usr/bin/python3
#pylint: disable=no-member

import numpy as np
from scipy import ndimage
import imageio
from PIL import Image, ImageFilter
import constants


def enhance(image):
	pil_image = Image.fromarray(image.astype("uint8"), "RGBA")
	sharpened = pil_image.filter(ImageFilter.SHARPEN)
	return np.asarray(sharpened)

def sobel(image, threshold):
	Gx = ndimage.sobel(image, axis=0)
	Gy = ndimage.sobel(image, axis=1)
	G = np.hypot(Gx, Gy)

	shape = np.shape(G)
	result = np.zeros(shape[0:2], dtype=bool)
	print(np.shape(result))

	result[(G[:, :, 0] + G[:, :, 1] + G[:, :, 2] + G[:, :, 3]) >= threshold] = True
	return result

class CircularRange:
	def __init__(self, begin, end, value):
		self.begin, self.end, self.value = begin, end, value

	def __repr__(self):
		return f"[{self.begin},{self.end})->{self.value}"
	
	def halfway(self):
		return int((self.begin + self.end) / 2)


class EdgesToGcode:
	def __init__(self, edges):
		self.edges = edges
		self.seen = np.zeros(np.shape(edges), dtype=bool)
		self.xSize, self.ySize = np.shape(edges)
	
	def getCircularArray(self, center, r, smallerArray = None):
		circumferenceSize = len(constants.circumferences[r])
		circularArray = np.zeros(circumferenceSize, dtype=bool)

		if smallerArray is None:
			smallerArray = np.ones(1, dtype=bool)
		smallerSize = np.shape(smallerArray)[0]
		smallerToCurrentRatio = smallerSize / circumferenceSize

		for i in range(circumferenceSize):
			x = center[0] + constants.circumferences[r][i][0]
			y = center[1] + constants.circumferences[r][i][1]

			if x not in range(self.xSize) or y not in range(self.ySize):
				circularArray[i] = False # consider pixels outside of the image as not-edges
			else:
				iSmaller = i * smallerToCurrentRatio
				a, b = int(np.floor(iSmaller)), int(np.ceil(iSmaller))
				
				if smallerArray[a] == False and (b not in range(smallerSize) or smallerArray[b] == False):
					circularArray[i] = False # do not take into consideration not connected regions (roughly)
				else:
					circularArray[i] = self.edges[x, y]
		
		return circularArray

	def toCircularRanges(self, circularArray):
		ranges = []
		circumferenceSize = np.shape(circularArray)[0]

		lastValue, lastValueIndex = circularArray[0], 0
		for i in range(1, circumferenceSize):
			if circularArray[i] != lastValue:
				ranges.append(CircularRange(lastValueIndex, i, lastValue))
				lastValue, lastValueIndex = circularArray[i], i
		
		ranges.append(CircularRange(lastValueIndex, circumferenceSize, lastValue))
		if len(ranges) > 1 and ranges[-1].value == ranges[0].value:
			ranges[0].begin = ranges[-1].begin - circumferenceSize
			ranges.pop() # the last range is now contained in the first one
		return ranges
	
	def nextPoints(self, point):
		"""
		Returns the radius of the circle used to identify the points and
		the points toward which propagate, in a tuple `(radius, [point0, point1, ...])`
		"""

		bestRadius = 0
		circularArray = self.getCircularArray(point, 0)
		allRanges = [self.toCircularRanges(circularArray)]
		for radius in range(1, len(constants.circumferences)):
			circularArray = self.getCircularArray(point, radius, circularArray)
			allRanges.append(self.toCircularRanges(circularArray))
			if len(allRanges[radius]) > len(allRanges[bestRadius]):
				bestRadius = radius
			if len(allRanges[radius]) > 1 and len(allRanges[-1]) == len(allRanges[-2]):
				# two consecutive circular arrays with the same number>1 of ranges
				break
		
		circularRanges = allRanges[bestRadius]
		points = []
		for circularRange in circularRanges:
			if circularRange.value == True:
				circumferenceIndex = circularRange.halfway()
				x = point[0] + constants.circumferences[bestRadius][circumferenceIndex][0]
				y = point[1] + constants.circumferences[bestRadius][circumferenceIndex][1]

				if x in range(self.xSize) and y in range(self.ySize) and not self.seen[x, y]:
					points.append((x,y))
		
		return bestRadius, points
					
	def propagate(self, point):
		pass


def pokeballEdges():
	image = imageio.imread("pokeball_small.png")

	edges = sobel(image, 128.0)
	imageio.imwrite("pokeballsobel.png", edges.astype(float))

	return edges

def testEdges():
	image = imageio.imread("test_edges.png")
	edges = np.zeros((np.shape(image)[1], np.shape(image)[0]), dtype=bool)

	for x, y in np.ndindex(np.shape(image)[0:2]):
		edges[y][x] = (image[x][y][0] > 128 and image[x][y][1] > 128 and image[x][y][2] > 128)
	
	return edges

def main():
	edges = testEdges()

	print("-----------------")
	for x, y in np.ndindex(np.shape(edges)):
		if y == 0 and x != 0: print()
		print("cɔ" if edges[x,y] else "  ", end="")
	print("\n-----------------")

	circularArray = None
	converter = EdgesToGcode(edges)
	for i in range(11):
		circularArray = converter.getCircularArray((14, 7), i, circularArray)
		#print(circularArray)
		sections = converter.toCircularRanges(circularArray)
		print(sections)
	
	print(converter.nextPoints((14,7)))

	#print(", ".join([str(c)[1:-1] for c in constants.circumferences]))

if __name__ == "__main__":
	main()