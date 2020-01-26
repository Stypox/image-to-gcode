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


class EdgesToGcode:
	def __init__(self, edges):
		self.edges = edges
		self.seen = np.zeros(np.shape(edges), dtype=bool)
		self.xSize, self.ySize = np.shape(edges)
	
	def getCircularSectionsArray(self, center, r, smallerArray = None):
		circumferenceSize = len(constants.circumferences[r])
		sectionsArray = np.zeros(circumferenceSize, dtype=bool)

		if smallerArray is None:
			smallerArray = np.ones(1, dtype=bool)
		smallerSize = np.shape(smallerArray)[0]
		smallerCurrentRatio = smallerSize / circumferenceSize

		for i in range(circumferenceSize):
			x = center[0] + constants.circumferences[r][i][0]
			y = center[1] + constants.circumferences[r][i][1]

			if x not in range(self.xSize) or y not in range(self.ySize):
				sectionsArray[i] = False # consider pixels outside of the image as not-edges
			else:
				iSmaller = i * smallerCurrentRatio
				a, b = int(np.floor(iSmaller)), int(np.ceil(iSmaller))
				
				if smallerArray[a] == False and (b not in range(smallerSize) or smallerArray[b] == False):
					sectionsArray[i] = False # do not take into consideration not connected regions (roughly)
				else:
					sectionsArray[i] = self.edges[x, y]
		
		return sectionsArray

	def circularSectionsRanges(self, circularSectionsArray):
		sections = [0]
		circumferenceSize = np.shape(circularSectionsArray)[0]

		lastValue = circularSectionsArray[0]
		for i in range(1, circumferenceSize):
			if circularSectionsArray[i] != lastValue:
				sections[-1] = (sections[-1], i, lastValue)
				sections.append(i)
				lastValue = circularSectionsArray[i]
		
		sections[-1] = (sections[-1], circumferenceSize, lastValue)
		if len(sections) > 1 and sections[-1][2] == sections[0][2]:
			sections[0] = (sections[-1][0] - circumferenceSize, sections[0][1], sections[0][2])
			return sections[:-1]
		else:
			return sections
	
	def nextPoints(self, point):
		"""
		Returns the radius of the circle used to identify the points and
		the points toward which propagate, in a tuple `(radius, [point0, point1, ...])`
		"""

		bestRadius = 0
		circularSectionsArray = self.getCircularSectionsArray(point, 0)
		allSections = [self.circularSectionsRanges(circularSectionsArray)]
		for radius in range(1, len(constants.circumferences)):
			circularSectionsArray = self.getCircularSectionsArray(point, radius, circularSectionsArray)
			allSections.append(self.circularSectionsRanges(circularSectionsArray))
			if len(allSections[radius]) > len(allSections[bestRadius]):
				bestRadius = radius
			if len(allSections[radius]) > 1 and len(allSections[-1]) == len(allSections[-2]):
				# two consecutive circular arrays with the same number>1 of sections
				break
		
		sections = allSections[bestRadius]
		points = []
		for section in sections:
			if section[2] == True:
				circumferenceIndex = int((section[0] + section[1]) / 2) # floor
				x = point[0] + constants.circumferences[circumferenceIndex][0]
				y = point[1] + constants.circumferences[circumferenceIndex][1]

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

	circularSectionsArray = None
	converter = EdgesToGcode(edges)
	for i in range(11):
		circularSectionsArray = converter.getCircularSectionsArray((14, 7), i, circularSectionsArray)
		#print(circularSectionsArray)
		sections = converter.circularSectionsRanges(circularSectionsArray)
		print(sections)

	#print(", ".join([str(c)[1:-1] for c in constants.circumferences]))

if __name__ == "__main__":
	main()