#!/usr/bin/python3
#pylint: disable=no-member

import numpy as np
from scipy import ndimage
import imageio
from PIL import Image, ImageFilter
import constants


def sobel(image, threshold):
	Gx = ndimage.sobel(image, axis=0)
	Gy = ndimage.sobel(image, axis=1)
	G = np.hypot(Gx, Gy)

	shape = np.shape(G)
	result = np.zeros(shape[0:2], dtype=bool)

	result[(G[:, :, 0] + G[:, :, 1] + G[:, :, 2] + G[:, :, 3]) >= threshold] = True
	return result


class CircularRange:
	def __init__(self, begin, end, value):
		self.begin, self.end, self.value = begin, end, value

	def __repr__(self):
		return f"[{self.begin},{self.end})->{self.value}"

	def halfway(self):
		return int((self.begin + self.end) / 2)

class Graph:
	class Node:
		def __init__(self, point, index):
			self.x, self.y = point
			self.index = index
			self.connections = {}

		def __repr__(self):
			return f"({self.y},{-self.x})"

		def _addConnection(self, to):
			self.connections[to] = False # i.e. not already used in gcode generation

		def toDotFormat(self):
			return (f"{self.index} [pos=\"{self.y},{-self.x}!\", label=\"{self.index}\\n{self.x},{self.y}\"]\n" +
				"".join(f"{self.index}--{conn}\n" for conn in self.connections if self.index < conn))


	def __init__(self):
		self.nodes = []

	def __getitem__(self, index):
		return self.nodes[index]

	def __repr__(self):
		return repr(self.nodes)


	def addNode(self, point):
		index = len(self.nodes)
		self.nodes.append(Graph.Node(point, index))
		return index

	def addConnection(self, a, b):
		self.nodes[a]._addConnection(b)
		self.nodes[b]._addConnection(a)

	def distance(self, a, b):
		return np.hypot(self[a].x-self[b].x, self[a].y-self[b].y)

	def areConnectedWithin(self, a, b, maxDistance):
		if maxDistance < 0:
			return False
		elif a == b:
			return True
		else:
			for conn in self[a].connections:
				if self.areConnectedWithin(conn, b, maxDistance - self.distance(conn, b)):
					return True
			return False

	def saveAsDotFile(self, filename):
		with open(filename, "w") as f:
			f.write("graph G {\nnode [shape=plaintext];\n")
			for node in self.nodes:
				f.write(node.toDotFormat())
			f.write("}\n")

	def saveAsGcodeFile(self, filename):
		with open(filename, "w") as f:

			def dfsGcode(i, insidePath):
				for connTo, alreadyUsed in self[i].connections.items():
					if not alreadyUsed:
						f.write(f"G{1 if insidePath else 0} X{self[i].y} Y{-self[i].x}\n")
						self[i].connections[connTo] = True
						self[connTo].connections[i] = True

						dfsGcode(connTo, True)
						insidePath = False

				if insidePath: # still inside path, i.e. no valid connections found
					f.write(f"G1 X{self[i].y} Y{-self[i].x}\n")

			for i in range(len(self.nodes)):
				if len(self[i].connections) == 0:
					f.write(f"G0 X{self[i].y} Y{-self[i].x}\nG1 X{self[i].y} Y{-self[i].x}\n")
				elif len(self[i].connections) == 1:
					dfsGcode(i, False)


class EdgesToGcode:
	def __init__(self, edges):
		self.edges = edges
		self.ownerNode = np.full(np.shape(edges), -1, dtype=int)
		self.xSize, self.ySize = np.shape(edges)
		self.graph = Graph()

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

	def getNextPoints(self, point):
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
			if len(allRanges[bestRadius]) >= 4 and len(allRanges[-2]) >= len(allRanges[-1]):
				# two consecutive circular arrays with the same or decreasing number>=4 of ranges
				break
			elif len(allRanges[radius]) == 2 and radius > 1:
				edge = 0 if allRanges[radius][0].value == True else 1
				if allRanges[radius][edge].end-allRanges[radius][edge].begin < len(constants.circumferences[radius]) / 4:
					# only two ranges but the edge range is small (1/4 of the circumference)
					if bestRadius == 1:
						bestRadius = 2
					break
			elif len(allRanges[radius]) == 1 and allRanges[radius][0].value == False:
				# this is a point-shaped edge not sorrounded by any edges
				break

		if bestRadius == 0:
			return 0, []

		circularRanges = allRanges[bestRadius]
		points = []
		for circularRange in circularRanges:
			if circularRange.value == True:
				circumferenceIndex = circularRange.halfway()
				x = point[0] + constants.circumferences[bestRadius][circumferenceIndex][0]
				y = point[1] + constants.circumferences[bestRadius][circumferenceIndex][1]

				if x in range(self.xSize) and y in range(self.ySize) and self.ownerNode[x, y] == -1:
					points.append((x,y))

		return bestRadius, points

	def propagate(self, point, currentNodeIndex):
		radius, nextPoints = self.getNextPoints(point)

		# depth first search to set the owner of all reachable connected pixels
		# without an owner and find connected nodes
		allConnectedNodes = set()
		def setSeenDFS(x, y):
			if (x in range(self.xSize) and y in range(self.ySize)
					and np.hypot(x-point[0], y-point[1]) <= radius + 0.5
					and self.edges[x, y] == True and self.ownerNode[x, y] != currentNodeIndex):
				if self.ownerNode[x, y] != -1:
					allConnectedNodes.add(self.ownerNode[x, y])
				self.ownerNode[x, y] = currentNodeIndex # index of just added node
				setSeenDFS(x+1, y)
				setSeenDFS(x-1, y)
				setSeenDFS(x, y+1)
				setSeenDFS(x, y-1)

		self.ownerNode[point] = -1 # reset to allow DFS to start
		setSeenDFS(*point)
		for nodeIndex in allConnectedNodes:
			if not self.graph.areConnectedWithin(currentNodeIndex, nodeIndex, 11):
				self.graph.addConnection(currentNodeIndex, nodeIndex)

		validNextPoints = []
		for nextPoint in nextPoints:
			if self.ownerNode[nextPoint] == currentNodeIndex:
				# only if this point belongs to the current node after the DFS,
				# which means it is reachable and connected
				validNextPoints.append(nextPoint)

		for nextPoint in validNextPoints:
			nodeIndex = self.graph.addNode(nextPoint)
			self.graph.addConnection(currentNodeIndex, nodeIndex)
			self.propagate(nextPoint, nodeIndex)
			self.ownerNode[point] = currentNodeIndex

	def addNodeAndPropagate(self, point):
		nodeIndex = self.graph.addNode(point)
		self.propagate(point, nodeIndex)

	def buildGraph(self):
		for point in np.ndindex(np.shape(self.edges)):
			if self.edges[point] == True and self.ownerNode[point] == -1:
				radius, nextPoints = self.getNextPoints(point)
				if radius == 0:
					self.addNodeAndPropagate(point)
				else:
					for nextPoint in nextPoints:
						if self.ownerNode[nextPoint] == -1:
							self.addNodeAndPropagate(nextPoint)

		return self.graph


def pokeballEdges():
	image = imageio.imread("pokeball_small.png")

	edges = sobel(image, 128.0)
	imageio.imwrite("pokeballsobel.png", edges.astype(float))

	return edges

def testEdges():
	image = imageio.imread("test_edges.png")
	edges = np.zeros(np.shape(image)[0:2], dtype=bool)

	for xy in np.ndindex(np.shape(image)[0:2]):
		edges[xy] = (image[xy][0] > 128 and image[xy][1] > 128 and image[xy][2] > 128)

	return edges

def main():
	edges = testEdges()

	if np.shape(edges)[0] < 50 and np.shape(edges)[1] < 50:
		print("-----------------")
		for x, y in np.ndindex(np.shape(edges)):
			if y == 0 and x != 0: print()
			print("cɔ" if edges[x,y] else "  ", end="")
		print("\n-----------------")

	circularArray = None
	converter = EdgesToGcode(edges)
	for i in range(11):
		circularArray = converter.getCircularArray((26,28), i, circularArray)
		#print(circularArray)
		sections = converter.toCircularRanges(circularArray)
		print(sections)
	print(converter.getNextPoints((26,28)))

	#converter.graph = []
	#converter.propagate((26, 28))

	converter.buildGraph()
	print(converter.graph)
	converter.graph.saveAsDotFile("graph.dot")
	converter.graph.saveAsGcodeFile("graph.nc")

if __name__ == "__main__":
	main()