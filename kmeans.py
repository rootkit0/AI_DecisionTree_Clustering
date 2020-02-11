#!/usr/bin/env python
# coding=utf-8
from math import sqrt
import random
import sys
from copy import deepcopy

def readfile(filename):
	with open(filename) as file:
		lines=[line for line in file]
		colnames=lines[0].strip().split('\t')[1:]
		rownames=[]
		data=[]
		for line in lines[1:]:
			p=line.strip().split('\t')
			rownames.append(p[0])
			data.append([float(x) for x in p[1:]])
		return rownames,colnames,data

def euclidean(v1, v2):
	distance = sqrt(sum([(v1 - v2) ** 2 for v1, v2 in zip(v1, v2)]))
	return distance	


def manhattan(v1, v2):
	distance = sum([abs(v1[i]-v2[i]) for i in xrange(len(v1))])
	return distance

def pearson(v1,v2):
	sum1 = sum(v1)
	sum2 = sum(v2)

	sum1Sq = sum([pow(v,2) for v in v1])
	sum2Sq = sum([pow(v,2) for v in v2])

	pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
	# pearson score
	num = pSum-(sum1 * sum2/len(v1))
	den = sqrt((sum1Sq-pow(sum1,2)/len(v1)) * (sum2Sq-pow(sum2,2)/len(v1)))
	if den==0: 
                return 0

	return 1.0-num/den

class bicluster:
	def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
		self.left = left
		self.right = right
		self.vec = vec
		self.id = id
		self.distance = distance

def hcluster(rows, distance=pearson):
	distances={}
	currentclustid=-1

	clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
	
        while len(clust)>1:
		lowestpair = (0, 1)
		closest = distance(clust[0].vec,clust[1].vec)

		for i in range(len(clust)):
			for j in range(i+1,len(clust)):
				if (clust[i].id,clust[j].id) not in distances:
					distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)
				if distances[(clust[i].id,clust[j].id)] < closest:
					closest = d
					lowestpair = (i,j)

		mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]	

		newcluster=bicluster(mergevec,left=clust[lowestpair[0]],right=clust[lowestpair[1]],distance=closest,id=currentclustid)
		currentclustid-=1
		del clust[lowestpair[1]]
		del clust[lowestpair[0]]
		clust.append(newcluster)
        return clust[0]

def printclust(clust, labels=None, n=0):
        for i in range(n):
                print (' '),
        if clust.id < 0:
                print ('-')
        else:
                if labels == None:
                    print (clust.id)
                else:
                    print (labels[clust.id])
        if clust.left != None:
                printclust(clust.left, labels=labels, n=n + 1)
        if clust.right != None:
                printclust(clust.right, labels=labels, n=n + 1)

def kcluster(rows, distance=pearson, k=4):
    # min and max values for each point
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
    # create centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                for i in range(len(rows[0]))] for j in range(k)]
    lastmatches = None

    for t in range(100):
        print 'Iteration %d' % t
        bestmatches = [[] for i in range(k)]
        # find nearest centroid
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)

                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)

        if t%3 == 0 and t != 0:
            for cent in range(0,k):
                if lastmatches[cent] != bestmatches[cent]:
                    print "RESETING CLUSTER: %d" %cent
                    clusters[cent] = [random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                        for i in bestmatches[cent]]

        # if the results are the same
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # else, move the centroids
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]

                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
    return bestmatches, clusters

if __name__ == "__main__":
    rownames, colnames, data= readfile("blogdata.txt")
    random.seed(6)
    kclust , clusters = kcluster(data)
    
    for i in range(0,len(kclust)):
        print "CENTROID OF THE ", i+1, "ELEMENT:"
        print kclust[i]
