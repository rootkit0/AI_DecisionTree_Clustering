#!/usr/bin/env python
# coding=utf-8
import argparse
import collections
import itertools
import math
import sys
import time
import copy
import random

#---------t3---------
def read(file_name):
	text =[]
	f= open(file_name)
	for line in f.readlines():
		splitted = line.split('\t')
		splitted[3] = int(splitted[3])
		splitted[4] = splitted[4][:-1]
		text.append(splitted)
	f.close()

	return text

def readfile(file):
        f = open(file, "r")
        data = []
        for line in f:
                data.append([x.strip() for x in line.split(',')])

        return data

#---------t4---------
def unique_counts(rows):
    counts = {}
    for row in rows:
        counts[row[-1]] = counts.get(row[-1], 0) + 1

    return counts

#---------t5---------
def gini_impurity(rows):
	total = float(len(rows))
	results = unique_counts(rows)
	imp = 0
	for k,v in results.items():
		imp += (v / total)**2
	imp = 1 - imp

	return imp

#---------t6---------
def entropy(rows):
	from math import log 
	log2 = lambda x:log(x)/log(2)
	results = unique_counts(rows)
	imp = 0.0
	total= float(len(rows))
	for k,v in results.items():
		imp += (v / total) * log2(v / total)
	imp = -imp
        
	return imp

#---------t7--------- 
def divideset(part, column, value):
	def split_num(prot):
                 return prot[column] >= value
	def split_str(prot): 
                return prot[column] == value

	split_fn = split_num if isinstance(value, (int, float)) else split_str
  
	set1, set2 = [], []
	for prot in part:
		s = set1 if split_fn(prot) else set2
		s.append(prot)

	return set1, set2

#---------t8---------
class decisionnode:
	def __init__(self, col=-1, value=None, results=None, tb = None, fb = None):
		self.col = col
		self.value = value
		self.results = results
		self.tb = tb
		self.fb = fb

	def __eq__(self, other):
		return self.col == other.col and self.value == other.value and self.results == other.results \
			   and self.tb == other.tb and self.fb == other.fb

	def setCol(self, col):
		self.col = col

	def setValue(self, value):
		self.value = value

	def setResults(self, results):
		self.results = results

	def setTb(self, tb):
		self.tb = tb

	def setFb(self, fb):
		self.fb = fb

#Practica 3
#---------t9 - Construcción del árbol de forma recursiva---------
def buildtree(rows, scoref=entropy, beta=0.0): 
        if len(rows) == 0: return decisionnode()
        current_score = scoref(rows)

        best_gain = 0.0
        best_criteria = None
        best_first_set= None
        best_second_set = None

        # Number of the columns that contains none, premium and basic
        count_column = len(rows[0]) - 1  

        for col in range(0, count_column):
                # List of different values
                columns={}
                for row in rows:
                        columns[row[col]]=1
                for value in columns.keys():
                        # Divide by sets
                        (set1, set2) = divideset(rows, col, value)
                        # Calculate the impurity 
                        p= float(len(set1))/len(rows)
                        gain_information = current_score - p * scoref(set1) - (1 - p) * scoref(set2)

                        if gain_information > best_gain and len(set1) > 0 and len(set2) > 0:
                                best_gain = gain_information
                                best_criteria = (col, value)
                                best_set_first = set1
                                best_set_second = set2

        # Call function recursively
        if beta < best_gain and best_gain > 0:           
                first_branch_true = buildtree(best_set_first, scoref, beta)
                second_branch_false = buildtree(best_set_second, scoref, beta)     
                return decisionnode(col = best_criteria[0], value = best_criteria[1], tb = first_branch_true, fb = second_branch_false)
        else:
                return decisionnode(results=unique_counts(rows))

#---------t10 - Construcción del árbol de forma iterativa---------
def buildtree_ite(rows, scoref=entropy, beta=0.0):
        if len(rows) == 0: return decisionnode()
        result = decisionnode()

	sub_list = [[rows,result, None, '']]

        while len(sub_list) != 0:
		rows, node, father, tb_and_fb = sub_list.pop(0)                
                current_score = scoref(rows)
                if current_score > beta and len(rows) != 0:
                        best_gain = 0.0
                        best_criteria = None
                        best_first_set= None
                        best_second_set = None
                        value = None
                        # Number of the columns that contains none, premium and basic
                        count_column = len(rows[0]) - 1

                        for col in range(0, count_column):           
                                for val in columns(rows, col):
                                        # Divide by sets
                                        set1, set2 = divideset(rows, col, val)
                                        # Calculate the impurity
                                        gain_information = current_score - (float(len(set1))/float(len(rows)) * scoref(set1)) - (float(len(set2))/float(len(rows)) * scoref(set2))
                                        if gain_information >= best_gain:
                                                best_gain = gain_information
                                                best_criteria = col
                                                best_set_first = set1
                                                best_set_second = set2
                                                value = val
			node.setValue(value)
                        node.setCol(best_criteria)
			sub_list.append([best_set_first, decisionnode(), node, 'tb'])
			sub_list.append([best_set_second, decisionnode(), node, 'fb'])
                else:
                        node.setResults(unique_counts(rows))
		if father != None:
                        if tb_and_fb == 'tb':
                                father.setTb(node)   
			else:
                                father.setFb(node)     
        return result

# aux function
def columns(rows, col):
    column = set()
    for row in rows:
        if row[col] not in column:
            column.add(row[col])
    return column

#---------Print tree---------
def printtree(tree,indent=''):
        # Base case
	if tree.results!=None:
		print "Tree results: " + str(tree.results) + "\n"
	else:
		# Print the criteria
		print "Criteria: " + str(tree.col)+':'+str(tree.value)+'? \n'
		# Print the branches
                print "Branches:"
		print indent+'T->',
		printtree(tree.tb,indent+'  ')  
		print indent+'F->',
		printtree(tree.fb,indent+'  ')

#---------t12 - Función de clasificación---------
def classify(obj, tree):
        # Base case
        if tree.results != None:
                return tree.results
        else:
                v = obj[tree.col]
                # Check the branches
                if isinstance(v, (int, float)):
                        if v < tree.value:
                                branch = tree.fb 
                        else:
                                branch = tree.tb            
                else:
                        if v != tree.value:
                                branch = tree.fb
                        else:
                                branch = tree.tb

                return classify(obj, branch)

#---------t13 - Evaluación del árbol---------
def test_performance(test_set, training_set):
        root = buildtree(test_set)
        correct_obj = 0
        incorrect_obj = 0

        for row in training_set:
                l = classify(row, root)
                key, value = max(l.iteritems(), key=l.get)

                if key != row[-1]:
                    incorrect_obj += 1
                else:
                    correct_obj += 1

        total_objects = correct_obj + incorrect_obj
        percent_correct = (float(correct_obj)/ total_objects)*100
        print "Total classified objects: ",total_objects
        print "Number of correct objects: ",correct_obj
        print "Percent of correct objects: ", percent_correct
        return percent_correct

#---------t14 - Evaluación del árbol---------
def quality_of_classify(rows, percent, percentatge = 0):
        # Randomizing the rows
        random.shuffle(rows)
        if percentatge == 0:
                percentatge = percent
        if percentatge >= 100:
                return

        test = int((len(rows)/100)*percentatge)

        if test == 0:
               test = int(float(len(rows)) / 100.0*percentatge)

        test_set = []
        training_set = []
    
        for fila in rows:
                if len(test_set) != test:
                    test_set.append(fila)
                else:
                    training_set.append(fila)
    
        print "\nTesting with the "+ str(percentatge) + "% of the space\n" 
        test_performance(test_set,training_set)
    
        quality_of_classify(rows, percent, percentatge + percent)

#---------t15 - Missing data---------
#La primera forma de solucionar esta falta de datos es sustituyendo o remplazando los datos restantes por 
#los valores que más se repitan estadísticamente dentro del dataset, otra posible forma seria sustituyendo 
#los valores restantes, (por ejemplo números por valores al azar) aunque esta situación provocaría que los datos 
#obtenidos al final tuviesen un sesgo muy diferente del que les correspondería en la realidad.

#---------t16 - Poda del árbol---------
def prune(tree, threshold):
        if tree.tb.results == None: prune(tree.tb, threshold)
        if tree.fb.results == None: prune(tree.fb, threshold)

        if tree.tb.results != None and tree.fb.results != None:
                tb,fb=[],[]
                # List of lists
                for v,c in tree.tb.results.items(): tb+=[[v]]*c
                for v,c in tree.fb.results.items(): fb+=[[v]]*c
                # Entropy
                calcul = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)
        	if calcul < threshold:
        	   tree.tb.results = None
        	   tree.fb.results = None
                   tree.results = unique_counts(tb + fb)

#---------Main---------
if __name__ == '__main__':
        #Sobre el dataset decision_tree_example.txt
        data = read(sys.argv[1])
	print(data)
	counts = unique_counts(data)
	print(counts)
	gini = gini_impurity(data)
	print(gini)
	entropia = entropy(data)
	print(entropia)
        tree = buildtree(data)
        printtree(tree)
        itTree = buildtree_ite(data)
        print(classify(['slashdot','France','yes',19], tree))
        prune(tree, 1.0)
        printtree(tree)
        quality_of_classify(data, 10)
        
        #Sobre el dataset grande
        print("\n---------Dataset 'cars.data' from 'archive.ics.uci.edu'---------")
        data2 = readfile("car.data")
        quality_of_classify(data2, 10)
