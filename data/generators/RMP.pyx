# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:03:30 2021

Implements a random message passing algorithm,
which does a random guess, then flips a bits if it satisfies more edges

@author: Jonathan Wurtz
"""

import networkx as nx
from numpy import *
import time
import itertools
import scipy
import tqdm
import pandas as pd

# Define the geometry of the graph
Ns = [10**2, 5*10**2, 10**3]#, 5*10**3, 10**4, 5*10**4, 10**5]
connectivity = 3

# Number of samples to try
nsamples = 1000

compiled_list = []
header = []

for index, N in enumerate(Ns):
	cut_fraction = []
	time_to_solution = []
	for sampind in range(nsamples):              # Try a bunch of graphs as to get statistics
		G = nx.random_regular_graph(connectivity,N)
		
		t1 = time.time()
		edges = array(list(G.edges))                        # Some convenient vectorized representations of the graph
		adj = array([[r] + list(q.keys()) for r,q in dict(G.adjacency()).items()])
		sorted_adjacency = adj[argsort(adj[:,0]),1::]
		
		SOLUTION = random.choice([-1,1],size=N)             # First, choose a random solution
		go_again = random.permutation(arange(N))
		
		#print(sorted_adjacency)
		while len(go_again)>0:                              # Repeat until there is nothing left to change
			go_again2 = []
			for i in go_again:                              # For every vertex... (excluding ones that didn't change last time, to be efficient)
				neighbors = list(sorted_adjacency[i])       # Look at the neighboring vertices
				if sum(SOLUTION[neighbors])*SOLUTION[i]>0:  # If it increases the MaxCut value...
					SOLUTION[i] *= -1                       #  flip the vertex to the other bipartition
					go_again2 += neighbors + [i]            # Cleanup: Re-check that vertex and its neighbor the next iteration
			
			go_again = random.permutation(unique(go_again2))
			
		t2 = time.time()
		
		cut_fraction.append(sum(1 - SOLUTION[edges[:,0]]*SOLUTION[edges[:,1]])/2/edges.shape[0]) # A megaline to compute the MaxCut value
		time_to_solution.append(t2-t1)
	compiled_list.append(cut_fraction)
	compiled_list.append(time_to_solution)
	header.append(f"N={N} cut fraction")
	header.append(f"N={N} time to solution (s)")


compiled_array = array(compiled_list).T
frame = pd.DataFrame(compiled_array)

frame.to_csv("../message_passing_data2.csv",index=None,header=header)

