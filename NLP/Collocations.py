# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 19:56:38 2018

@author: Shubham
"""

from __future__ import print_function
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
from collections import defaultdict
import operator

# consider adding the sys argument for dataset and measurement type

file = open('Collocations')

# choice for counting unigram/bigram
class Collocations:
    def __init__(self, data, measure):
        self.data = data
        self.measure = measure
        self.count_uni = 0
        self.count_bi = 0
        self.uni = {}
        self.bi = {}
        
    # function to count unigrams and bigrams
    def count(self):
        # store counts of unigrams and bigram
        uni = defaultdict(int); bi = defaultdict(int)
        
        # store counts for both 
        count_uni = 0; count_bi = 0
        data_split = self.data.split()
        punctuations = [',', '.', '?', '!', "'", '"', ':', ';', '-']

        #counting unigrams
        for ele in data_split:
            if ele not in punctuations:
                count_uni += 1
                uni[ele.lower()] += 1
        
        # counting bigrams
        if data_split[0].lower() not in punctuations and data_split[1].lower() not in punctuations:
            key = (data_split[0].lower(), data_split[1].lower())
            bi[key] += 1
        
        for i in range(2, len(data_split)):
            if data_split[i-1].lower() not in punctuations and data_split[i].lower() not in punctuations:
                count_bi += 1
                key = (data_split[i-1].lower(), data_split[i].lower())
                bi[key] += 1
         
        # remove entries which have counts less than 5
        bi_keys = list(bi.keys())
        for i in range(len(bi_keys)):
            if bi[bi_keys[i]] < 5:
                bi.pop(bi_keys[i])
                
        self.count_uni = count_uni
        self.count_bi = count_bi
        self.uni = uni
        self.bi = bi
        
    
    # calculate the Pointwise Mutual Information
    # for each bigram
    def PMI(self, bigram):
        term1, term2 = bigram
        
        # calculating n-gram probabilities
        k1, k2 = bigram
        key = (k1.lower(), k2.lower())
        P_x_y = self.bi[key]
        P_x = self.uni[term1.lower()] 
        P_y = self.uni[term2.lower()]   
        if P_x == 0 or P_y == 0:
            P_x = 1
            P_y = 1
        # calculating the PMI score
        res = (P_x_y * self.count_bi) / (P_x * P_y)
        res = np.log2(res)
        
        return res    
    # chi-square measure for a a bigram
    def chi_square(self, bigram):
        term1, term2 = bigram
        term1 = term1.lower()
        term2 = term2.lower()
        keys = list(self.bi.keys())
        A = 0; B = 0; C = 0; D = 0
        N = sum(self.bi.values())
        for ele in keys:
            if (term1 in ele) and (term2 in ele):
                A += self.bi[ele]
            if (term1 in ele) and (term2 not in ele):
                B += self.bi[ele]
            elif (term1 not in ele) and (term2 in ele):
                C += self.bi[ele]
            elif (term1 not in ele) and (term2 not in ele):
                D += self.bi[ele]
                
        E_A = (((A + B)*(A + C)) / N) + 1
        E_B = (((B + D)*(A + B)) / N) + 1
        E_C = (((A + C)*(C + D)) / N) + 1
        E_D = (((B + D)*(C + D)) / N) + 1
        
        term_A = (np.square(A - E_A)) / E_A
        term_B = (np.square(B - E_B)) / E_B
        term_C = (np.square(C - E_C)) / E_C
        term_D = (np.square(D - E_D)) / E_D
        res = term_A + term_B + term_C + term_D
        return res
    
    # PMI and chi_square scores over the entire dictionary
    # return top 20 scores
    def score(self):
        self.count()
        bi_PMI = defaultdict(float)
        bi_chi_square = defaultdict(float)
        count_i = 0 
        
        # iteritems does not work for defaultdict
        if self.measure == "PMI":
            for ele in list(self.bi.keys()):
                count_i += 1
                bi_PMI[ele] = self.PMI(ele)
#                print(count_i)
            bi_PMI_sort = sorted(bi_PMI.items(), key=lambda kv: kv[1], reverse=True)
            ans = bi_PMI_sort[:20]
        else:
            for ele in list(self.bi.keys()):
                count_i += 1
                bi_chi_square[ele] = self.chi_square(ele)
#                print(count_i)
            bi_chi_square_sort = sorted(bi_chi_square.items(), key=lambda kv: kv[1], reverse=True)
            ans = bi_chi_square_sort[:20]
        return ans
        
if __name__ == "__main__":
    data = file.read()
    choice = str(sys.argv[1])
    col = Collocations(data, choice)
    ans = col.score()
    with open("Collocations.answers2.txt", "a") as f:
        f.write('\n' + "%s measure"  %choice) 
	f.write('\n')
        for ele in ans:
            f.write('(' + ele[0][0] + "," + ele[0][1] + ')' + ":" + str(ele[1]) + '\n')
    print (ans)