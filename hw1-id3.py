# -*- coding: utf-8 -*-
""" 
Program: hw1.py
Programmed By: Adam Morse
Description: An implementation of the id3 decision tree. Ability to classify 
             sample data using Classify(tree, sample) function. 
Trace Folder: Morse1
"""

import math

""" Given training data """
training_data = [
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
]

""" Chosen training data """
training_data_1 = [
    ({'outlook':'Sunny', 'temp':'Hot', 'humidity':'High', 'wind': 'Weak'}, False),
    ({'outlook':'Sunny', 'temp':'Hot', 'humidity':'High', 'wind': 'Strong'}, False),
    ({'outlook':'Overcast', 'temp':'Hot', 'humidity':'High', 'wind': 'Weak'}, True),
    ({'outlook':'Rain', 'temp':'Mild', 'humidity':'High', 'wind': 'Weak'}, True),
    ({'outlook':'Rain', 'temp':'Cool', 'humidity':'Normal', 'wind': 'Weak'}, True),
    ({'outlook':'Rain', 'temp':'Cool', 'humidity':'Normal', 'wind': 'Strong'}, False),
    ({'outlook':'Overcast', 'temp':'Cool', 'humidity':'Normal', 'wind': 'Strong'}, True),
    ({'outlook':'Sunny', 'temp':'Mild', 'humidity':'High', 'wind': 'Weak'}, False),
    ({'outlook':'Sunny', 'temp':'Cool', 'humidity':'Normal', 'wind': 'Weak'}, True),
    ({'outlook':'Rain', 'temp':'Mild', 'humidity':'Normal', 'wind': 'Weak'}, True),
    ({'outlook':'Sunny', 'temp':'Mild', 'humidity':'Normal', 'wind': 'Strong'}, True),
    ({'outlook':'Overcast', 'temp':'Mild', 'humidity':'High', 'wind': 'Strong'}, True),
    ({'outlook':'Overcast', 'temp':'Hot', 'humidity':'Normal', 'wind': 'Weak'}, True),
    ({'outlook':'Rain', 'temp':'Mild', 'humidity':'High', 'wind': 'Strong'}, False)
    ]

 

""" Grab attributes for dataset """
def getAttributes(D):
    attr = []
    for i in D[0][0]:
        attr.append(i)
    return attr
#print(getAttributes(training_data))

""" Grab true/false values for dataset """
def getAttributeValues(D):       
    attrValues = []
    for i in D:
        attrValues.append(i[1])        
    return attrValues           
#print(getAttributeValues(training_data))  

""" Grab total count of false and true count for dataset """
def getAttributeValuesCount(av):        
    trueCount = 0
    falseCount = 0 
    for i in range(len(av)):
        if av[i] == True:       
            trueCount += 1          # Increment true count for all true values found
        else:
            falseCount += 1         # Increment false count for all false values found
    count = (trueCount, falseCount)     # return list values format; i.e (9,5)
    return count
#print(getAttributeValuesCount(getAttributeValues(training_data)))

""" Grab the true/false count for the children attributes """
def getAvCount(d, a):
    attrList = []  
    for k in a:
        attrDict = dict()
        for i in d:
            if i[0][k] in attrDict:
                if i[1] == True:            # Add count if true is found
                   attrDict[i[0][k]][0] += 1
                else:                        # Add count if false is found
                   attrDict[i[0][k]][1] += 1
            else:
                if i[1] == True:            # Populate dictionary for true values
                    attrDict[i[0][k]] = [1,0]
                else:                       # Populate dictionary for false values
                    attrDict[i[0][k]] = [0,1]
        attrList.append(attrDict)
    newAttrList = []                        # Clean up output (make it easier to work with)
    for i in attrList:
        attrProb = []
        for key in i:
            attrProb.append(i[key])         # Only grab list items of truthy values (remove names)
        newAttrList.append(attrProb)
    return (newAttrList, attrList)
#print(getAvCount(training_data, getAttributes(training_data)))

""" Grab the entropy """
def entropy(ent):
    total = 0
    for p in ent:
        p = p / sum(ent)
        if p != 0:
            total += p * math.log(p,2)
        else:
            total += 0
    total *= -1
    return total

""" Grab the information gain """
def gain(d, a):
    total = 0
    for v in a:
        total += sum(v) / sum(d) * entropy(v)   # Total entropy of dataset
    gain = entropy(d) - total
                                                # Find entropy for each attribute
    return gain

""" id3 decision tree implementation """
def id3(D, A):
    prob_dict = []
    p_list = []
    (p_list, prob_dict) = getAvCount(D, A)      # Assign p_list to newAttrList and prob_dict to attrList
    av = getAttributeValues(D)                  # Assign av to attribute values that contain true/false list
    av_count = getAttributeValuesCount(av)            # assign av_count to total count list i.e (9,5)
    hire = (av_count)
    
    prob = [0,0]        # Holds counter to determine truthy values
    for i in D:
        if i[1] == False:       # Assign y values of prob[x, y] to false
            prob[1] += 1
        else:
            prob[0] += 1        # Assign x values of prob[x, y] to true
            
    if prob[0] == 0:            # If no x values in prob[x, y], return False
        tree = False
        return tree
    if prob[1] == 0:            # If no y values in prob[x, y], return True
        tree = True
        return tree

    maxGain = ('none', 0, 0)
    for j in range(len(p_list)):
        info_gain = gain(hire, p_list[j])       # Call info gain 
        if info_gain > maxGain[1]:
            maxGain = (A[j], info_gain, j)
    maxIndex = maxGain[0]
    A.remove(maxIndex)                  # Remove highest gain attribute from set
    
    v_list = []
    for i in prob_dict[maxGain[2]]:     # Attribute to split with child values
        v_list.append(i)
    s_data = []                         # Holder for splitting data set
    for i in range(len(v_list)):
        s_i = []
        for j in D:
            if(j[0][maxIndex] == v_list[i]):   
                s_i.append(j)
        s_data.append(s_i)
    split_tree = dict()                 # Dictionary that contains split data
    for i in range(len(v_list)):
        split_tree[v_list[i]] = id3(s_data[i], A)   # Split data and recursive call id3(D or (s_data[i]),A)
    if prob[0] > prob[1]:                           # D now equals modified data set that no 
                                                    # longer contains max info gain attribute
        split_tree[None] = True
    else:                                           # Handle unexpected or missing values (None)                 
        split_tree[None] = False
    
    
    tree = (maxIndex, split_tree);                  # tree contains the max info gain attributes
                                                    # that split, and the remaining data in the dictionary
   
    return tree

""" Classify function for sample data """
def classify(tree, sample):
    c_sample = None
    t_key = tree[0]
    sample_key = None
    for i in tree[1]:                           
        if t_key in sample:
            sample_key = sample[t_key]      # Grab attribute that matches training_data to split
        else:
            t_key = None

    if sample_key not in tree[1]:           # None attribute
        sample_key = None
        
    if tree[1][sample_key] == True or tree[1][sample_key] == False:     
        c_sample = tree[1][sample_key]      # Obtain truthy value from tree
    else:
        c_sample = classify(tree[1][sample_key], sample)    # Travese tree to determine children values
    return c_sample

""" Program main """                   
def main():
    if(__name__ == "__main__"):
        
        """ Function calls for creating of id3 for given data set """
        
        a = getAttributes(training_data)              # Grab attributes for given data set
        ID3_given_set = id3(training_data, a)         # assgin ID3_given_set to given data set(hire)
        print(ID3_given_set)                          # print id3 tree
        

        #Classify samples for given training_data
        
        classify_sample_given = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"}
        #classify_sample_given = {"level" : "Senior"}
        print(classify(ID3_given_set, classify_sample_given))   # print called classified sample
        
        
        """ Function calls for creating of id3 for chosen data set (comment out given set(lines 212 - 221)
            then uncomment lines 226 - 236 to run chosen sample set"""
        """
        a = getAttributes(training_data_1)            # Grab attributes for chosen data set
        ID3_chosen_set = id3(training_data_1, a)      # assgin ID3_chosen_set to chosen data set(play tennis)
        print(ID3_chosen_set)                         # print id3 tree
        
        #Classify samples for chosen training_data
        
        classify_sample_chosen = {'outlook':'Rain', 'temp':'Mild', 'humidity':'High', 'wind': 'Weak'}
        #classify_sample_chosen = {'outlook':'Tornado'}
        print(classify(ID3_chosen_set, classify_sample_chosen))   # print called classified sample
        """
main()

#---------------------------------End of Program-------------------------------