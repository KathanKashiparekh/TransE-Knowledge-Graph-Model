import numpy as np 
import pandas as pd  


# 1. Read in entit2id and relation2id files.
# 2. Read in the test.txt file as well
# 3. For each triple in text, convert it to format of triple2id.


entity=pd.read_table('entity2id.txt',header=None)
relation=pd.read_table('relation2id.txt',header=None)
test=pd.read_table('test.txt',header=None)
#print list(entity[0]).index('/m/06cx9')
# print test.iloc[0][1]
# print test.head()

for i in range(test.shape[0]):
	e1=list(entity[0]).index(test.iloc[i][0])
	e2=list(entity[0]).index(test.iloc[i][1])
	r=list(relation[0]).index(test.iloc[i][2])
	print e1,e2,r
