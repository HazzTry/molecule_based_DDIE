import csv
import pandas as pd


filename='smiles2.csv'
new_file='small_molecule.txt'

d=pd.read_csv(filename,usecols=['name','smiles'])
a=d.columns
b=d.values
# c=b[0]
# print(str(c))

with open(new_file,'a',encoding='utf-8') as w:
	for data in b:
		words="	".join(data)
		words=words+"\n"
		w.writelines(words)

