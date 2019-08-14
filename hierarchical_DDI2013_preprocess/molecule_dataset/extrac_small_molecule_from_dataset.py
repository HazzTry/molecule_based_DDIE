
molecule={}
molecule_indata={}
with open('small_molecule.txt') as f:
	for line in f:
		line=line.strip().split("	")

		smiles=line[-1]
		
		name=str(line[0])
		#print(name+"..")
		#print(name,smiles)
		molecule[name.lower()]=smiles

print(len(molecule))

dataset=open('small_molecule_dataset_train1.txt','a',encoding='utf-8')
num_of_molecule_sentence=0
#with open ('train_processed2.txt', 'r',encoding='utf-8') as r:
fp = open('test_processed2.txt', 'r',encoding='utf-8')
samples = fp.read().strip().split('\n\n')	
for sample in samples:

	sent, entities, relation = sample.strip().split('\n')
	e1, e1_t, e2, e2_t = entities.split('\t')
	if e1.lower() in molecule and e2.lower() in molecule:
		num_of_molecule_sentence=num_of_molecule_sentence+1
		#print(num_of_molecule_sentence)
		molecule_indata[e1.lower()]=1
		molecule_indata[e2.lower()]=1

		# if e1.lower() in molecule:

		# 	molecule_indata[e1.lower()]=1
		# elif  e2.lower() in molecule:

		# 	molecule_indata[e2.lower()]=1
		dataset.writelines(str(sent)+'\n'+str(entities)+'\n'+str(relation)+'\n\n')

print(num_of_molecule_sentence)
needed_molecule=open("needed_molecule3.txt",'a',encoding='utf-8')
for k,v in molecule_indata.items():
	needed_molecule.writelines(str(k)+"	"+str(molecule[str(k)])+'\n')

needed_molecule.close()
fp.close()
dataset.close()