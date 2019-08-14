w=open('all_molecule_smiles.txt','a',encoding='utf-8')
maxlength=0
with open('small_molecule.txt','r',encoding='utf-8') as f:
	for line in f:
		smile=line.strip().split('	')[1]	
		smile=list(smile)
		if len(smile) > maxlength:
			maxlength=len(smile)
		smile=" ".join(smile)
		w.writelines(smile+"\n")
print(maxlength)
w.close()