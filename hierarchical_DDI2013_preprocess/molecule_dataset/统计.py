false1=0
int1=0
advice1=0
effect1=0
mech1=0

#with open ('train_processed2.txt', 'r',encoding='utf-8') as r:
fp = open('small_molecule_dataset_test2.txt', 'r',encoding='utf-8')
samples = fp.read().strip().split('\n\n')	
for sample in samples:

	sent, entities, relation = sample.strip().split('\n')
	e1, e1_t, e2, e2_t = entities.split('\t')
	if relation.lower()=='false':
		false1=false1+1
	elif relation.lower()=='int':
		int1=int1+1
	elif relation.lower()=='effect':
		effect1=effect1+1
	elif relation.lower()=='mechanism':
		mech1=mech1+1
	elif relation.lower()=='advise':
		advice1=advice1+1

print(false1,advice1,int1,effect1,mech1)
fp.close()
