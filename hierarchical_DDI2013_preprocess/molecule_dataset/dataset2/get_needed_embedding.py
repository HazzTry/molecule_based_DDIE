
# #part1
# all_molecule,needed_molecule=[],[]

# with open('small_molecule.txt','r',encoding='utf-8') as f:
# 	for line in f:
# 		line=line.strip().split('	')
# 		molecule=line[0]
# 		all_molecule.append(molecule.lower())

# with open('needed_molecule3.txt','r',encoding='utf-8') as f:
# 	for line in f:
# 		line=line.strip().split('	')
# 		molecule=line[0]
# 		needed_molecule.append(molecule)
# #print(len(all_molecule),len(needed_molecule))


# index_list=[]

# for molecule in needed_molecule:
# 	index_list.append(all_molecule.index(molecule))

# #print(len(index_list))

# print(index_list)
# i=0
# j=0
# embedding_list=[]
# w=open("needed_embedding3.txt",'a',encoding='utf-8')

# with open("embedding.txt",'r',encoding='utf-8') as f:
# 	for line in f:
# 		line=line.strip()
# 		embedding_list.append(line)


# #w.close()
# print(len(embedding_list))

# for i in index_list:
# 	w.writelines(str(embedding_list[i])+'\n')

# w.close()

####################################################################
#part2
####################################################################

name,embedding=[],[]

with open("needed_molecule3.txt",'r',encoding='utf-8') as f:
	for line in f:
		molecule=line.strip().split('	')[0]
		name.append(molecule)

w=open('needed_embedding4.txt','a',encoding='utf-8')
i=0
with open('needed_embedding3.txt','r',encoding='utf-8') as f:
	for line in f:
		line=line.strip()
		w.writelines(name[i]+"	"+line+'\n')
		i=i+1
w.close()