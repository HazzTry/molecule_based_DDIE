www=open("train_pos.txt",'a')
with open("train_processed2_pos.txt",'r') as f:
	for line in f:
		pos_string=""
		#print(line)
		#line=line.strip("[")
		line=line.strip()
		line=line.strip("]")
		line=line.strip("[")
		#print(line)
		#print(type(line))
		#print(eval(line))

		for i in eval(line):
			pos_string=pos_string+" "+str(i[1])
		www.writelines(pos_string.strip()+'\n')

			#print(i[1])
		#print(type(eval(line)[0]))

