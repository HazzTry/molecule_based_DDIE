with open("test_processed2_pos1.txt",'r') as f:
	for line in f:
		#print(line)
		#line=line.strip("[")
		line=line.strip()
		line=line.strip("]")
		line=line.strip("[")
		#print(line)
		#print(type(line))
		#print(eval(line))
		for i in eval(line):
			print(i[1])
		#print(type(eval(line)[0]))

