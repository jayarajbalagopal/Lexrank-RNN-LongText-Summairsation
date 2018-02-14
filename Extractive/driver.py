import os
import sys

try:
	filename = sys.argv[1]	
	
	if(os.path.exists("../Inputs/"+filename)):
		os.system("javac Controller.java")
		file = "java Controller ../Inputs/"
		file = file + filename
		os.system(file)

		with open("../Outputs/out.txt","rb") as fp:
			print(fp.read())
	else:
		print("no such file")

except:
	print("no file")


