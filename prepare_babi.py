import re
import sys

def prepare_dataset(input_file):
	data = []
	for line in input_file:
		g = re.search(r"^[1]{1}\s",line)
		if g:
			story = []
		line = re.sub(r"^[\d]+\s",'',line).strip().lower()
		line = re.sub(r"[\.\?,]",'',line)
		vals = line.split('\t')
		if len(vals) > 1:
			question = vals[0]
			answer = vals[1]
			data.append(([x for x in story],question, answer))
		else:
			story.append(line)
	return data
		

if __name__ == '__main__':
	file_name = 'qa2_two-supporting-facts_train.txt'
	with open(file_name,'r') as inp:
		data = prepare_dataset(inp)
	print (data)
	
	