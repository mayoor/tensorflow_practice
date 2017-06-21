import os
from os import walk 
import json

import xml.etree.ElementTree as ET

def getVOCDataSet(root_folder):
	data = {'images':[]}
	image_folder = os.path.join('JPEGImages',root_folder)
	for (dirpath, dirnames, filenames) in walk(image_folder):
		for image in filenames:
			image_file = os.path.join(dirpath,image)
			bnddata = getBndBox(image_file)
			if bnddata:
				im = {'location':image_file, 'bindbox':bnddata} 
				data['images'].append(im)
	return data

def getBndBox(file):
	print (file)
	file_name = file[file.index('\\')+1:file.index('.jpg')]
	#print (file_name)
	file_path = 'Annotations\\'+file_name+'.xml'
	#print (file_path)

	tree = ET.parse(file_path)
	root = tree.getroot()
	objs = root.findall('object')
	num_objects = len(objs)
	if num_objects == 1:
		bnd = objs[0].find('bndbox')
		xmin = bnd.find('xmin').text
		ymin = bnd.find('ymin').text
		xmax = bnd.find('xmax').text
		ymax = bnd.find('ymax').text
		return {'xmin':xmin,'ymin':ymin,'xmax':xmax, 'ymax':ymax}
	else:
		return None
	
def main():
	data = getVOCDataSet('.')
	with open('voc_annot.json','w') as voc_file:
		voc_file.write(json.dumps(data))

	
if __name__=='__main__':
	main()