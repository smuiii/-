import time
import face_recognition as fr
from PIL import Image, ImageDraw
import cv2
import os
from absl import app, flags, logging
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import pandas as pd
from datetime import date
import numpy as np
import math



face_detected_path = r"/Users/kikkk/Desktop/FaceRecognition/Detected Faces"
face_lib_path = Path("/Users/kikkk/Desktop/FaceRecognition/Face Lib")
image_detected = r"/Users/kikkk/Desktop/FaceRecognition/Detected Faces/2.jpeg"
check_in_records_path = Path("/Users/kikkk/Desktop/FaceRecognition/check_in_records.csv")
check_in_records_txt_path = Path("/Users/kikkk/Desktop/FaceRecognition/check_in_records.txt")



# load and encoding
# [Face Lib]
# returns -> [face_lib_encoding_list, face_lib_names_list]
def face_lib_encodings(face_lib_path):

	#pprint(list(face_lib_path.glob("*.jpg")))

	# filter all the face path in the lib
	# image -> .jpg
	face_lib = list(face_lib_path.glob("*.jpg"))
	#print(type(face_lib))

	# face_lib_encoding_list
	face_lib_encoding_list = []

	# face_lib_names_list
	face_lib_names_list = []



	for img_path in face_lib:

		# load
		img = fr.load_image_file(img_path)

		# encoding
		img_encoding = fr.face_encodings(img)[0]

		# append to the face_lib encoding  list
		face_lib_encoding_list.append(img_encoding)

		# image name 
		img_name = img_path.stem
		face_lib_names_list.append(img_name)


	print("【-Person in Face Lib:】\n",face_lib_names_list, end = "\n\n")


	return face_lib_encoding_list, face_lib_names_list



### draw face location
#input image type -> npArray
# return -> two kind of images
# [image_located, face_cropped]
def face_location_draw(img_npArray, face_locations, bboxThickness = 2, color = (0, 255, 0)):
	image_located = img_npArray.copy()
	faceNum = len(face_locations)
	for i in range(0, faceNum):
		top =  face_locations[i][0]
		right =  face_locations[i][1]
		bottom = face_locations[i][2]
		left = face_locations[i][3]

		start = (left, top)
		end = (right, bottom)

		# face crop
		face_cropped = image_located[top: bottom, left: right] 

		# bbox drawing
		cv2.rectangle(image_located, start, end, color, bboxThickness)


	return image_located, face_cropped



### show face location and crop
def show_face_loc_crop(image):
	face_locations = fr.face_locations(image)
	if face_locations:
		print("【Face has been located.】")
		print("【the face coordinate is】:", face_locations, end="\n\n")
		print("【You can check in the image popped up.】")

		# face located & face cropped
		image_located, face_cropped = face_location_draw(image, face_locations)

		#change npArray image to PIL type.
		image_located = Image.fromarray(image_located)
		face_cropped = Image.fromarray(face_cropped)

		# show 
		image_located.show()
		face_cropped.show()






### check whether one is in face lib or not
def check_in_face_lib(image_encoding, face_lib_encoding_list, face_lib_names_list, tolerance = 0.5):

	# results
	results = fr.compare_faces(face_lib_encoding_list, image_encoding, tolerance)
	#print(results, end = "\n\n")

	# distance -> similarity
	distance = fr.face_distance(image_encoding, face_lib_encoding_list)
	print("【Similarity to the one in face lib】:\n", distance, end = "\n\n")

	# who's this one?
	for i in range(len(results)):
		if results[i] == True:
			### check who
			face_person = face_lib_names_list[i]

			### check in
			# 0 ->1 at that day
			#df[str(date.today())].replace(check_in_records_csv.loc[face_person, str(date.today())], 1, inplace = True)

			# calculate the sum(times)
			#check_in_records_csv.loc[face_person, str(date.today())] = 1

			
			if face_person not in check_in_records.keys():
				check_in_records.setdefault(face_person, 1)

			check_in_records[face_person] += 1
			
			#save
			check_in_records_txt_path.write_text(str(check_in_records))



			print("【Checking...】\n【This one is one of us!】\n【And it should be】 -> {name}, \nDetecting distance is {distance:.4f}".format(name = face_person, distance = distance[i]), end = "\n\n")
			print("-------【You're check in】------")
			print("---【You have workd for {number} days】---".format(number = check_in_records[face_person]))
			print("-------【Have a good day】------")
		


# check in information
def check_in_info(check_in_records_path):
	data_csv = pd.read_csv(check_in_records_path)
	pprint(data_csv)

	return data_csv


# check info records 
# csv file - initial
def check_in_records_csv_init(face_lib_names_list):

	data_index = face_lib_names_list

	data_columns = {str(date.today()): np.zeros(len(face_lib_names_list)),
							  "Times": np.zeros(len(face_lib_names_list))
	}

	data_csv = pd.DataFrame(data_columns, index = data_index)

	pprint(data_csv)

	# save
	data_csv.to_csv(check_in_records_path)
	print("check_in_records has been created!")


	return data_csv



# check info records 
# csv file - insert coloum
def check_in_records_csv_insert_column(data_csv: pd.DataFrame):
	data_csv.insert(len(data_csv.columns) - 1, "new column", np.zeros(len(face_lib_names_list)))
	#data_csv.insert(len(data_csv.columns) - 1, str(date.today()), np.zeros(len(face_lib_names_list)))
	
	pprint(data_csv)

	return data_csv

def check_in_info(check_in_records_txt_path):
	print("\n--------------\n")
	print("【Check-in Information:】")
	print(check_in_records_txt_path.read_text())
	print("")


### main()
if __name__ == "__main__":

	
	
	### checkIn list

	check_in_records = { 'Anne': 0,
						 'Professor': 0, 
						 'Kyrie': 0, 
						 'James': 0, 
						 'Trump': 0
	}


	### load and encoding
	### [image detected]

	image = fr.load_image_file(image_detected)
	#print(type(image))
	image_encoding = fr.face_encodings(image)[0]

	show_face_loc_crop(image)

	# load and encoding
	# [Face Lib]
	# returns -> [face_lib_encoding_list, face_lib_names_list]
	face_lib_encoding_list, face_lib_names_list = face_lib_encodings(face_lib_path)

	

	# Checking and output
	# return -> [reslult list]
	check_in_face_lib(image_encoding, face_lib_encoding_list, face_lib_names_list, 0.4)

	# create check_in_records
	#check_in_records = check_in_records_csv_init(face_lib_names_list)

	# check in info
	# check_in_records = check_in_info(check_in_records_path)
	check_in_info(check_in_records_txt_path)



