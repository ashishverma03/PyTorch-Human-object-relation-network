import numpy
import pickle
import pdb

CLASSES = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
                'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')

with open("test_scores", "rb") as fl:
	scores = pickle.load(fl)

with open("image_test", "rb") as fb:
	image_names = pickle.load(fb)
# pdb.set_trace()

for i in range(len(CLASSES)):
	text_file_name = 'comp10_action_test_'+CLASSES[i]+'.txt'

	for j in range(len(image_names)):
		image_name = image_names[j]
		image_name = image_name[0]
		score = scores[j]
		num_objects = score.shape[1]
		# pdb.set_trace()

		for k in range(num_objects):

			with open(text_file_name, "a") as f:
				f.writelines(f"{image_name} {k+1} {score[0][k][i]}\n")
				# print(f"my roll is {1234}")
				f.close()