"""
Author: vitoloper

"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.patches as patches

ID = 'id'
IMAGES = 'images'
CATEGORIES = 'categories'
ANNOTATIONS = 'annotations'
FILE_NAME = 'file_name'
IMAGE_ID = 'image_id'
BBOX = 'bbox'

found = False
bboxes = []

if len(sys.argv) < 3:
	print ('Usage: {} image json'.format(sys.argv[0]))
	sys.exit(1)

# Read arguments from command line
imgfile = sys.argv[1]
jsonfile = sys.argv[2]

print('Image file: {}'.format(imgfile))
print('JSON COCO annotation file: {}'.format(jsonfile))

# Open file and load JSON content in datastore
if jsonfile:
	with open(jsonfile, 'r') as f:
		datastore = json.load(f)

# Check if image is present in JSON file
imgfilename = imgfile.split('/')[-1]	# Get file name only without path

for idx, image in enumerate(datastore[IMAGES]):
	if image[FILE_NAME] == imgfilename:
		found = True
		break

# Exit if image is not found in JSON file
if not found:
	print('Image not found in JSON file')
	sys.exit(1)

# Otherwise, get all the image annotations
image_id = datastore[IMAGES][idx][ID]
print('Image found in JSON file (id: ' + str(image_id) + ')')
for idx, annotation in enumerate(datastore[ANNOTATIONS]):
	if annotation[IMAGE_ID] == image_id:
		# Get the annotation associated with the image
		bboxes.append(annotation[BBOX])

print('Number of annotations: ' + str(len(bboxes)))

# bboxes contains the bboxes of each annotation
# print(bboxes)

# Open image files
im = np.array(Image.open(imgfile))

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Add rectangle to the axes
for bbox in bboxes:
	rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)

# Show plot
plt.show()
