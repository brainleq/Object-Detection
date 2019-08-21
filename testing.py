from imutils import paths
import numpy as np
import cv2
import sys

"""
Initial testing for object recognition
"""

# using python 3.6.8
print('Python Version: ' + sys.version)

# load network model and object ids
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
	"bvlc_googlenet.caffemodel")

# preprocess images to blobs
imagePaths = sorted(list(paths.list_images("images")))
image = cv2.imread(imagePaths[7])
resized = cv2.resize(image, (224, 224))
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))

# feed into network and get output probabilities
net.setInput(blob)
preds = net.forward()

# show the highest probability object
id = np.argmax(preds[0])
text = "Label: {}, {:.2f}%".format(classes[id],
	preds[0][id] * 100)
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
