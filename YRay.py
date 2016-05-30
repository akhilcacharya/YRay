"""
YRay
"""
import sys
import cv2
import numpy
import glob

N_CLASSES = 0
SEPARATOR = "\\"
CASCADE_XML = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_XML)

recognizer = cv2.createEigenFaceRecognizer()

def extract_face(img_path):
    img = cv2.imread(img_path)
    grayscale = cv2.imread(img_path, 0)

    faces = face_cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.5,
        minNeighbors=1,
        minSize=(10, 10),
        flags = 0
    )

    if len(faces) != 1:
        return numpy.asarray([])

    (x, y, w, h) = faces[0]

    # Crop the image to 30, 30
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cropped = img[y:y+h, x:x+w]

    #scale to 50x50
    resized = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_AREA)

    # Return image as nparray if single face, or None if multiple

    result = numpy.asarray(resized, dtype=theano.config.floatX)

    return result

def get_faces(folder):
    global N_CLASSES

    faces = []
    # Iterate through all folders in the folder
    for label_dir in glob.glob(folder + SEPARATOR + "*"):
        # Folder name is the label
        label = label_dir.split(SEPARATOR)[1]
        N_CLASSES = N_CLASSES + 1
        for img_dir in glob.glob(label_dir + SEPARATOR + "*"):
            # Grab face as nparray
            img = extract_face(img_dir)
            if len(img) > 0:
                faces.append((img, label))
    return faces

def train_faces(face_tuples):
    # Decompose tuple list into arrays of images (X) & arrays of labels (Y)
    X = []
    Y = []

    unique_labels = {}

    for t in face_tuples:
        X.append(t[0])
        unique_labels[t[1]] = True

    for t in face_tuples:
        #Set label vector
        Y.append(unique_labels.keys().index(t[1]))

    X_train = X[:int(len(X) * 0.7)]
    X_test = X[int(len(X) * 0.7):]

    Y_train = Y[:int(len(Y) * 0.7)]
    Y_test = Y[int(len(Y) * 0.7):]

    print "Training Dataset"

    recognizer.train(numpy.asarray(X_train), numpy.asarray(Y_train))

    print "Testing Dataset"
    idx = 0
    for img in X_test:
        prediction, conf = recognizer.predict(img)
        print "Actual: %s, Prediction: %s, Conf: %f" % (unique_labels.keys()[Y_test[idx]], unique_labels.keys()[prediction], conf)
        idx = idx + 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: YRay_train.py data/"
        exit()
    folder = sys.argv[1]
    print "Getting faces"
    faces = get_faces(folder)
    print "Got %d faces" % (len(faces))
    train_faces(faces)
