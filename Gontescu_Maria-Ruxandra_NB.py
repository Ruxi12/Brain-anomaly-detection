import numpy as np
import sklearn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from keras.utils import img_to_array
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# sa le avem ca string
train_labels = pd.read_csv('unibuc-brain-ad/data/train_labels.txt', dtype=str)
val_labels = pd.read_csv('unibuc-brain-ad/data/validation_labels.txt', dtype=str)
sample_labels = pd.read_csv('unibuc-brain-ad/data/sample_submission.txt', dtype=str)

def read_image(id):
    image = Image.open("unibuc-brain-ad/data/data/" + id + '.png')
    image = image.convert('L')
    return img_to_array(image)


train_data = [read_image(row['id']) for _, row in train_labels.iterrows()]

train_data = np.array(train_data)
#train_data = train_data.reshape(-1, 224, 224, 1) / 255.0


# transformare in tensori
train_data = np.array(train_data)
# le si normalizam
# normalizare standard
mean_train = np.mean(train_data, axis = 0)
std_train = np.std(train_data, axis = 0)
train_data = (train_data - mean_train) / std_train


val_data = [read_image(row['id']) for _,row in val_labels.iterrows()]
val_data = (val_data - mean_train) / std_train
val_data = np.array(val_data)

test_data = [read_image(row['id']) for _,row in sample_labels.iterrows()]
test_data = np.array(test_data)
test_data = (test_data - mean_train) /  std_train

norm_val = np.sum(np.abs(test_data ** 2), axis=1, keepdims=True) + 10 ** -8
test_data = test_data / norm_val
test_data = np.array(test_data)
# genereaza num_bins intervale
def get_intervals(num_bins):
    bins = np.linspace(start=0, stop=256, num=num_bins)
    return bins

def values_to_bins(x,bins):
    new_x = np.zeros(x.shape)
    for i, elem in enumerate(x):
        new_x[i] = np.digitize(elem, bins) # index of the bin that each element in x belongs to
    return new_x - 1

train_et = []
for _, row in train_labels.iterrows():
  train_et.append(int(row['class']))

n_samples = train_data.shape[0]
n_features = train_data.shape[1] * train_data.shape[2] * train_data.shape[3]
train_data = train_data.reshape(n_samples, n_features)


print("MultinomialNB")
# transform to bins
bins = get_intervals(10)
train_data = values_to_bins(train_data,bins)
val_data = values_to_bins(val_data, bins)
test_data = values_to_bins(test_data, bins)
# instanta a clasei
classifier = MultinomialNB()
train_et = np.array(train_et)
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
classifier.fit(train_data, train_et)

# lucrez pe validare
val_et = []
for _, row in val_labels.iterrows():
    val_et.append(int(row['class']))

val_data = val_data.reshape(val_data.shape[0], val_data.shape[1] * val_data.shape[2] * val_data.shape[3])
val_data = scaler.fit_transform(val_data)

# vedem scorul pe validare
score = classifier.score(val_data, val_et)
print("Scorul pe validare ", score)

y_pred = classifier.predict(val_data)

f1 = f1_score(val_et, y_pred)
print("F1 score ", f1)
precision = precision_score(val_et, y_pred)
print('Preciosion =', precision)
recall = recall_score(val_et, y_pred)
print ('Recall = ', recall)
cm = confusion_matrix(val_et, y_pred)
print(cm)
# afisare
test_et = []
for _, row in sample_labels.iterrows():
    test_et.append(int(row['class']))
# reshape 4D array to 2D
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2] * test_data.shape[3])
#test_data - same shape as before, values scaled  0 - 1.
test_data = scaler.fit_transform(test_data)

predict = classifier.predict(test_data)
with open ('output.csv', 'w') as f:
    f.write("id,class\n")
    for i, row in sample_labels.iterrows():
        output = row['id'] + ',' + str(predict[i]) + "\n"
        f.write(output)


