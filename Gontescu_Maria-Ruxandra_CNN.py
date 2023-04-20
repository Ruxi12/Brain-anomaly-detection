import numpy as np
import sklearn
from PIL import Image
from keras_preprocessing.image import img_to_array
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# for dataloder
class CTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def read_image(id):
    image = Image.open("unibuc-brain-ad/data/data/" + id + '.png')
    return img_to_array(image)

# sa le avem ca string
train_labels = pd.read_csv('unibuc-brain-ad/data/train_labels.txt', dtype=str)
val_labels = pd.read_csv('unibuc-brain-ad/data/validation_labels.txt', dtype=str)
sample_labels = pd.read_csv('unibuc-brain-ad/data/sample_submission.txt', dtype=str)

# array cu array-uri ( imaginile transf in numpy )
train_data = [read_image(row['id']) for _,row in train_labels.iterrows()]
# transformare in tensori
train_data = np.array(train_data)
# le si normalizam
# normalizare standard
mean_train = np.mean(train_data, axis = 0)
std_train = np.std(train_data, axis = 0)
train_data = (train_data - mean_train) / std_train


train_data = torch.from_numpy(train_data)

val_data = [read_image(row['id']) for _,row in val_labels.iterrows()]
val_data = np.array(val_data)

val_data = (val_data - mean_train) / std_train

val_data = torch.from_numpy(np.array(val_data) )


test_data = [read_image(row['id']) for _,row in sample_labels.iterrows()]
test_data = np.array(test_data)
test_data = (test_data - mean_train) /  std_train
test_data = torch.from_numpy(np.array(test_data) )

# Define data augmentation transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, hue=0.15, saturation=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_et = []
for _, row in val_labels.iterrows():
    val_et.append(row['class'])

# creare dataset
train_dataset = CTDataset(train_data, train_labels['class'].astype('int'))
val_dataset = CTDataset(val_data, val_labels['class'].astype('int'))

# Apply data augmentation to the training dataset
for data, labels in train_dataset:
    # print(data.shape)
    data= data.reshape(-1, 224, 224, 1) / 255.0
    data = torch.squeeze(data)
    data = transform(data)
    data = np.array(data)
    data = torch.from_numpy(data)
print("s-a realizat data augmentation")

# ii da cate 32 de imagini
# am trimis 64 de poze pentru ca in acest mod exista sanse mai mari sa avem si una din clasa minoritara
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


class CTClassifier(nn.Module):
    def __init__(self):
        super(CTClassifier, self).__init__()

        # convolutional layer that takes a 3-channel input image and produces a 64-channel output,
        # using a 3x3 kernel with stride 1 and 1-pixel padding.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # Input channels: 64   Output : 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1000, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        # CONV = dot product between their weights and a small region they are connected to in the input volume
        # POOL =  downsampling operation along the spatial dimensions
        x = x.view(-1, 224, 224, 3)    # different shape -  no problems for output
        x = torch.permute(x, (0, 3, 1, 2))   # change the order of dimensions of the tensor
        x = self.conv1(x)
        x = self.conv12(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv22(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.pool5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


device=torch.device('mps')
# instanta a modelului
model = CTClassifier()
# se va antrenena pe cpu
model.to(device)

counts = train_labels['class'].value_counts()
# 2 pentru ca avem 2 clase de 0 si 1

weights = [np.float32(counts['0']/ train_labels.shape[0]), np.float32(counts['1'] / train_labels.shape[0])]
# 1 -
weights = torch.from_numpy(np.array(weights)).to(device)

print(weights)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weights)  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=0.00001, threshold=0.01, verbose=True, factor=0.05)
# lr = learning rate
f1 = []
max_epochs = 100
no_improvement = 0
last_val_loss = 18661636
earl_count = 20
# Train the model
predicted_et = []
for epoch in range(max_epochs):
    train_loss = 0
    model.train()
    for i, (data, labels) in enumerate(train_loader):   # primeste un batch de 64 de imagini
        data, labels = data.to(device), labels.to(device)
        # Forward pass
        outputs = model(data)
        # torch tine minte toate operatiile ce au dus la calcularea lui loss
        loss = criterion(outputs, labels)
        # Backward and optimize  -  initializare
        optimizer.zero_grad()
        # ia tot arborele de computatii care a dus la loss, il parcurge invers si calculeaza gradientele
        loss.backward()
        # optimizer decide cum sa schimbe valorile
        optimizer.step()
        # .item ca sa scot valoarea numerica, nu arborele
        train_loss += loss.item()
    # Validate the model
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0
        # turn to evaluation mode
        model.eval()
        for i, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels.to(device))
            val_loss += loss.item()
            #_, predicted = torch.max(outputs.data, 1)
            # calculez f1 pentru fiecare epoca
            predicted = torch.argmax(outputs.cpu(), axis=1) # tensor cu rezultate; dim: (# of samples)
            for elem in predicted:
                predicted_et.append(elem)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tp_aux = (predicted == (torch.mul(labels,2) - torch.tensor([1]))).sum().item()
            tp += tp_aux
            fp += (predicted == torch.ones(predicted.shape[0])).sum().item() - tp_aux
            fn += (labels == torch.ones(predicted.shape[0])).sum().item() - tp_aux

    print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, max_epochs, train_loss, val_loss))
    # verificare early stopping
    scheduler.step(val_loss)
    if val_loss * 1.001 > last_val_loss:
        no_improvement += 1
    else:
        no_improvement = 0
        last_val_loss = val_loss
    if no_improvement == earl_count:
        print('Early stopping, no improvement in last {} epochs'.format(earl_count))
        # ii salvam si state-ul
        break
    try:
        p = tp/(tp+fp)   # precision
        r = tp/(tp + fn)   # recall
        m = 100 * 2 * p * r/(p+r)
        f1.append[m]
        print('Accuracy of the model on validation: {} %; F1 score on validation: {} %'.format(100 * correct / total, m))
    except:
        print('Accuracy of the model on validation: {}%'.format(100 * correct / total))

    #     break
    model.train()
    # print('Accuracy of the model on validation: {} %; F1 score on validation: {} %'.format(100 * correct / total, 100 * 2 * p * r/(p+r)))
    # print('Accuracy of the model on validation: {}; F1 score on validation:  %'.format(100 * correct / total))
# fac precision intre valorile de validare si cele prezise
precision = precision_score(val_et, predicted_et)
print('Precision for this model: {:.4f}'.format(precision))
recall = recall_score(val_et, predicted_et)
print('Recall for this model:{:.4f}'.format(recall))
cm = confusion_matrix(val_et, predicted_et)
print(cm)
with torch.no_grad():
    with open('output.csv', 'w') as f:
        f.write("id,class\n")
        model.eval()
        for i, row in sample_labels.iterrows():
            output = model(test_data[i].to(device))
            result = torch.argmax(output.cpu()).item()
            output = row['id'] + "," + str(result) + "\n"
            f.write(output)




