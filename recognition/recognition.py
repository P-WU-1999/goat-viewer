import os
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATA_PATH = "../datasets/recognition"     # The data for recognition should be preprocessed
MODEL_PATH = "model.pth"             # Path of your recognition model

num_goats = 10      # Number of goats

true_labels = []
predicted_labels = []
group_id = 0
i = 0

# Read the folder
subnames = os.listdir(DATA_PATH)

for foler_names in subnames:
    path = DATA_PATH + '/' + str(foler_names)
    print("=============================================\nPath: " + str(path))
    print("=============================================")
    names = os.listdir(path)

    correct = 0
    total = len(names)
    group_id += 1

    for name in names:
        i = i+1
        # This should be same as the training data index
        data_class = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        image_path = os.path.join(path, name)
        image = Image.open(image_path)
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),
                                                  torchvision.transforms.ToTensor()])
        image = transforms(image)

        model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

        image = torch.reshape(image, (1, 1, 64, 64))      # Should be same as the training data
        model.eval()
        with torch.no_grad():
            output = model(image)

        result = data_class[int(output.argmax(1))]

        # Print the result
        print("image: " + format(name) + ", Result: " + format(result))

        # Record the result
        true_labels.append(group_id)
        predicted_labels.append(int(result))

# Define the class labels (assuming 3 classes in this example)
class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Normalize the confusion matrix
normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a heatmap of the normalized confusion matrix
sns.heatmap(normalized_cm, annot=True, cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

# Add labels and a title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')

# Show the plot
plt.show()
