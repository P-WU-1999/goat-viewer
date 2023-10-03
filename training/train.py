import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from NNmodels import VGGGray

# Dataset path
TRAIN_PATH = '../datasets/train'
VALID_PATH = '../datasets/valid'

if torch.cuda.is_available:
    print("Using GPU: " + str(torch.cuda.get_device_name()))    # If using GPU
else:
    print("Using CPU")     # If using CPU

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
    transforms.ToTensor(),
])

# Load datasets using the DataLoader
dataset_train = ImageFolder(TRAIN_PATH, transform=transform)
dataset_valid = ImageFolder(VALID_PATH, transform=transform)
print("=============================================\nDataset List:")
print(dataset_train.class_to_idx)
print(dataset_valid.class_to_idx)
print("=============================================\n")

dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
dataloader_test = DataLoader(dataset_valid, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

# Create a directory for TensorBoard logs
log_dir = "logs"
writer = SummaryWriter(log_dir)

# Clean Tensorboard log files
files = os.listdir(log_dir)

for file in files:
    file_path = os.path.join(log_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Parameters
# It will automatically run multiple training attempts within a range of learning rate and save the best
init_lr = 0.0050        # The starting value of learning rate
end_lr = 0.0080         # The ending value of learning rate
lr_step = 0.0002        # The changed value of every time it changes
attempt_num = 5         # How many attempts will try on one learning rate
epoch = 300             # How many epoch per attempt
num_goats = 10          # How many goats you want classify

# Initialize the training
count = 0
best_acc = -1
his_best_acc = best_acc
his_best_att = -1

train_num = (((end_lr - init_lr) // lr_step) + 1) * attempt_num
lr = init_lr

# Start training
for count in range(int(train_num)):

    # Increase learning rate
    if (count + 1) % attempt_num == 0:
        lr += lr_step

    print("=============================================\nAttempt: " + str(count))
    print("learning rate = " + str(lr))

    # Load the training model
    model = VGGGray.VGG16Gray(num_classes=num_goats)
    model.eval()

    # Transfer model to GPU
    model = model.cuda()

    # Optimizer
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()        # transfer loss to GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.01)

    train_step = 0
    best_acc = -1

    # Start train loop
    for i in range(epoch):
        print("---------------------------------------------\nEpoch: " + str(i))
        model.train()

        for data in dataloader_train:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = train_step + 1

            # Information output
            if train_step % 100 == 0:
                writer.add_scalar("valid_loss " + format(count) + " " + format(lr), loss.item(), i)

        # Validating
        model.eval()
        total_test_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for data in dataloader_test:
                imgs, targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss+loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

            if total_accuracy > best_acc:   # save the best
                best_acc = total_accuracy

            if total_accuracy > his_best_acc:
                his_best_acc = total_accuracy
                his_best_att = count
                print("\n<Model updated> \nNew model accuracy = " + format(his_best_acc / len(dataset_valid)) + "\n")
                torch.save(model, "best.pth")

            print("total_loss:" + format(total_test_loss))
            print("Accuracy: " + format(total_accuracy / len(dataset_valid)))
            print("Attempt Best: " + format(best_acc / len(dataset_valid)))
            print("History best: " + format(his_best_acc / len(dataset_valid)))

            writer.add_scalar("Accuracy" + format(count) + " " + format(lr), (total_accuracy / len(dataset_valid)), i)

# Print the training data
print("=============================================\nTraining complete!")
print("Best Model:")
print("Attempt number: " + format(his_best_att))
print("Accuracy: " + format(his_best_acc / len(dataset_valid)))
print("=============================================")
writer.close()