import copy
import os
import random
import torch
import torchvision.models as models
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch import nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pillow_heif import register_heif_opener
import seaborn as sns

register_heif_opener()
# Set random seed for reproducibility
manualSeed = 999

random.seed(manualSeed)
torch.manual_seed(manualSeed)

data_root = '../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Images/'
image_size = 256
batch_size = 16
lr = 0.0002
beta1 = 0.5
num_epochs = 2000


def custom_collate(batch):
    # 'batch' is a list of tuples, where each tuple contains (data, label)
    # Extract the data from each tuple and store it in a list
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Return a tuple containing a single tensor with all the data samples and a list of labels
    return torch.stack(data), labels


class EarlyStopping:
    def __init__(self, patience=100, save_path=''):
        self.patience = patience
        self.counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.best_epoch = -1

    def __call__(self, epoch, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path + 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch}. Loading the best model from epoch {self.best_epoch}.")
                model.load_state_dict(torch.load(self.save_path + 'best_model.pth'))
                return True  # Stop training
        return False  # Continue training


def train_early_stopping(model, train_loader, valid_loader, lr, beta1, num_epochs, save_model=True, save_path='',
                         save_interval=10, early_stopping_flag=True, patience=100, use_best_accuracy_model=True):
    with open(save_path + 'run_info.txt', 'w') as f:
        f.write(f'Learning rate: {lr}\n')
        f.write(f'Beta1: {beta1}\n')
        f.write(f'Number of epochs: {num_epochs}\n')

    # Initialize model, optimizer, and criterion
    optimizerG = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize EarlyStopping callback
    early_stopping = EarlyStopping(patience=patience, save_path=save_path)
    best_accuracy = 0
    best_epoch = 0
    # Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        for i, data in enumerate(train_loader):
            images, labels = data['data'], data['label']
            images = images.to(device)
            labels = labels.to(device)

            optimizerG.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizerG.step()
            batch_losses.append(loss.item())
            # if i % 10 == 0:
            #     print('[%d/%d][%d/%d]\tLoss: %.8f'
            #           % (epoch, num_epochs, i, len(train_loader), loss.item()))
        average_loss = np.mean(batch_losses)
        print(f'Epoch {epoch} average loss: {average_loss}')
        epoch_losses.append(average_loss)
        # Validation accuracy calculation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                model.eval()
                images, labels = data['data'], data['label']
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # _, predicted = torch.max(outputs.data, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probs, dim=1)
                total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                correct += (predicted_classes == labels.long()).sum().item()
            model.train()
        accuracy = 100 * correct / total
        print('Accuracy of the network on the validation images: %d %%' % accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'best_epoch_model.pth')
            print(f'Best model so far at epoch {best_epoch}. Accuracy: {best_accuracy}')

        # Save model
        if save_model and epoch % save_interval == 0:
            torch.save(model.state_dict(), save_path + 'resnet_model_epoch' + str(epoch) + '.pth')

        # Check for early stopping
        if early_stopping_flag:
            if early_stopping(epoch, model, accuracy):
                break  # Stop training

    with open(save_path + 'epoch_losses.txt', 'w') as ff:
        for loss in epoch_losses:
            ff.write(f'{loss}\n')
        ff.close()
    if use_best_accuracy_model:
        print(f'Best model so far at epoch {best_epoch}. Accuracy: {best_accuracy}')
        print(f'Loading the best model from epoch {best_epoch}.')
        model.load_state_dict(torch.load(save_path + 'best_epoch_model.pth'))
    # Save final model
    if save_model:
        torch.save(model.state_dict(), save_path + 'resnet_model_final.pth')
    return model


def train(train_loader, valid_loader, lr, beta1, num_epochs, save_model=True, save_path='', save_interval=10):
    with open(save_path + 'run_info.txt', 'w') as f:
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Learning rate: {lr}\n')
        f.write(f'Beta1: {beta1}\n')
        f.write(f'Number of epochs: {num_epochs}\n')
    optimizerG = torch.optim.Adam(resnet_model.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images, labels = data['data'], data['label']
            images = images.to(device)
            labels = labels.to(device)
            optimizerG.zero_grad()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            # loss = torch.nn.functional.cross_entropy(outputs, labels) # this does not apply softmax internally
            loss.backward()
            optimizerG.step()
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.8f'
                      % (epoch, num_epochs, i, len(train_loader),
                         loss.item()))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data['data'], data['label']
                images = images.to(device)
                labels = labels.to(device)
                outputs = resnet_model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
        print('Accuracy of the network on the validation images: %d %%' % (
                100 * correct / total))
        if save_model and epoch % save_interval == 0:
            torch.save(resnet_model.state_dict(), save_path + 'resnet_model_epoch' + str(epoch) + '.pth')
    if save_model:
        torch.save(resnet_model.state_dict(), save_path + 'resnet_model_final.pth')


def class_prediction(model, model_path, input_images, device='cpu'):
    # TODO: Add functionality to predict on a single image or a group of images
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    with torch.no_grad():
        images = input_images.to(device)
        outputs = model(images)
        predicted_classes = torch.argmax(outputs, dim=1)  # Get predicted class labels


def image_class_prediction(model, model_path, input_images, image_names, device='cpu', save_flag=False, save_path='',
                           evaluate_predictions=True, labels=[], output_name='predictions',
                           classes=None):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with (torch.no_grad()):
        # images = input_images.to(device)
        images = torch.stack([tensor for tensor in input_images], dim=0).to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probs, dim=1)
        predicted_classes = predicted_classes.to('cpu').numpy()
        if evaluate_predictions:
            correct = 0
            total = 0
            for i in range(len(predicted_classes)):
                if predicted_classes[i] == labels[i]:
                    correct += 1
                total += 1
            print(
                'Accuracy of the network on the test images: %d %%' % (accuracy_score(labels, predicted_classes) * 100))
            print('Recall of the network on the test images: %d %%' % (
                    recall_score(labels, predicted_classes, average='macro') * 100))
            print('Precision of the network on the test images: %d %%' % (
                    precision_score(labels, predicted_classes, average='macro') * 100))
            print('F1 score of the network on the test images: %d %%' % (
                    f1_score(labels, predicted_classes, average='macro') * 100))
            print('Confusion matrix of the network on the test images: \n', (
                confusion_matrix(labels, predicted_classes)))
        df = pd.DataFrame({'image_name': image_names,
                           'class': [classes[i] for i in predicted_classes],
                           'class_name': [classes[i] for i in labels]})


        # Create confusion matrix
        conf_mat = confusion_matrix(labels, predicted_classes)

        # Setting up the figure size and DPI for clarity
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Adjust figsize and dpi as needed

        # Create heatmap
        sns.set(font_scale=1.5)  # Increase font scale for better readability
        heatmap = sns.heatmap(conf_mat, annot=True, cmap='Greens', square=True, cbar=False,
                              annot_kws={'size': 20},  # Increase annotation text size
                              xticklabels=['alligator', 'longitudinal', 'transverse'],
                              yticklabels=['alligator', 'longitudinal', 'transverse'],
                              ax=ax)
        # Setting title and labels with adjusted sizes
        heatmap.tick_params(axis='x', labelsize=16)
        heatmap.tick_params(axis='y', labelsize=16)
        heatmap.set_title('ResNet18 (C-WGAN-GP Augmentation - Best Epoch) Confusion Matrix', pad=30)
        heatmap.xaxis.tick_top()  # x axis on top
        heatmap.xaxis.set_label_position('top')
        # Adjust layout
        plt.tight_layout()  # Automatically adjust subplot params
        # Save the figure
        fig.savefig(os.path.join(save_path, 'confusion_matrix.png'))

        # Show plot
        plt.show()

        if save_flag:
            df.to_csv(save_path + f'{output_name}.csv')
            with open(save_path + 'metrics.txt', 'w') as f:
                f.write(
                    f'Accuracy of the network on the test images: {accuracy_score(labels, predicted_classes) * 100}\n')
                f.write(
                    f'Recall of the network on the test images: {recall_score(labels, predicted_classes, average="macro") * 100}\n')
                f.write(
                    f'Precision of the network on the test images: {precision_score(labels, predicted_classes, average="macro") * 100}\n')
                f.write(
                    f'F1 score of the network on the test images: {f1_score(labels, predicted_classes, average="macro") * 100}\n')
                f.write(
                    f'Confusion matrix of the network on the test images: \n {confusion_matrix(labels, predicted_classes)}\n')
        return df


def index_images(directory_path, classes, transform_func=None, single_label=None):
    images = []  # To store the opened images
    files = []  # To store the class indices
    for class_name in classes:
        class_images = []
        class_files = []
        class_path = os.path.join(directory_path, class_name)  # Full path to the class directory
        image_files = os.listdir(class_path)  # List of all image files in the directory
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)  # Full path to the image file
            img = Image.open(image_path)
            class_files.append(image_file)
            # Perform any necessary operations on 'img' here
            class_images.append(img)
        transformed_images = [transform_func(image) for image in class_images]
        images.append(transformed_images)
        files.append(class_files)
    if single_label is not None:
        file_classes = np.full(len(files[0]), single_label)
    else:
        file_classes = np.concatenate([np.full(len(files[i]), i) for i in range(len(files))])
    images = np.array([item for row in images for item in row])

    files = [item for row in files for item in row]

    return images, files, file_classes


def augment_class(num_images, class_name, class_path, transform_func=None):
    files = os.listdir(class_path)
    random.shuffle(files)
    if num_images > len(files):
        print('Number of images to augment is greater than the number of images in the class')
        num_images = len(files)
    images = []
    for i in range(num_images):
        images.append(Image.open(os.path.join(class_path, files[i])))
    print(f'Augmenting class: {class_name}, number of images: {num_images}')
    transformed_images = [transform_func(image) for image in images]
    return np.array(transformed_images)


def get_previous_split(train_split_address, valid_split_address):
    """
    split the data based on the previous split
    """
    with open(train_split_address, 'r') as f:
        train_files = f.readlines()
    train_files = [x.strip() for x in train_files]
    with open(valid_split_address, 'r') as f:
        valid_files = f.readlines()
    valid_files = [x.strip() for x in valid_files]
    return train_files, valid_files


def split_data_prev(train_files, valid_files):
    """
    split the data based on the previous split
    """
    train_indices = []
    valid_indices = []
    for i, file in enumerate(files):
        if file in train_files:
            train_indices.append(i)
        elif file in valid_files:
            valid_indices.append(i)
    return train_indices, valid_indices


def split_data(images, files, file_classes, new_split=True, test_size=0.2, random_state=manualSeed,
               train_split_address='', valid_split_address=''):
    if new_split:
        train_indices, valid_indices = train_test_split(np.arange(len(images)), test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=file_classes)
    else:
        train_files, valid_files = get_previous_split(train_split_address, valid_split_address)
        train_indices, valid_indices = split_data_prev(train_files, valid_files)
    train_data = images[train_indices]
    train_labels = file_classes[train_indices]
    train_files = [files[i] for i in train_indices]
    valid_data = images[valid_indices]
    valid_labels = file_classes[valid_indices]
    valid_files = [files[i] for i in valid_indices]
    return train_data, train_labels, train_files, valid_data, valid_labels, valid_files


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.stack([tensor for tensor in data], dim=0)  # Convert data and labels to PyTorch tensors
        self.labels = torch.tensor(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        if self.transform:
            self.data[idx] = self.transform(self.data[idx])

        return sample


if __name__ == '__main__':
    run_address = './output/classification/ResNet18/augmented_CWGAN/test2'
    # run_address = './output/classification/ResNet18/augmented_WGAN/test7'
    # run_address = './output/classification/ResNet18/test24'
    # run_address = './output/classification/ResNet18/detection/test2'
    if not os.path.exists(run_address):
        os.makedirs(run_address)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))]  # changed to a single channel
    )
    original_classes_name = ['alligator', 'longitudinal', 'transverse']
    # classes_name = ['alligator', 'longitudinal', 'transverse','no_crack']
    classes_name = ['alligator','longitudinal', 'transverse']
    images, files, file_classes = index_images(os.path.join(data_root, 'class_seperated'), classes_name,
                                               transform_func=transform)
    df = pd.DataFrame(columns=['image_name', 'class'], data={'image_name': files, 'class': file_classes})
    df.to_csv(f'{run_address}/labels_og.csv')
    train_data, train_labels, train_files, valid_data, valid_labels, valid_files = split_data(images, files,
                                                                                              file_classes,
                                                                                              test_size=0.2,
                                                                                              new_split=True,
                                                                                              train_split_address=f'{run_address}/train_files.txt',
                                                                                              valid_split_address=f'{run_address}/valid_files.txt')

    # Adding more images to the block validation data
    # new_images, new_files, new_file_classes = index_images(
    #     '../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Images/', ['block'],
    #     transform_func=transform, single_label=classes_name.index('block'))
    # valid_data = np.concatenate((valid_data, new_images))
    # valid_labels = np.concatenate((valid_labels, new_file_classes))
    # valid_files = np.concatenate((valid_files, new_files))

    with open(f'{run_address}/train_files.txt', 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)
    with open(f'{run_address}/valid_files.txt', 'w') as f:
        for item in valid_files:
            f.write("%s\n" % item)

    # Data augmentation
    for class_name in original_classes_name:
        augmented_data = augment_class(num_images=200, class_name=class_name,
                                       class_path=os.path.join(data_root, f'augmented/C_WGAN_GP/{class_name}'),
                                       transform_func=transform)

        train_data = np.concatenate((train_data, augmented_data))
        train_labels = np.concatenate((train_labels, np.full(len(augmented_data), classes_name.index(class_name))))

    num_classes = len(classes_name)

    resnet_model = models.resnet18(pretrained=False)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
    # resnet_model.fc = nn.Sequential(nn.Linear(resnet_model.fc.in_features, num_classes), nn.Softmax(dim=1))
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model.to(device)
    train_dataset = CustomDataset(train_data, train_labels)
    valid_dataset = CustomDataset(valid_data, valid_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # best_resnet_model = train_early_stopping(model=resnet_model, train_loader=train_loader, valid_loader=valid_loader,
    #                                          lr=lr,
    #                                          beta1=beta1,
    #                                          num_epochs=num_epochs,
    #                                          save_model=True,
    #                                          save_path=f'{run_address}/', save_interval=100, patience=300,
    #                                          early_stopping_flag=False,
    #                                          use_best_accuracy_model=True)

    image_class_prediction(model=resnet_model, model_path=f'{run_address}/resnet_model_final.pth', device=device,
                           image_names=valid_files, input_images=valid_data, labels=valid_labels, save_flag=True,
                           save_path=f'{run_address}/',
                           evaluate_predictions=True, classes=classes_name)
