import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from dataset import CardsDataset
import logging
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random

logging.basicConfig(filename='test.log', filemode='w', format='%(message)s')
PATH = "cnn.pt"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=2),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(64),
                                            nn.MaxPool2d(2),

                                            nn.Conv2d(64, 128, kernel_size=3),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(128),
                                            nn.MaxPool2d(2),

                                            nn.Conv2d(128, 192, kernel_size=3),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(192),
                                            nn.MaxPool2d(2),

                                            )

        self.classifier = nn.Sequential(nn.Linear(114240, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 40))

    def forward(self, x):
        x = self.feat_extractor(x)
        n, c, h, w = x.shape
        x = x.view(n, -1)
        x = self.classifier(x)
        return x


class Trainer(object):
    def __init__(self, model, device, train_loader, test_loader, criterion,
                 optimizer):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.softmax = nn.Softmax(dim=1)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epochs=10):
        print('Initial accuracy: {}'.format(self.evaluate()))
        for epoch in range(1, epochs + 1):
            self.model.train()  # set the model to training mode
            for images, labels in tqdm(self.train_loader,
                                       total=len(self.train_loader)):
                self.optimizer.zero_grad()  # don't forget this line!
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.softmax(self.model(images))
                loss = self.criterion(output, labels)
                loss.backward()  # compute the derivatives of the model
                optimizer.step()  # update weights according to the optimizer

            print('\nTest Accuracy at epoch {}: {}'.format(epoch, self.evaluate(epoch)))

    def evaluate(self,epoch=0):
        self.model.eval()  # set the model to eval mode
        total_train = 0
        total_test = 0
        logging.warning("E, {}".format(epoch))
        for images, labels in tqdm(self.test_loader,
                                   total=len(self.test_loader)):

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.softmax(self.model(images))
            predicted = torch.max(output, dim=1)[1]  # argmax the output
            total_test += (predicted == labels).sum().item()


        for images, labels in tqdm(self.train_loader,
                                   total=len(self.train_loader)):

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.softmax(self.model(images))
            predicted = torch.max(output, dim=1)[1]  # argmax the output
            total_train += (predicted == labels).sum().item()

        logging.warning("{}, {}".format(total_test / len(self.test_loader.dataset),
                                        total_train / len(self.train_loader.dataset)));
        return  total_test / len(self.test_loader.dataset), total_train / len(self.train_loader.dataset)

seeds = ["bastoni", "spade", "coppe", "denari"]
cards_names = ["asso", "2", "3", "4", "5", "6", "7", "fante", "cavallo", "re"]

def show_example(path, prediction):
    image = Image.open(path)
    font = ImageFont.truetype("arial.ttf", 25)

    x = 75
    y = 550

    draw = ImageDraw.Draw(image)
    w, h = font.getsize(prediction)
    draw.rectangle((x, y, x + w, y + h), fill='white')

    ImageDraw.Draw(
        image  # Image
    ).text(
        (x, y),  # Coordinates
        prediction,  # Text
        (0, 0, 0),  # Color
        font = font
    )
    image.show()

def get_random_image():
    random_seed = random.choice(seeds)
    random_card = random.choice(cards_names)

    random_file = random.choice(os.listdir("data/"+random_card+'_'+random_seed+'/'))
    return "data/"+random_card+'_'+random_seed+'/'+random_file

def get_example(loaded_model):
    transform_image = transforms.Compose([transforms.Resize((600, 300)), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    img_path = get_random_image()

    input_image = Image.open(img_path)

    input_image = transform_image(input_image)
    input_image = input_image.unsqueeze(0)

    if torch.cuda.is_available():
        input_image = input_image.cuda()

    softmax = nn.Softmax(dim=1)
    output = softmax(loaded_model(input_image))
    pred = torch.max(output, dim=1)[1]  # argmax the output
    pred = pred.item()

    str_pred = cards_names[pred - int(pred/10)*10] + ' di ' + seeds[int(pred/10)]

    return img_path, str_pred



if __name__ == '__main__':
    config = {'lr': 1e-4,
              'momentum': 0.9,
              'weight_decay': 0.001,
              'batch_size': 8,
              'epochs': 120,
              'device': 'cuda:0',
              'seed': 314}

    # set the seeds to repeat the experiments
    print(torch.cuda.get_device_name(0))

    if 'cuda' in config['device']:
        torch.cuda.manual_seed_all(config['seed'])
    else:
        torch.manual_seed(config['seed'])


    transform = transforms.Compose([transforms.Resize((600,300)),transforms.ToTensor(),
                                    transforms.Normalize((0.5,),
                                                         (0.5,))])
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    batch_size = 8
    train_loader = DataLoader(CardsDataset(img_dir="data",train=True,transform=transform),
        batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(
        CardsDataset(img_dir="data",train=False,transform=transform),
        batch_size=config['batch_size'], shuffle=False)

    if os.path.isfile(PATH):
        model = torch.load(PATH)
        model.eval()
        trainer = Trainer(model, config['device'], train_loader, test_loader,
                          criterion, optimizer)
        print('\nTest Accuracy at epoch {}: {}'.format(1, trainer.evaluate(1)))

        path, prediction = get_example(model)
        show_example(path, prediction)
    else:
        trainer = Trainer(model, config['device'], train_loader, test_loader,
                          criterion, optimizer)
        trainer.train(epochs=config['epochs'])
        torch.save(model,PATH)
