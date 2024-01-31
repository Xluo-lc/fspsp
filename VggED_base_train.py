from dataloader import *
from VggED_base import *
import argparse
import logging
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import torch
import torchvision
from model import *
from tqdm import tqdm

def train():
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


    dataset = SalicondDataset(images_path=args.datapath, mode=args.mode, transform=trans)
    trainload = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True)
    model = DeepNet()
    model.to(device)
    #best_model_wts = copy.deepcopy(model.state_dict())

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainload, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            #print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            if i % 312 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 312))
                logging.info("==> Evaluating the model at: {}".format(epoch + 1))
                logging.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 312))
        if epoch % 20 == 0:
            torch.save(model.state_dict(), './model/epoch_{}.pth'.format(epoch))
            #torch.save(model.state_dict(), '{}/epoch_{}.pth'.format('./model', epoch))

    #model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './model/5_16.pth')



if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--datapath',help='Path of the dataset', default=r'C:\Users\18817\Documents\PHcode\DatasetAll\SALICON\images\train', type=str)
    parser.add_argument('--epochs',help='Number of training epochs',default=201,type=int)
    parser.add_argument('--log_name', help='Name of logging file', default='x', type=str)
    parser.add_argument('--mode', help='Name of logging file', default='train', type=str)
    parser.add_argument('--batch', help='Number of training batch', default=32, type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    logging.basicConfig(filename=args.log_name, level=logging.INFO)

    logging.info('Started training')
    train()
    logging.info('Finished training')


