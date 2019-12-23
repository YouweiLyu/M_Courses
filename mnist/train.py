import torch
import numpy as np
import argparse
import time

import Network
import utils
import datasets

def main(args):
    start_time = time.time()
    model = utils.build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestone, gamma=0.1, last_epoch=-1)
    train_loader, test_loader = datasets.dataloader_builder(args)

    train_loss = np.zeros(args.epochs)
    test_loss, test_accu = np.zeros(args.epochs), np.zeros(args.epochs)
    print('\n\r\t#### Start Training ####')
    for epoch in range(args.epochs):
        train_loss[epoch] = trainer(model, train_loader, optimizer, criterion)
        test_loss[epoch], test_accu[epoch] = tester(model, test_loader)
        scheduler.step()
        # print(scheduler.get_lr()[0])
        print('| Epoch: {0:3d} | Training Loss: {1:.6f} | Test Accuracy: {2:.2f} | Test Loss {3:.6f} |'.format(epoch, train_loss[epoch], test_accu[epoch], test_loss[epoch]))
    print('\t#### Time Consumed: {0:.3f} second ####\n\r'.format(time.time()-start_time))
    utils.saveCheckpoint(args.cp_dir, args.model_name, epoch, model, optimizer, test_accu, train_loss, args.bn, args.weight_decay)
    utils.plotCurve(args, train_loss/args.trn_batch, test_loss, test_accu)

def trainer(model, train_loader, optimizer, criterion):
    model.train()
    for idx, (img, label) in enumerate(train_loader):
        img, label = img.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor)
        pred = model(img)
        optimizer.zero_grad()
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        t_loss = loss.item()
    return t_loss

def tester(model, test_loader):
    model.eval()
    tst_loss, correct = 0, 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            img, label = img.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor)
            pred = model(img)
            tst_loss = torch.nn.functional.cross_entropy(pred, label, reduction='sum').item()
            pred_label = pred.max(1, keepdim=True)[1]
            correct += pred_label.eq(label.view_as(pred_label)).sum().item()
    t_loss = tst_loss / len(test_loader.dataset)
    t_accu = correct * 100 / len(test_loader.dataset)
    return t_loss, t_accu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr',         default=0.001)
    parser.add_argument('--epochs',     default=70)
    parser.add_argument('--model_name', default='LeNet_5_woPooling')
    parser.add_argument('--milestone',  default=[25,50])
    parser.add_argument('--bn',         default=True)
    parser.add_argument('--weight_decay',default=1e-5)

    parser.add_argument('--cp_dir',     default='./checkpoint/')
    parser.add_argument('--data_path',  default='./data/')
    parser.add_argument('--trn_batch',  default=200)
    parser.add_argument('--val_batch',  default=200)
    parser.add_argument('--num_workers',default=2)

    args = parser.parse_args()
    main(args)
