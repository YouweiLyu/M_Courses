import numpy as np
import torch
import struct
import os
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import datasets

def decode_image_file(file_path):
    """
    Decode the mnist image file
    """ 
    bin_data = open(file_path, 'rb').read()
    offset = 0
    fmt_header = '>4i' 
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('magic number:%d, the number of the images: %d, size of a image: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    fmt_image = '>' + str(image_size) + 'B'  
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_label_file(file_path):
    """
    Decode the mnist label file
    """
    bin_data = open(file_path, 'rb').read()
    offset = 0
    fmt_header = '>2i'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number:%d, the number of the images: %d' % (magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def build_model(args):
    print("Creating {} Model".format(args.model_name))
    models = __import__('Network.'+ args.model_name)
    model_file = getattr(models, args.model_name)
    model = getattr(model_file, args.model_name)(args.bn)
    model = model.cuda()
    print(model)
    return model

def saveCheckpoint(save_path, model_name, epoch=-1, model=None, optimizer=None, accu=0, loss=0, bn=False, w_d=0):
    state   = {'state_dict': model.state_dict(), 'model': model_name}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'accuracy':accu, 'loss':loss}
    if bn:
        torch.save(state,   os.path.join(save_path, '{}_wBN_EPC({})_wd({})_AC({:.2f}).pth.tar'.format(model_name, epoch, w_d, accu[-1])))
        torch.save(records, os.path.join(save_path, '{}_wBN_EPC({})_wd({})_AC({:.2f})_rec.pth.tar'.format(model_name, epoch, w_d, accu[-1])))
    else:
        torch.save(state,   os.path.join(save_path, '{}_oBN_EPC({})_wd({})_AC({:.2f}).pth.tar'.format(model_name, epoch, w_d, accu[-1])))
        torch.save(records, os.path.join(save_path, '{}_oBN_EPC({})_wd({})_AC({:.2f})_rec.pth.tar'.format(model_name, epoch, w_d, accu[-1])))

def plotCurve(path, rec_list, label_list, color_list, title):
    # plt.grid()
    # x = np.linspace(1, train_loss.shape[0], train_loss.shape[0])
    # plt.plot(x, train_loss, label='Training Loss', linewidth=2)
    # plt.plot(x, test_loss, label='Test Loss', linewidth=2)
    # plt.legend()
    # plt.xlabel('Epochs', fontdict={'size':12})
    # plt.ylabel('Loss', fontdict={'size':12})
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # max_idx = np.argmax(test_accu)
    # if args.bn:
    #     plt.savefig(os.path.join(args.cp_dir, '{0}_wBN_EPC({1})_wd({2})_AC({3:.2f})_LossFig.png'.format(args.model_name, args.epochs, args.weight_decay, test_accu[max_idx])))
    # else:
    #     plt.savefig(os.path.join(args.cp_dir, '{0}_oBN_EPC({1})_wd({2})_AC({3:.2f})_LossFig.png'.format(args.model_name, args.epochs, args.weight_decay, test_accu[max_idx])))
    # plt.close()
    plt.rcParams['figure.dpi'] = 300
    plt.grid()
    for i in range(len(rec_list)):
        rec = torch.load('./checkpoint/'+rec_list[i])['accuracy']
        x = np.linspace(1, rec.shape[0], rec.shape[0])
        max_idx = rec.shape[0]- 1 - np.argmax(rec[::-1])
        plt.plot(x, rec, label=label_list[i], color=color_list[i])
        plt.annotate('{0:.2f}'.format(rec[max_idx]), xy=(x[max_idx], rec[max_idx]), size=12, xytext=(x[max_idx], rec[max_idx]-0.32*(i+1)), 
                                                                                            arrowprops=dict(arrowstyle='->',
                                                                                            color=color_list[i]))
    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontdict={'size':12})
    plt.ylabel('Accuracy', fontdict={'size':12})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=18)
    plt.savefig(path)    
    plt.close()

if __name__ == '__main__':
    #### MLP—1 ####
    # path = './MLP1.png'
    # title = 'MLP_1'
    # rec_list = ['MLP_oBN_EPC(69)_wd(0)_AC(98.37)_rec.pth.tar','MLP_wBN_EPC(69)_wd(0)_AC(98.40)_rec.pth.tar',
    #             'MLP_oBN_EPC(69)_wd(1e-05)_AC(98.39)_rec.pth.tar', 'MLP_wBN_EPC(69)_wd(1e-05)_AC(98.42)_rec.pth.tar']
    #### MLP-2 ####
    # path = './MLP2.png'
    # title = 'MLP_2'
    # rec_list = ['MLP_2_oBN_EPC(79)_wd(0)_AC(98.45)_rec.pth.tar','MLP_2_wBN_EPC(79)_wd(0)_AC(98.58)_rec.pth.tar',
    #             'MLP_2_oBN_EPC(79)_wd(1e-05)_AC(98.51)_rec.pth.tar', 'MLP_2_wBN_EPC(79)_wd(1e-05)_AC(98.60)_rec.pth.tar']
    #### LeNet-5 ####
    # path = './LeNet5.png'
    # title = 'LeNet-5'
    # rec_list = ['LeNet_5_oBN_EPC(69)_wd(0)_AC(99.15)_rec.pth.tar','LeNet_5_wBN_EPC(69)_wd(0)_AC(99.25)_rec.pth.tar',
    #             'LeNet_5_oBN_EPC(69)_wd(1e-05)_AC(99.20)_rec.pth.tar', 'LeNet_5_wBN_EPC(69)_wd(1e-05)_AC(99.38)_rec.pth.tar']

    #### LeNet-5 w/o. pooling ####
    path = './LeNet5_woPooling.png'
    title = 'LeNet-5 w/o. Maxpooling'
    rec_list = ['LeNet_5_woPooling_oBN_EPC(69)_wd(0)_AC(99.13)_rec.pth.tar','LeNet_5_woPooling_wBN_EPC(69)_wd(0)_AC(99.33)_rec.pth.tar',
                'LeNet_5_woPooling_oBN_EPC(69)_wd(1e-05)_AC(99.10)_rec.pth.tar', 'LeNet_5_woPooling_wBN_EPC(69)_wd(1e-05)_AC(99.42)_rec.pth.tar']
    
    
    label_list = ['default', 'w.BN', 'w.WD', 'w.BN&WD']
    color_list = ['indianred', 'orange', 'mediumseagreen', 'deepskyblue']
    plotCurve(path, rec_list, label_list, color_list, title)
    