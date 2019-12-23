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

def plotCurve(args, train_loss, test_loss, test_accu):
    plt.grid()
    x = np.linspace(1, train_loss.shape[0], train_loss.shape[0])
    plt.plot(x, train_loss, label='Training Loss', linewidth=2)
    plt.plot(x, test_loss, label='Test Loss', linewidth=2)
    plt.legend()
    plt.xlabel('Epochs', fontdict={'size':12})
    plt.ylabel('Loss', fontdict={'size':12})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    max_idx = np.argmax(test_accu)
    if args.bn:
        plt.savefig(os.path.join(args.cp_dir, '{0}_wBN_EPC({1})_wd({2})_AC({3:.2f})_LossFig.png'.format(args.model_name, args.epochs, args.weight_decay, test_accu[max_idx])))
    else:
        plt.savefig(os.path.join(args.cp_dir, '{0}_oBN_EPC({1})_wd({2})_AC({3:.2f})_LossFig.png'.format(args.model_name, args.epochs, args.weight_decay, test_accu[max_idx])))
    plt.close()
    plt.grid()
    plt.plot(x, test_accu, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Epochs', fontdict={'size':12})
    plt.ylabel('Accuracy', fontdict={'size':12})
    plt.annotate('{0:.3f}'.format(test_accu[max_idx]), xy=(x[max_idx], test_accu[max_idx]), xytext=(x[max_idx], test_accu[max_idx]-1), arrowprops=dict(arrowstyle='->'))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if args.bn:
        plt.savefig(os.path.join(args.cp_dir, '{0}_wBN_EPC({1})_wd({2})_AC({3:.2f}).png'.format(args.model_name, args.epochs, args.weight_decay, test_accu[max_idx])))
    else:
        plt.savefig(os.path.join(args.cp_dir, '{0}_oBN_EPC({1})_wd({2})_AC({3:.2f}).png'.format(args.model_name, args.epochs, args.weight_decay, test_accu[max_idx])))    
    plt.close()

if __name__ == '__main__':
    pass
    