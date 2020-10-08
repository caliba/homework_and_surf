from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import argparse
import torch.backends.cudnn as cudnn
import numpy
def train_model(model, best_acc,start_epoch,criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:#,'test']:#, 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            '''
            if phase == 'test' and epoch_acc > best_acc:
                print('Saving..')
                state = {
                    'model_ft': model.module ,#if use_cuda else net,
                    'acc': epoch_acc,
                    'epoch': epoch,
                }
            '''
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            state = {
                    'model_ft': model.module ,#if use_cuda else net,
                    'acc': epoch_acc,
                    'epoch': epoch,
                }
            if epoch%10 ==0 :
                print('Saving  every 10 epochs..')
                torch.save(state, './checkpoint/{}_ckpt.t7'.format(epoch))
            print('Saving..')
            torch.save(state, './checkpoint/ckpt.t7')
			

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='PyTorch car images Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')      
    parser.add_argument('--resume_e', default='', type=str, help='resumef rom checkpoint ')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--test','-t', action='store_true', help='test')
    args = parser.parse_args()
    acc = 0.0
    startepoch = 0 
    num_class=96
    #data processing 
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(-30, 30),#旋转不定角度
            transforms.Resize(800),
            transforms.RandomCrop(640),#随机裁剪640大小
            transforms.RandomAffine(15),#放射变化
            #transforms.CenterCrop(640),
            torchvision.transforms.RandomGrayscale(0.05),#转灰度图
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),#调节亮度，透明度 对比度
            transforms.RandomHorizontalFlip(0.7),#水平翻转概率
            transforms.RandomVerticalFlip(0.5),#垂直翻转概率
            #
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


        ])

        ,

		'test': transforms.Compose([
            transforms.Resize(800),
            transforms.RandomRotation(-30, 30),  # 旋转不定角度
            transforms.Resize(800),
            transforms.RandomCrop(640),  # 随机裁剪640大小
            transforms.RandomAffine(15),  # 放射变化
            # transforms.CenterCrop(640),
            torchvision.transforms.RandomGrayscale(0.05),  # 转灰度图
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 调节亮度，透明度 对比度
            transforms.RandomHorizontalFlip(0.6),  # 水平翻转概率
            transforms.RandomVerticalFlip(0.5),  # 垂直翻转概率
            #transforms.CenterCrop(640),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.RandomCrop(640,480),
            transforms.RandomRotation(-30, 30),  # 旋转不定角度
            transforms.Resize(800),
            transforms.RandomCrop(640,480),  # 随机裁剪640大小
            transforms.RandomAffine(15),  # 放射变化
            # transforms.CenterCrop(640),
            torchvision.transforms.RandomGrayscale(0.05),  # 转灰度图
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 调节亮度，透明度 对比度
            transforms.RandomHorizontalFlip(0.6),  # 水平翻转概率
            transforms.RandomVerticalFlip(0.5),  # 垂直翻转概率
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    data_dir = 'E:/ai_match/cloud_classification_all'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}#,'test']}#, 'val']}
    image_datasets['test']=datasets.ImageFolder('E:/ai_match/Test', data_transforms['test'])
    classes = [d for d in os.listdir('E:/ai_match/cloud_classification_all/train') if
               os.path.isdir(os.path.join('E:/ai_match/cloud_classification_all/train', d))]
    classes.sort()
    # mapping between id and classes
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class ={ i:classes[i] for i in range(len(classes))}
    print("class to id:{}".format(class_to_idx))
    print("id to class:{}".format(idx_to_class))
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=6,
                                                 shuffle=True,
                                                 num_workers=2) for x in ['train']}#, 'test']}#, 'val']}
    dataloders['test']=torch.utils.data.DataLoader(image_datasets['test'],
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1) 
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}#, 'val']}
    print('Test dataset size : {}'.format(dataset_sizes['test']))
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    if args.resume:
    # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        model_ft = checkpoint['model_ft']
        acc = checkpoint['acc']
        print('best acc :{}'.format(acc))
        startepoch = 0#checkpoint['epoch']
    elif args.resume_e:
        print('==> Resuming from checkpoint {}..'.format(args.resume_e))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume_e)
        model_ft = checkpoint['model_ft']
        acc = checkpoint['acc']
        print('best acc :{}'.format(acc))
        startepoch = 0#checkpoint['epoch']
    else:
        print('==> Building model..')
    # get model and replace the original fc layer with your fc layer
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_class)

    if use_gpu:
        model_ft = model_ft.cuda()
        model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))
		# speed up slightly
        cudnn.benchmark = True

    # define loss function  定义损失函数
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    # define lr
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    if args.test:
        data_gen = enumerate(dataloders['test'])
        output=[]
        import time
        import numpy as np
        proc_start_time = time.time()
        for i, data in data_gen:
    
            input_var = torch.autograd.Variable(data[0])
            # compute output
            rst= model_ft(input_var)
            rst=rst.detach().cpu().numpy().copy()
            # measure accuracy and record loss  测量精度
            end = time.time()
            output.append(rst.reshape(1,num_class))
            cnt_time = time.time() - proc_start_time
            if i%500 ==0:
                print('image {} done, total {}/{}, average {} sec/image'.format(i, i+1,
                                                                    dataset_sizes['test'],
                                                                    float(cnt_time) / (i+1)))
        image_pred = [np.argmax(x[0]) for x in output]
        print('pred nums : {}'.format(len(image_pred)))
        files=os.listdir('/share2/cloud_classification/Test/Test' )
        files.sort()
        print('test files nums : {}'.format(len(files)))
        #record output for test  精度记录
        if os.path.exists('/share2/cloud_classification/final_results_0.csv'):
            os.remove('/share2/cloud_classification/final_results_0.csv')
        f = open('/share2/cloud_classification/final_results_0.csv','a')
        f.write('FileName,type')
        for i in range(len(files)):
            #f.write('\n'+'{}'.format(files[i])+','+'{}'.format(idx_to_class[image_pred[i]]))
            f.write('\n'+'{}'.format(files[i]))
            f.write(','+'{}'.format(idx_to_class[image_pred[i]]))
        f.close()
        print('results written succeed')
    else:
        #start training 开始训练
        model_ft = train_model(model=model_ft,best_acc=acc,start_epoch=startepoch,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=100)