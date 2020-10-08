import os
import shutil
opath =
tar_path =

f = open(, 'r')  # 由于我使用的pycharm已经设置完了路径，因此我直接写了文件名
for lines in f:
    ls = lines.strip('\n').strip('\ufeff').replace(' ', '').split(',')
    print(ls[0])
    # if not ';' in ls[1]:
    if '1' in '11':
        if not os.path.exists(tar_path + 'train' + '/' + ls[1]):
            os.makedirs(tar_path + 'train' + '/' + ls[1])
        # os.system("cp G:/Pycharm/AI_test2/cloud_classification/Train/ ./rename_test")
        shutil.copy(opath + '/Train/' + ls[0], tar_path + 'train' + '/' + ls[1])
f.close()