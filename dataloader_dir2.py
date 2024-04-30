import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import random
import torch
import csv
from tqdm import tqdm
def label_make(class_num,label):
    label = [ 1 if (i == label) else 0 for i in range(class_num) ]
    return torch.Tensor(label)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset)])
    print(mean_)
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset)])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return [(mean_r, mean_g, mean_b), (std_r, std_g, std_b)]

def get_diagnosis_from_id(csv_file, id_code):
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['id_code'] == id_code:
                return row['diagnosis']
    return None

# 시드 설정
seed = 42
set_seed(seed)
class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ImageFolder(data.Dataset):
    def __init__(self, Image_path,Label_path,img_size,Normalize=None,aug_triger=False):
        self.Image_path = Image_path
        self.filelist = os.listdir(Image_path)
        self.img_size = img_size
        self.Label_path = Label_path
        self.img_Normalize = Normalize
        self.aug = aug_triger
    def __getitem__(self, index):
        filename = self.filelist[index]
        IMAGE1 = Image.open(os.path.join(self.Image_path , filename)).convert('RGB')

        Image_Transform = []
        Image_Transform.append(T.Resize((self.img_size[0], self.img_size[1])))
        Image_Transform.append(T.ToTensor())
        
        if( self.aug ):
            select_num = random.randint(0,4)
            if select_num == 0:
                pass
            elif select_num == 1:
                Image_Transform.append(T.RandomHorizontalFlip())
            elif select_num == 2:
                Image_Transform.append(AddGaussianNoise(0, 1))
            elif select_num == 3:
                Image_Transform.append(T.RandomRotation([-8, 8]))
            elif select_num == 4:
                Image_Transform.append(T.RandomVerticalFlip())

        Image_Transform = T.Compose(Image_Transform)
        
        IMAGE1 = Image_Transform(IMAGE1)


        Image_Transform = []

        if( self.img_Normalize == None ):
            # Image_Transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            pass
        else:
            Image_Transform.append(T.Normalize(self.img_Normalize[0], self.img_Normalize[1]))
        Image_Transform = T.Compose(Image_Transform)
        IMAGE = Image_Transform(IMAGE1)
        label = None
        file_name = filename.split('.')[0]
        diagnosis = get_diagnosis_from_id(self.Label_path, file_name)
        label = label_make(5,int(diagnosis))

        

        return IMAGE, label
    def __len__(self):
        """Returns the total number of font files."""
        return len(self.filelist)
    

def get_loder_main(train_path,valid_path,test_path,Label_train_path,Label_valid_path,Label_test_path,img_size, batch_size, shuffle,numworkers):
    train_dataset = ImageFolder(train_path,Label_train_path,img_size,None,False)#2를 사용하는 이유는 aug가 정의되어 있지 않기 때문임.
    vaild_dataset = ImageFolder(valid_path,Label_valid_path,img_size,None,False)
    test_dataset = ImageFolder(test_path,Label_test_path,img_size,None,False)
    
    Nomalize = calculate_norm(train_dataset+vaild_dataset+test_dataset)

    train_dataset = ImageFolder(train_path,Label_train_path,img_size,Nomalize,True)
    vaild_dataset = ImageFolder(valid_path,Label_valid_path,img_size,Nomalize,False)
    test_dataset = ImageFolder(test_path,Label_test_path,img_size,Nomalize,False)

    train_dataset_loader = data.DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory = True,
                            num_workers=numworkers)
    
    vaild_dataset_loader = data.DataLoader(dataset=vaild_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory = True,
                            num_workers=numworkers)
    
    test_dataset_loader = data.DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory = True,
                            num_workers=numworkers)
    

    return train_dataset_loader, vaild_dataset_loader, test_dataset_loader