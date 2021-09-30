##Train Set + Val Set +Test Set

import torch
from torchvision import transforms, datasets

batch_size = 200
train_db = datasets.MNIST("../data", train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)

test_db = datasets.MNIST("..\data", train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

print("train:", len(train_db), "test:", len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])#分割
print("db_1:", len(train_db), "db_2:", len(val_db))
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_db,batch_size=batch_size, shuffle=True)
print("train_loader:", len(train_loader), "val_loader:", len(val_loader),"test_loader",len(test_loader))
