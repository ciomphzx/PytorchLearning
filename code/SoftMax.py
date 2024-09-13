import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 下载数据集
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../dataset/FashionMNIST/train', train=True, 
    transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../dataset/FashionMNIST/test', train=False,
    transform=trans, download=True
)
print('train set num: ', len(mnist_train))
print('test set num: ', len(mnist_test))
print('image size is ', mnist_train[0][0].shape)

#读取图像标签
def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                   'coat',  'sandal', 'shirt', 'sneaker',
                   'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 可视化样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_cols * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=20)))
show_images(X.reshape(20, 28, 28), 2, 10, titles=get_labels(y))


        
