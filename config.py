# Global config variables
_cifar_labels = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
log_every = 50

dataset = 'cifar10'
if dataset == 'cifar10':
    dataset_labels= _cifar_labels

# Victim Params
model_type = 'vit' # Options: 'simple_cnn' 'vit' 'vgg16'
pretrained = True
pretrained_path = 'victim_model.pt'
epochs = 1
learning_rate = 0.01
batch_size = 32


dip_trace_length = 600
t = 100
beta = 0.5
similar_threshold = 0.5

