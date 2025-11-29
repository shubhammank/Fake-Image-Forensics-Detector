from torchvision import transforms
train_transforms=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
test_transforms=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])