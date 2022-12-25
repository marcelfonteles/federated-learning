from torchvision import datasets, transforms

def get_dataset():
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    # sample training data amongst users
    # if args.iid:
    #     # Sample IID user data from Mnist
    #     user_groups = mnist_iid(train_dataset, args.num_users)
    # else:
    #     # Sample Non-IID user data from Mnist
    #     if args.unequal:
    #         # Chose uneuqal splits for every user
    #         user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
    #     else:
    #         # Chose euqal splits for every user
    #         user_groups = mnist_noniid(train_dataset, args.num_users)
    #
    # return train_dataset, test_dataset, user_groups