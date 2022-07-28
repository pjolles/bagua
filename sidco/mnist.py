from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging
import bagua.torch_api as bagua


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MiniNet(nn.Module):
    def __init__(self):
        super(MiniNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1152, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.fuse_optimizer:
            optimizer.fuse_step()
        else:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader, losses=None, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    if losses is not None and epoch is not None:
        losses[epoch-1] = test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sidco",
        help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam, async, sidco",
    )
    parser.add_argument(
        "--async-sync-interval",
        default=500,
        type=int,
        help="Model synchronization interval(ms) for async algorithm",
    )
    parser.add_argument(
        "--set-deterministic",
        action="store_true",
        default=False,
        help="set deterministic or not",
    )
    parser.add_argument(
        "--fuse-optimizer",
        action="store_true",
        default=False,
        help="fuse optimizer or not",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="net",
        help="net or mininet",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="debug, info, warning, error",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="compression ratio for sidco algorithm",
    )

    args = parser.parse_args()
    if args.set_deterministic:
        print("set_deterministic: True")
        np.random.seed(666)
        random.seed(666)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666 + int(bagua.get_rank()))
        torch.set_printoptions(precision=10)

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    if args.log_level == "debug":
        log_level = logging.DEBUG
    elif args.log_level == "info":
        log_level = logging.INFO
    elif args.log_level == "warning":
        log_level = logging.WARNING
    elif args.log_level == "error":
        log_level = logging.ERROR

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(log_level)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if bagua.get_local_rank() == 0:
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )

    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset1, num_replicas=bagua.get_world_size(), rank=bagua.get_rank()
    )
    train_kwargs.update(
        {
            "sampler": train_sampler,
            "batch_size": args.batch_size // bagua.get_world_size(),
            "shuffle": False,
        }
    )
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.model == "net":
        model = Net().cuda()
    elif args.model == "mininet":
        model = MiniNet().cuda()
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.DecentralizedAlgorithm()
    elif args.algorithm == "low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
    elif args.algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.ByteGradAlgorithm()
    elif args.algorithm == "qadam":
        from bagua.torch_api.algorithms import q_adam

        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100
        )
        algorithm = q_adam.QAdamAlgorithm(optimizer)
    elif args.algorithm == "async":
        from bagua.torch_api.algorithms import async_model_average

        algorithm = async_model_average.AsyncModelAverageAlgorithm(
            sync_interval_ms=args.async_sync_interval,
        )
    elif args.algorithm == "sidco":
        from sidco import Sidco

        algorithm = Sidco(ratio=args.ratio)
    elif args.algorithm == "test":
        import algotest

        algorithm = algotest.TestAlgorithm()
    else:
        raise NotImplementedError

    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    losses = np.zeros(args.epochs)

    for epoch in range(1, args.epochs + 1):
        if args.algorithm == "async":
            model.bagua_algorithm.resume(model)

        train(args, model, train_loader, optimizer, epoch)

        if args.algorithm == "async":
            model.bagua_algorithm.abort(model)

        test(model, test_loader, losses, epoch)
        scheduler.step()

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("algorithm: " + args.algorithm + ", ratio (only if applicable): " + str(args.ratio))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xticks(np.arange(0, args.epochs))
    ax.set_ylim(0)
    ax.grid(True)

    plt.savefig("plot.png")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
