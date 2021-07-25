import time
import argparse

import torch
import numpy as np
import torch_geometric.nn as pyg_nn

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from matplotlib.pyplot as plt

from models.model import GNNStack
from models.utils import build_optimizer

def train(dataset, args):
    print("Node task. test set size:", np.sum(dataset[0]['train_mask'].numpy()))
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                     args)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_accs = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
            print(f"epoch: {epoch} train_loss: {total_loss:.3f} val_acc: {test_acc}")
        else:
            test_accs.append(test_accs[-1])
    return test_accs, losses


def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]
        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    return correct / total

def main():
    #Training settings
    parser = argparse.ArgumentParser(description='GCN Node classification')
    parser.add_argument('--model_type', type=str, default='GrapeSage', metavar='N',
                        help='input one of the GCN models(default:GraphSage')

    parser.add_argument('--dataset', type=str, default='cora', metavar='N',
                        help='input the name of dataset in PyG')

    parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                        help='input the number of layers to stack(default: 2)')

    parser.add_argument('--heads', type=int, default=2, metavar='N',
                        help='input the number of heads for GAT(default: 2)')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training(default: 32')

    parser.add_argument('--hidden_dim', type=int, default=32, metavar='N',
                        help='input hidden dimension in conv model(default: 32)')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='input probability for dropout(default: 0.5)')

    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='input the number of epochs')

    parser.add_argument('--opt', type=str, default='adm', metavar='N',
                        help='input optimizer(default: adm)')

    parser.add_argument('--opt_scheduler', type=str, default='step', metavar='N',
                        help='input optimizer scheduler(none, step, cos)')

    parser.add_argument('--opt_restart', type=int, default=0, metavar='N',)

    parser.add_argument('--weight_decay', type=float, default=5e-3, metavar='N',
                        help='input weight decay(default: 5e-3)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='input learning rate')

    parser.add_argument('--opt_decay_step', type=int, default=100, metavar='N',
                        help='input optimizer decay step(default: 100)')

    parser.add_argument('--opt_decay_rate', type=float, default=0.1, metavar='N',
                        help='input optimizer decay rate')

    args = parser.parse_args()

    # Match the dimension.
    if args.model_type == 'GraphSage':
      args.heads = 1

    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/cora', name='Cora')
    else:
        raise NotImplementedError("Unknown dataset")

    test_accs, losses = train(dataset, args)

    print("Maximum accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))

    plt.title(dataset.name)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()