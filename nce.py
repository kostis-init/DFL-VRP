import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    """
    Noise-contrastive estimation loss
    """
    def __init__(self, num_sampled, num_classes, dim):
        super(NCELoss, self).__init__()
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.dim = dim
        self.embedding = nn.Embedding(num_classes, dim)
        self.log_sigmoid = nn.LogSigmoid()
        self.sample_weights = torch.ones(num_classes)
        self.sample_weights[0] = 0  # set weight of the first class (PAD) to 0
        self.sample_weights /= self.sample_weights.sum()  # normalize weights

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]

        # get positive samples
        pos_emb = self.embedding(targets)
        pos_score = torch.bmm(inputs.view(batch_size, 1, self.dim), pos_emb.permute(0, 2, 1)).squeeze()
        pos_loss = self.log_sigmoid(pos_score).sum()

        # get negative samples
        neg_indices = torch.multinomial(self.sample_weights, batch_size * self.num_sampled, replacement=True)
        neg_emb = self.embedding(neg_indices).view(batch_size, self.num_sampled, self.dim)
        neg_score = torch.bmm(inputs.view(batch_size, 1, self.dim), neg_emb.permute(0, 2, 1)).squeeze()
        neg_loss = self.log_sigmoid(-neg_score).sum()

        # compute total loss
        loss = -(pos_loss + neg_loss) / batch_size
        return loss
