import torch


class OHEMHingeLoss(torch.autograd.Function):
    """This class is the core implementation for the completeness loss in
    paper.

    It compute class-wise hinge loss and performs online hard example mining
    (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        """Calculate OHEM hinge loss.

        Args:
            pred (torch.Tensor): Predicted completeness score.
            labels (torch.Tensor): Groundtruth class label.
            is_positive (int): Set to 1 when proposals are positive and
                set to -1 when proposals are incomplete.
            ohem_ratio (float): Ratio of hard examples.
            group_size (int): Number of proposals sampled per video.

        Returns:
            torch.Tensor: Returned class-wise hinge loss.
        """
        num_samples = pred.size(0)
        if num_samples != len(labels):
            raise ValueError(f'Number of samples should be equal to that '
                             f'of labels, but got {num_samples} samples and '
                             f'{len(labels)} labels.')

        losses = torch.zeros(num_samples, device=pred.device)
        slopes = torch.zeros(num_samples, device=pred.device)
        for i in range(num_samples):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0

        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_length = int(group_size * ohem_ratio)
        loss = torch.zeros(1, device=pred.device)
        for i in range(losses.size(0)):
            loss += sorted_losses[i, :keep_length].sum()
        ctx.loss_index = indices[:, :keep_length]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_groups = losses.size(0)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        labels = ctx.labels
        slopes = ctx.slopes

        grad_in = torch.zeros(ctx.shape, device=ctx.slopes.device)
        for group in range(ctx.num_groups):
            for idx in ctx.loss_index[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = (
                    slopes[loc] * grad_output.data[0])
        return torch.autograd.Variable(grad_in), None, None, None, None
