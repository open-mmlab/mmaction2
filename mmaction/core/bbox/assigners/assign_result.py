import torch


class AssignResult:
    """Store the assign results.

    Args:
        num_gts (int): Number of gt_bboxes in this image
        gt_inds (Tensor[long]): The assigned gt_inds to each proposal bbox, Set
            as 0 indicates that no gt bbox is assigned to the proposal bbox.
        max_overlaps (Tensor[float]): The maximum overlap of a proposal bbox
            to each gt_bboxes.
        labels (Tensor[float]): The assigned labels to each proposal bbox.
            Default: None.
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        # The index of the assigned gt bbox
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
