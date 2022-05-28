import torch

__all__ = ['MeterS3DIS']


class MeterS3DIS:
    def __init__(self, metric='iou', num_classes=13):
        super().__init__()
        assert metric in ['overall', 'class', 'iou']
        self.metric = metric
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_seen = [0] * self.num_classes
        self.total_correct = [0] * self.num_classes
        self.total_positive = [0] * self.num_classes
        self.total_seen_num = 0
        self.total_correct_num = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, logits=True):
        # outputs: B x 13 x num_points, targets: B x num_points
        if logits:
            predictions = outputs.argmax(1)
        else:
            predictions = outputs
        if self.metric == 'overall':
            self.total_seen_num += targets.numel()
            self.total_correct_num += torch.sum(targets == predictions).item()
        else:
            # self.metric == 'class' or self.metric == 'iou':
            for i in range(self.num_classes):
                itargets = (targets == i)
                ipredictions = (predictions == i)
                self.total_seen[i] += torch.sum(itargets).item()
                self.total_positive[i] += torch.sum(ipredictions).item()
                self.total_correct[i] += torch.sum(itargets & ipredictions).item()

    def compute(self,perclass=True,map=None):
        if map is None:
            map = {i:i for i in range(self.num_classes)}
        if self.metric == 'class':
            accuracy = 0
            for i in range(self.num_classes):
                total_seen = self.total_seen[i]
                if total_seen == 0:
                    accuracy += 1
                else:
                    accuracy += self.total_correct[i] / total_seen
            return accuracy / self.num_classes
        elif self.metric == 'iou':
            iou = 0
            for i in range(self.num_classes):
                total_seen = self.total_seen[i]
                if total_seen == 0:
                    iou += 1
                    if perclass:
                         print(f'class {map[i]} the iou is 1')
                else:
                    total_correct = self.total_correct[i]
                    classiou = total_correct / (total_seen + self.total_positive[i] - total_correct)
                    iou+=classiou
                    if(perclass):
                         print(f'class {map[i]} the iou is {classiou}')
            return iou / self.num_classes
        else:
            return self.total_correct_num / self.total_seen_num
