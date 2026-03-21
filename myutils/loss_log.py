

class LossLog:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_dict = {}

    def update(self, loss_dict):
        for loss_name, loss in loss_dict.items():
            if loss_name not in self.loss_dict:
                self.loss_dict[loss_name] = {"loss_sum": 0.0, "count": 0}
            self.loss_dict[loss_name]["loss_sum"] += loss
            self.loss_dict[loss_name]["count"] += 1

    def average(self, loss_name):
        if loss_name not in self.loss_dict or self.loss_dict[loss_name]["count"] == 0:
            return 0.0
        return self.loss_dict[loss_name]["loss_sum"] / self.loss_dict[loss_name]["count"]

    def __str__(self):
        res = " ".join(["%s: %.3f"%(name,self.average(name)) for name in self.loss_dict])
        return res




