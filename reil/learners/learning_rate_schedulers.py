class LearningRateScheduler:
    def __init__(self, initial_lr: float) -> None:
        self.initial_lr = initial_lr

    def new_rate(self, epoch: int, current_lr: float) -> float:
        return self.initial_lr


ConstantLearningRate = LearningRateScheduler
