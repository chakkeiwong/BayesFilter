"""Host epoch control matching the pinned Younis bearings training source.

This is orchestration, not an array/numerical kernel. Source anchors:
src/trainers/trainer.py:613-699; trainer_utils.py:8-58,260-311.
Preserves the source's equality, negative-loss and rate-floor event behavior.
"""
from dataclasses import asdict, dataclass
import math


@dataclass
class SourcePlateau:
    lr: float
    threshold: float
    factor: float
    patience: int
    cooldown: int
    min_lr: float
    start_epoch: int = 0
    eps: float = 1e-8
    last_epoch: int = 0
    best: float | None = None
    num_bad_epochs: int = 0
    cooldown_counter: int = 0

    def step(self, value):
        self.last_epoch += 1
        if self.last_epoch < self.start_epoch:
            return False
        # Torch defaults: mode=min, threshold_mode=rel. In particular, do not
        # substitute an absolute decrease test when a validation NLL is negative.
        if self.best is None or value < self.best * (1 - self.threshold):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        event = self.num_bad_epochs > self.patience
        if event:
            proposed = max(self.lr * self.factor, self.min_lr)
            if self.lr - proposed > self.eps:
                self.lr = proposed
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        # The source reports an event even when eps/min_lr prevents a change.
        return event


@dataclass
class SourceEarlyStopping:
    patience: int
    start_offset: int
    min_delta: float = 0.0
    max_lr_change: int = 3
    counter: int = 0
    best_loss: float | None = None
    early_stop: bool = False
    lr_change_counter: int = 0

    def lr_changed(self):
        self.lr_change_counter += 1
        if self.lr_change_counter <= self.max_lr_change:
            self.counter = 0

    def step(self, value):
        if self.start_offset > 0:
            self.start_offset -= 1
            return
        if self.best_loss is None:
            self.best_loss = value
        elif self.best_loss - value > self.min_delta:
            self.best_loss, self.counter = value, 0
        elif self.best_loss - value < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class SourceEpochController:
    """One source scheduler per optimizer group, then source early stopping."""

    schema = "kdm_source_epoch_controller_v1"

    def __init__(self, learning_rates, scheduler, *, patience, start_offset):
        if not learning_rates:
            raise ValueError("At least one source optimizer group is required")
        self.schedulers = {}
        for name, rate in learning_rates.items():
            args = dict(scheduler, lr=float(rate))
            obj = SourcePlateau(**args)
            if (not math.isfinite(obj.lr) or obj.lr < 0
                    or not 0 < obj.factor < 1 or obj.min_lr < 0
                    or obj.patience < 0 or obj.cooldown < 0
                    or obj.start_epoch < 0 or obj.threshold < 0):
                raise ValueError("Invalid source scheduler configuration")
            self.schedulers[name] = obj
        if patience <= 0 or start_offset < 0:
            raise ValueError("Invalid early-stopping configuration")
        self.early = SourceEarlyStopping(patience=patience, start_offset=start_offset)
        self.epoch = -1
        self.best_validation_loss = None
        self.best_checkpoint_epoch = None

    @property
    def learning_rates(self):
        return {name: scheduler.lr for name, scheduler in self.schedulers.items()}

    def observe(self, validation_loss):
        value = float(validation_loss)
        if not math.isfinite(value):
            raise ValueError("Nonfinite validation loss; invalid training attempt")
        if self.early.early_stop:
            raise ValueError("Source controller has already stopped")
        self.epoch += 1
        improved = self.best_validation_loss is None or value < self.best_validation_loss
        if improved:
            self.best_validation_loss = value
        events = [scheduler.step(value) for scheduler in self.schedulers.values()]
        changed = any(events)
        if changed:
            self.early.lr_changed()
        self.early.step(value)
        save = improved and not self.early.early_stop
        if save:
            self.best_checkpoint_epoch = self.epoch
        return dict(epoch=self.epoch, validation_loss=value,
                    new_best_loss=improved, scheduler_event=changed,
                    learning_rates=self.learning_rates, stop=self.early.early_stop,
                    save_best=save, best_checkpoint_epoch=self.best_checkpoint_epoch)

    def state_dict(self):
        return dict(schema=self.schema, epoch=self.epoch,
                    best_validation_loss=self.best_validation_loss,
                    best_checkpoint_epoch=self.best_checkpoint_epoch,
                    schedulers={k: asdict(v) for k, v in self.schedulers.items()},
                    early=asdict(self.early))

    @classmethod
    def from_state_dict(cls, state):
        if state["schema"] != cls.schema:
            raise ValueError("Unsupported controller state")
        obj = cls.__new__(cls)
        obj.schedulers = {k: SourcePlateau(**v) for k, v in state["schedulers"].items()}
        obj.early = SourceEarlyStopping(**state["early"])
        obj.epoch = state["epoch"]
        obj.best_validation_loss = state["best_validation_loss"]
        obj.best_checkpoint_epoch = state["best_checkpoint_epoch"]
        return obj
