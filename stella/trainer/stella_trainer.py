from transformers import Trainer

from ..tuner.layer import StellaLayer
from ..tuner.model import StellaModel


class StellaTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self.optimizer.register_step_pre_hook(self._optim_step_pre_hook)
        self.optimizer.register_step_post_hook(self._optim_step_post_hook)

    def _optim_step_pre_hook(self, optimizer, args, kwargs):
        model: StellaModel = self.model
        model.pre_optimizer_step()

    def _optim_step_post_hook(self, optimizer, args, kwargs):
        model: StellaModel = self.model
        model.post_optimizer_step()

    def get_decay_parameter_names(self, model) -> list[str]:
        decay_parameters = super().get_decay_parameter_names(model)
        # exclude non-decay parameters in StellaLayer.non_decay_parameters
        decay_parameters = [
            name for name in decay_parameters if all(pname not in name for pname in StellaLayer.non_decay_param_names)
        ]
        return decay_parameters
