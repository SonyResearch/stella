from __future__ import annotations

import math
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from functools import partial

import torch
from torch import nn
from tqdm import tqdm

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)
from peft.utils.other import get_pattern_key

from .config import StellaConfig
from .layer import StellaLayer, dispatch_default
from .stiefel import euclidean2riemannian, exp_map, polar_retraction, tangent_project


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs['adapter_names'] = adapter_names
    return args, kwargs


class StellaModel(BaseTuner):
    """Creates Riemannian Low Rank Adapter (Stella) model from a pretrained
    transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`StellaConfig`]): The configuration of the Stella model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Lora model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`StellaConfig`]): The configuration of the Lora model.
    """

    prefix: str = 'stella_'

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _check_new_adapter_config(self, config: StellaConfig) -> None:
        """A helper method to check the config when a new adapter is being
        added.

        Raise a ValueError if there is something wrong with the config or if it
        conflicts with existing adapters.
        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != 'none'):
            raise ValueError(
                f'{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, '
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(stella_config, key):
        return check_target_module_exists(stella_config, key)

    def _prepare_model(self, peft_config: StellaConfig, model: nn.Module):
        r"""A private method to modify the model structure before adapter is
        applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

    def _create_and_replace(
        self,
        stella_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(stella_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(stella_config.alpha_pattern.keys(), current_key)
        r = stella_config.rank_pattern.get(r_key, stella_config.r)
        alpha = stella_config.alpha_pattern.get(alpha_key, stella_config.lora_alpha)

        kwargs = {
            'r': r,
            'lora_alpha': alpha,
            'lora_dropout': stella_config.lora_dropout,
            'diag_s': stella_config.stella_diag_s,
            'fan_in_fan_out': stella_config.fan_in_fan_out,
            'init_lora_weights': stella_config.init_lora_weights,
            'use_rslora': stella_config.use_rslora,
        }

        if isinstance(target, StellaLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=stella_config.lora_dropout,
                init_lora_weights=stella_config.init_lora_weights,
                use_rslora=stella_config.use_rslora,
            )
        else:
            new_module = self._create_new_module(stella_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, 'base_layer'):
            child = child.base_layer

        if not hasattr(new_module, 'base_layer'):
            if hasattr(new_module, 'W_q'):  # HQQ
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, 'bias'):
                new_module.bias = child.bias

        if getattr(child, 'state', None) is not None:
            if hasattr(new_module, 'base_layer'):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device('meta')
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ('ranknum' in name):
                weight = (
                    child.qweight
                    if hasattr(child, 'qweight')
                    else child.W_q
                    if hasattr(child, 'W_q')
                    else child.weight
                    if hasattr(child, 'weight')
                    else next(child.parameters())
                )
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == 'none':
                continue

            if bias == 'all':
                for n, p in model.named_parameters():
                    if 'bias' in n:
                        p.requires_grad = True
            elif bias == 'lora_only':
                for m in model.modules():
                    if isinstance(m, StellaLayer) and hasattr(m, 'bias') and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f'Requested bias: {bias}, is not implemented.')

    @staticmethod
    def _create_new_module(stella_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced Stella layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        if stella_config._custom_modules:
            # Experimental custom Stella module support. Allows users to pass a custom mapping for unsupported layer
            # types by implementing their own Stella layers.
            def dynamic_dispatch_func(target, adapter_name, stella_config, **kwargs):
                new_module = None

                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                for key, custom_cls in stella_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        dispatchers.append(dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, stella_config=stella_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f'Target module {target} is not supported. Currently, only the following modules are supported: '
                '`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, '
                '`transformers.pytorch_utils.Conv1D`.'
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == 'model':  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config['inference_mode'] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-
        enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output
        of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != 'none':
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    'output as the the base model would without adaption.'
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'stella' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, StellaLayer):
                if module.merged:
                    warnings.warn('Adapter cannot be set when the model is merged. Unmerging the model first.')
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop('adapter_names', None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError('Cannot pass `adapter_names` when the model is in training mode.')

        # Check that users only passed actually existing adapters.
        # Note: We cannot do this on the layer level, as each individual layer may not have each adapter. Still, we want
        # to check that there is at least one layer with the given name, or else something like typos can easily slip.
        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, StellaLayer):
                expected_adapters |= layer.stella_Vt.keys()
                expected_adapters |= layer.stella_embedding_Vt.keys()
        unique_adapters = {name for name in adapter_names if name != '__base__'}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f'Trying to infer with non-existing adapter(s): {", ".join(sorted(unexpected_adapters))}')

        hook_handles = []
        for module in self.modules():
            if isinstance(module, StellaLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support
        merging.
        """
        super()._check_merge_allowed()
        if getattr(self.model, 'quantization_method', None) == 'gptq':
            raise ValueError('Cannot merge Stella layers when the model is gptq quantized')
        if self.peft_config.get('layer_replication'):
            raise ValueError('Cannot merge Stella layers when base model layers are replicated')

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config['model_type'] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError('Please specify `target_modules` in `peft_config`')
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config['model_type']]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ):
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = 'Unloading ' + ('and merging ' if merge else '') + 'model'
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, 'base_layer'):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, 'base_layer'):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        """Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f'Adapter {adapter_name} does not exist')
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, StellaLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> torch.nn.Module:
        r"""This method merges the Stella layers into the base model. This is
        needed if someone wants to use the base model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """Gets back the base model by removing all the stella modules without
        merging.

        This gives back the original base model.
        """
        return self._unload_and_optionally_merge(merge=False)

    @torch.no_grad()
    def pre_optimizer_step(self):
        """This method should be called before the optimizer step."""
        self.params = defaultdict(list)
        grads = defaultdict(list)
        assert len(self.active_adapters) == 1
        active_adapter = self.active_adapters[0]
        for key in self.targeted_module_names:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, StellaLayer):
                U = target.stella_U[active_adapter].weight
                Vt = target.stella_Vt[active_adapter].weight
                if U.grad is not None:
                    self.params[U.shape].append(U)
                    grads[U.shape].append(U.grad)
                if Vt.grad is not None:
                    self.params[Vt.t().shape].append(Vt)
                    grads[Vt.t().shape].append(Vt.grad.t())

        self.cache = defaultdict(list)
        for k, ps in self.params.items():
            ps_reshape = [p if p.size(0) > p.size(1) else p.t() for p in ps]
            x = torch.stack(ps_reshape, dim=0)
            self.cache[k] = x.detach()
            g = torch.stack(grads[k], dim=0)
            # egrad2rgrad
            g_riemannian = euclidean2riemannian(x, g)
            for p, g_r in zip(ps, g_riemannian):
                if p.size(0) < p.size(1):
                    g_r = g_r.t()
                p.grad.copy_(g_r)

    @torch.no_grad()
    def post_optimizer_step(self):
        r"""This method should be called after the optimizer step."""
        retraction_type = self.peft_config[self.active_adapters[0]].stella_retraction
        if retraction_type == 'polar':
            retraction = polar_retraction
        elif retraction_type == 'exp_map':
            retraction = exp_map
        else:
            raise ValueError(f'Unknown retraction method: {retraction}')

        grad_scaling = self.peft_config[self.active_adapters[0]].stella_grad_scaling
        if isinstance(grad_scaling, bool) and grad_scaling:
            c = self.config.hidden_size
        elif isinstance(grad_scaling, float):
            c = grad_scaling
        else:
            c = None

        for k, pre_ps in self.cache.items():
            grad = torch.stack([p if p.size(0) > p.size(1) else p.t() for p in self.params[k]], dim=0) - pre_ps
            m = grad.size(1)
            if c is not None and m != c:
                # gradient scaling
                grad.div_(math.sqrt(m / c))
            # project to tangent space
            grad = tangent_project(pre_ps, grad)
            # retraction
            pre_ps = retraction(pre_ps, grad)
            for p, xo in zip(self.params[k], pre_ps):
                if p.size(0) < p.size(1):
                    xo = xo.t()
                p.copy_(xo)

        self.params = None
        self.cache = None
