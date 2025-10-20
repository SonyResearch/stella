from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose

from .config import StellaConfig


class DiagonalLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return x * self.weight


class StellaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        'stella_U',
        'stella_S',
        'stella_Vt',
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        'r',
        'lora_alpha',
        'scaling',
        'lora_dropout',
    )
    # All names of parameters that should not be decayed
    non_decay_param_names = (
        'stella_U',
        'stella_Vt',
    )

    def __init__(
        self,
        base_layer: nn.Module,
        diag_s=False,
        **kwargs,
    ) -> None:
        self.base_layer = base_layer
        self.diag_s = diag_s
        self.kwargs = kwargs

        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.stella_U: Mapping[str, nn.Linear] = nn.ModuleDict({})
        self.stella_S: Mapping[str, nn.Linear] = nn.ModuleDict({})
        self.stella_Vt: Mapping[str, nn.Linear] = nn.ModuleDict({})
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, 'ds_shape') else base_layer.weight.shape
            )
        elif hasattr(base_layer, 'infeatures') and hasattr(base_layer, 'outfeatures'):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, 'input_size') and hasattr(base_layer, 'output_size'):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, 'codebooks') and base_layer.__class__.__name__ == 'QuantizedLinear':
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, 'w_bit') and base_layer.__class__.__name__ == 'WQLinear_GEMM':
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == 'EetqLinear':
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, 'W_q') and base_layer.__class__.__name__ == 'HQQLinear':
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, 'in_features') and hasattr(base_layer, 'out_features'):
                in_features, out_features = (
                    base_layer.in_features,
                    base_layer.out_features,
                )
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.",
                UserWarning,
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.stella_Vt[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        if self.diag_s:
            self.stella_S[adapter_name] = DiagonalLinear(
                r,
            )
        else:
            self.stella_S[adapter_name] = nn.Linear(r, r, bias=False)
        self.stella_U[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # some inits requires access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        with gather_params_ctx(self.get_base_layer().weight):
            self.reset_stella_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_stella_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        elif init_lora_weights is True:
            init_lora_weights = 'rando'

        if adapter_name in self.stella_Vt.keys():
            U = self.stella_U[adapter_name].weight
            S = self.stella_S[adapter_name].weight
            Vt = self.stella_Vt[adapter_name].weight

        if init_lora_weights.startswith('svd'):
            mode = init_lora_weights.split('_')[-1]
            self.init_svd(U, S, Vt, mode=mode)
        elif init_lora_weights == 'rando':
            self.init_rando(U, S, Vt)
        else:
            raise ValueError(f'Unknown initialization method {init_lora_weights}')

    @torch.no_grad()
    def init_rando(self, U: Tensor, S: Tensor, Vt: Tensor):
        nn.init.orthogonal_(U)
        nn.init.orthogonal_(Vt)
        if not self.diag_s:
            nn.init.zeros_(S)
            r = min(*S.shape)
            S.data[:r, :r] = torch.eye(r).to(S.device)

    @torch.no_grad()
    def init_svd(self, U: Tensor, S: Tensor, Vt: Tensor, mode: str = 'major'):
        # TODO: support pissa init
        # TODO: try pissa-like online merging
        print(f'init_svd: {mode}')
        if self.diag_s:
            r = r2 = S.size(0)
        else:
            r, r2 = S.shape
        dtype = U.dtype
        weight: Tensor = self.get_base_layer().weight
        weight = weight.cpu().float()
        weight = transpose(weight, self.fan_in_fan_out)
        L, _, Rt = torch.linalg.svd(weight, full_matrices=False)
        if mode == 'major':
            U.data = L[:, :r].to(dtype=dtype, device=U.device)
            Vt.data = Rt[:r2, :].to(dtype=dtype, device=Vt.device)
        elif mode == 'minor':
            U.data = L[:, -r:].to(dtype=dtype, device=U.device)
            Vt.data = Rt[-r2:, :].to(dtype=dtype, device=Vt.device)
        else:
            raise ValueError(f'Unknown mode {mode}')
        if not self.diag_s:
            nn.init.zeros_(S)
            r = min(*S.shape)
            S.data[:r, :r] = torch.eye(r).to(S.device)

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.stella_Vt.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.stella_Vt.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of
        the model."""
        adapter_names = kwargs.get('adapter_names', None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                'Length of `adapter_names` should be the same as the number of inputs, but got '
                f'{len(adapter_names)} and {len(x)} respectively.'
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = 'Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first.'
            raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == '__base__':
                continue
            if active_adapter not in self.stella_Vt.keys():
                continue

            stella_Vt = self.stella_Vt[active_adapter]
            stella_S = self.stella_S[active_adapter]
            stella_U = self.stella_U[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to Stella layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(stella_Vt.weight.dtype)
            stella_output = stella_U(stella_S(stella_Vt(dropout(sub_batch)))) * scaling
            result[sub_batch_indices_list[i]] += stella_output.to(torch_result_dtype)

        return result


class Linear(nn.Module, StellaLayer):
    # Stella implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        StellaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Merge the active adapter weights into the base weights.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.stella_Vt.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken'
                        )

                    base_layer.weight.data = orig_weights

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """This method unmerges all merged adapter layers from the base
        weights."""
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.stella_Vt.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.stella_U[adapter].weight.device
        dtype = self.stella_U[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == 'cpu' and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_Vt = self.stella_Vt[adapter].weight
        weight_S = self.stella_S[adapter].weight
        if self.diag_s:
            weight_S = weight_S.diag()
        weight_U = self.stella_U[adapter].weight

        if cast_to_fp32:
            weight_Vt = weight_Vt.float()
            weight_S = weight_S.float()
            weight_U = weight_U.float()

        output_tensor = transpose(weight_U @ weight_S @ weight_Vt, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop('adapter_names', None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.stella_Vt.keys():
                    continue
                stella_Vt = self.stella_Vt[active_adapter]
                stella_S = self.stella_S[active_adapter]
                stella_U = self.stella_U[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(stella_Vt.weight.dtype)

                result = result + stella_U(stella_S(stella_Vt(dropout(x)))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return 'stella.' + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    stella_config: StellaConfig,
    **kwargs,
) -> torch.nn.Module | None:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs['fan_in_fan_out']:
            warnings.warn(
                'fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. '
                'Setting fan_in_fan_out to False.'
            )
            kwargs['fan_in_fan_out'] = stella_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs['fan_in_fan_out']:
            warnings.warn(
                'fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.'
            )
            kwargs['fan_in_fan_out'] = stella_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
