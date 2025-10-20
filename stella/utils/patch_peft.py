import os
import warnings

import torch

import peft
from peft.utils.other import EMBEDDING_LAYER_NAMES, check_file_exists_on_hf_hub
from peft.utils.save_and_load import (
    get_embedding_layer_name,
    has_valid_embedding_base_layer,
)

from .extend_enum import EnumBase, extend_enum


# monkey patching to add STELLA to the PeftType enum
@extend_enum(peft.utils.peft_types.PeftType)
class PeftType(EnumBase):
    STELLA = 'STELLA'


def patch_peft_type():
    peft.utils.peft_types.PeftType = PeftType
    peft.utils.PeftType = PeftType
    peft.PeftType = PeftType


def patch_peft_constants():
    peft.utils.constants.PEFT_TYPE_TO_PREFIX_MAPPING[PeftType.STELLA] = 'stella_'


def get_peft_model_state_dict(
    model, state_dict=None, adapter_name='default', unwrap_compiled=False, save_embedding_layers='auto'
):
    """Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, '_orig_mod', model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    # TUNER SPECIFIC CODE
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'lora_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('lora_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (('lora_' in k and adapter_name in k) or ('bias' in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f'.{adapter_name}', ''): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

        if config.use_dora:
            # Here we take care of a refactor of DoRA which changed lora_magnitude_vector from a ParameterDict to a
            # ModuleDict with a DoraLayer instance. The old parameter is now the "weight" attribute of that layer. Since
            # we want the state_dict format not to change, we remove the "weight" part.
            new_dora_suffix = f'lora_magnitude_vector.{adapter_name}.weight'

            def renamed_dora_weights(k):
                if k.endswith(new_dora_suffix):
                    k = k[:-7]  # remove ".weight"
                return k

            to_return = {renamed_dora_weights(k): v for k, v in to_return.items()}

    elif config.peft_type == PeftType.STELLA:
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'stella_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'stella_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'stella_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('stella_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {
            k: v.contiguous()
            for k, v in to_return.items()
            if (('stella_' in k and adapter_name in k) or ('bias' in k))
        }

    elif config.peft_type == PeftType.BOFT:
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'boft_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'boft_' in k or 'bias' in k}
        elif bias == 'boft_only':
            to_return = {}
            for k in state_dict:
                if 'boft_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('boft_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if 'hada_' in k}

    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if 'lokr_' in k}

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split('.')[-1].startswith('adaption_')}

    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return['prefix_task_cols'] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return['prefix_task_rows'] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            if config.inference_mode:
                prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
            else:
                prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return['prompt_embeddings'] = prompt_embeddings

    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if 'ia3_' in k}

    elif config.peft_type == PeftType.OFT:
        to_return = {k: state_dict[k] for k in state_dict if 'oft_' in k}

    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if 'poly_' in k}

    elif config.peft_type == PeftType.LN_TUNING:
        to_return = {k: state_dict[k] for k in state_dict if 'ln_tuning_' in k}

    elif config.peft_type == PeftType.VERA:
        to_return = {k: state_dict[k] for k in state_dict if 'vera_lambda_' in k}
        if config.save_projection:
            # TODO: adding vera_A and vera_B to `self.get_base_layer` would
            # make name to match here difficult to predict.
            if f'base_model.vera_A.{adapter_name}' not in state_dict:
                raise ValueError(
                    'Model was initialised to not save vera_A and vera_B but config now specifies to save projection!'
                    ' Set `config.save_projection` to `False`.'
                )
            to_return['base_model.vera_A.' + adapter_name] = state_dict['base_model.vera_A.' + adapter_name]
            to_return['base_model.vera_B.' + adapter_name] = state_dict['base_model.vera_B.' + adapter_name]
    elif config.peft_type == PeftType.FOURIERFT:
        to_return = {k: state_dict[k] for k in state_dict if 'fourierft_' in k}
    elif config.peft_type == PeftType.XLORA:
        to_return = {k: state_dict[k] for k in state_dict if 'internal_xlora_classifier' in k}
    elif config.peft_type == PeftType.HRA:
        to_return = {k: state_dict[k] for k in state_dict if 'hra_' in k}
    elif config.peft_type == PeftType.VBLORA:
        to_return = {}
        # choose the most efficient dtype for indices
        if config.num_vectors < 2**8:
            indices_dtype = torch.uint8
        elif config.num_vectors < 2**15:
            indices_dtype = torch.int16
        elif config.num_vectors < 2**31:
            indices_dtype = torch.int32
        else:
            indices_dtype = torch.int64
        if config.save_only_topk_weights:
            # in save_only_topk_weights mode, we save topk_indices and topk_weights for parameter efficiency
            for k in state_dict:
                if 'vblora_logits' in k:
                    logits, indices = state_dict[k].topk(config.topk)
                    to_return.update({k + '_topk_indices': indices.to(dtype=indices_dtype)})
                    to_return.update({k + '_topk_weights': torch.softmax(logits, dim=-1)[:, :, :-1].contiguous()})
        else:
            to_return = {k: state_dict[k] for k in state_dict if 'vblora_logits' in k}
        to_return['base_model.vblora_vector_bank.' + adapter_name] = state_dict[
            'base_model.vblora_vector_bank.' + adapter_name
        ]
    elif config.peft_type == PeftType.BONE:
        to_return = {k: state_dict[k] for k in state_dict if 'bone_' in k}
    else:
        raise ValueError(f'Unknown PEFT type passed: {config.peft_type}')

    # MODULES TO SAVE
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in state_dict.items():
            if any(f'{module_name}.modules_to_save.{adapter_name}' in key for module_name in model.modules_to_save):
                to_return[key.replace('modules_to_save.', '')] = value

    # DEAL WITH EMBEDDINGS
    # check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == 'auto'
        and hasattr(config, 'target_modules')
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn('Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.')
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == 'auto':
        vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', None)
        model_id = getattr(config, 'base_model_name_or_path', None)

        # For some models e.g. diffusers the text config file is stored in a subfolder
        # we need to make sure we can download that config.
        has_base_config = False

        # ensure that this check is not performed in HF offline mode, see #1452
        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, 'config.json'))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, 'config.json')
            if exists is None:
                # check failed, could not determine if it exists or not
                warnings.warn(
                    f'Could not find a config file in {model_id} - will assume that the vocabulary was not modified.'
                )
                has_base_config = False
            else:
                has_base_config = exists

        # check if the vocab size of the base model is different from the vocab size of the finetuned model
        if (
            vocab_size
            and model_id
            and has_base_config
            and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size)
        ):
            warnings.warn(
                'Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.'
            )
            save_embedding_layers = True
        else:
            save_embedding_layers = False

    if save_embedding_layers and hasattr(model, 'get_input_embeddings'):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                # support from version >= 0.6.2
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn('Could not identify embedding layer(s) because the model is not a 🤗 transformers model.')

    # REMOVE ADAPTER NAME
    to_return = {k.replace(f'.{adapter_name}', ''): v for k, v in to_return.items()}
    return to_return


def patch_save_and_load():
    peft.get_peft_model_state_dict = get_peft_model_state_dict
    peft.utils.get_peft_model_state_dict = get_peft_model_state_dict
    peft.utils.save_and_load.get_peft_model_state_dict = get_peft_model_state_dict
    peft.tuners.lora.model.get_peft_model_state_dict = get_peft_model_state_dict
    peft.peft_model.get_peft_model_state_dict = get_peft_model_state_dict
