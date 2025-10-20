from peft import PeftType, mapping, peft_model

from . import StellaConfig, StellaModel


def patch_peft_mapping():
    # monkey patching to add STELLA to the PEFT_TYPE_TO_CONFIG_MAPPING
    mapping.PEFT_TYPE_TO_CONFIG_MAPPING['STELLA'] = StellaConfig
    mapping.PEFT_TYPE_TO_TUNER_MAPPING['STELLA'] = StellaModel


def patch_peft_model():
    peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.STELLA] = StellaModel
