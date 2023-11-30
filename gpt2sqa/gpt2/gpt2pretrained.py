import logging

import torch
import os
from torch import nn

from gpt2sqa.gpt2.gpt2config import GPT2Config
from gpt2sqa.file_utils import cached_path, CONFIG_NAME, WEIGHTS_NAME
from gpt2sqa.gpt2.layer_norm import LayerNorm


logger = logging.getLogger(__name__)

#
# PRETRAINED_MODEL_ARCHIVE_MAP = {"gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin"}
# PRETRAINED_CONFIG_ARCHIVE_MAP = {"gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json"}
#

class GPT2PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def set_tied(self):
        pass

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path='gpt2'
    ):
        """
        Instantiate a GPT2PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `gpt2`
                - a path or url to a pretrained model archive containing:
                    . `gpt2_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a GPT2Model instance
                - a path or url to a pretrained model archive containing:
                    . `gpt2_config.json` a configuration file for the model
                    . a TensorFlow checkpoint with trained weights
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific GPT class
        """

        archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file)
            resolved_config_file = cached_path(config_file)
        except EnvironmentError:
            logger.error(
                "Model name or path '{}' was not found. "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path, pretrained_model_name_or_path,
                    archive_file, config_file
                )
            )
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info("loading weights file {}".format(archive_file))
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))
        # Load config
        config = GPT2Config.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config)
        return model
