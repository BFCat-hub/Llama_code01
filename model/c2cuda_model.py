import logging

from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
import Config

logger = logging.getLogger(__name__)


class C2CUDAModel(LlamaPreTrainedModel):
    def __init__(self, config, max_seq_length: int):
        super().__init__(config)

        self.max_seq_length = max_seq_length
        self.llama = AutoModelForCausalLM(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, s_input_ids, t_input_ids, s_attention_mask, t_attention_mask):
        s_input_ids = s_input_ids.view(-1, s_input_ids.size(-1))
        t_input_ids = t_input_ids.view(-1, t_input_ids.size(-1))
        s_attention_mask = s_attention_mask.view(-1, s_attention_mask.size(-1))
        t_attention_mask = t_attention_mask.view(-1, t_attention_mask.size(-1))
        llama_out_puts = self.llama(s_input_ids, attention_mask=s_attention_mask)
