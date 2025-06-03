import functools
import os
import torch
import torch.nn.functional as F
import torchmetrics
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizer


from dataset import get_hh
#from dpocriterion import Criterion
from utils import setup, make_dataloader, empty_cache, mcq

device = setup()
MODEL_PATH = Path(os.environ["DSDIR"]) / "HuggingFace" / "allenai" / "tulu-3-sft-mixture" / "data"
DS_PATH = Path(os.environ["DSDIR"]) / "HuggingFace_Models" / "meta-llama" / "Meta-Llama-3-70B"

BATCH_SIZE = 128
EPOCHS = 2

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id





