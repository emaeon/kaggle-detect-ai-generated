from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field


@dataclass
class Report:
    with_tracking: bool = False
    name: str = 'wandb'
    wandb_id: str = 'your_wandb_id'

@dataclass
class Path:
    dataset_path: str = './data/with_instruction.csv'
    output_dir: str = './outputs/${now:%Y-%m-%d}'

@dataclass
class Model:
    name: str = "/kaggle/input/llama-2/pytorch/7b-hf/1"
    hidden_size: int = 768

@dataclass
class Data:
    max_seq_len: int = 128

@dataclass
class Param:
    seed: int = 42
    epochs: int = 1
    lr: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    weight_decay: float = 1e-3

@dataclass
class LLMConfig:
    report: Report = field(default_factory=Report)
    path: Path = field(default_factory=Path)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    param: Param = field(default_factory=Param)


cs = ConfigStore.instance()
cs.store(name='config', node=LLMConfig)