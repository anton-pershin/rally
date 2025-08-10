import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rally.utils.common import get_config_path
from rally.utils.console import console


CONFIG_NAME = "config_chat"


def chat(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(chat)()

