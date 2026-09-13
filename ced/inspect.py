"""Print the model structure and unique parameter count."""

import argparse
import json

from .config import ModelConfig
from .model import CEDLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/micro.json")
    args = parser.parse_args()
    config = ModelConfig.from_json(args.config)
    model = CEDLanguageModel(config)
    count = model.parameter_count()
    report = {
        "config": config.to_dict(),
        "parameters": count,
        "parameter_storage_fp32_mib": round(count * 4 / 2**20, 2),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
