import os
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        status   = True
        messages = []
        data_dir = Path(self.config.data_dir)

        for split in self.config.required_files:
            split_path = data_dir / split
            if not split_path.exists():
                status = False
                messages.append(f"MISSING split: {split_path}")
                continue
            for sub in ["images", "labels"]:
                sub_path = split_path / sub
                if not sub_path.exists():
                    status = False
                    messages.append(f"MISSING subfolder: {sub_path}")
                else:
                    count = len(list(sub_path.iterdir()))
                    if count == 0:
                        status = False
                        messages.append(f"EMPTY: {sub_path}")
                    else:
                        logger.info(f"OK {sub_path} ({count} files)")

        yaml_path = data_dir / "data.yaml"
        if not yaml_path.exists():
            status = False
            messages.append(f"MISSING data.yaml: {yaml_path}")
        else:
            logger.info(f"OK {yaml_path}")

        os.makedirs(Path(self.config.status_file).parent, exist_ok=True)
        with open(self.config.status_file, "w") as f:
            f.write(f"Validation status: {status}\n")
            if messages:
                f.write("\nIssues:\n" + "\n".join(messages))

        if not status:
            logger.error("Validation FAILED:\n" + "\n".join(messages))
        else:
            logger.info("Validation PASSED.")
        return status
