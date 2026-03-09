import os
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """Validate that all required split folders and subfolders exist.

        Checks for:
            <data_dir>/train/images/   <data_dir>/train/labels/
            <data_dir>/valid/images/   <data_dir>/valid/labels/
            <data_dir>/test/images/    <data_dir>/test/labels/
            <data_dir>/data.yaml

        Returns:
            bool: True if all required paths exist, False otherwise.
        """
        try:
            validation_status = True
            messages = []

            data_dir = Path(self.config.data_dir)

            # Check top-level required folders (train / valid / test)
            for required in self.config.required_files:
                split_path = data_dir / required
                if not split_path.exists():
                    validation_status = False
                    msg = f"MISSING split folder: {split_path}"
                    messages.append(msg)
                    logger.warning(msg)
                else:
                    logger.info(f"Found split folder: {split_path}")

                    # Also validate images/ and labels/ subdirectories
                    for sub in ["images", "labels"]:
                        sub_path = split_path / sub
                        if not sub_path.exists():
                            validation_status = False
                            msg = f"MISSING subfolder: {sub_path}"
                            messages.append(msg)
                            logger.warning(msg)
                        else:
                            file_count = len(list(sub_path.iterdir()))
                            logger.info(
                                f"Found {sub_path} with {file_count} files"
                            )

            # Check data.yaml
            yaml_path = data_dir / "data.yaml"
            if not yaml_path.exists():
                validation_status = False
                msg = f"MISSING data.yaml at: {yaml_path}"
                messages.append(msg)
                logger.warning(msg)
            else:
                logger.info(f"Found data.yaml at: {yaml_path}")

            # Write status file
            os.makedirs(Path(self.config.status_file).parent, exist_ok=True)
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}\n")
                if messages:
                    f.write("\nIssues found:\n")
                    f.write("\n".join(messages))

            logger.info(
                f"Data validation complete — status: {validation_status}"
            )
            return validation_status

        except Exception as e:
            os.makedirs(Path(self.config.status_file).parent, exist_ok=True)
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: False\nError: {str(e)}")
            raise e