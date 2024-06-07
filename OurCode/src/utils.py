import os


DATA_FOLDER = os.environ.get("PPREC_DATA_FOLDER")


def get_data_folder(data_folder: str | None = None) -> str:
    if data_folder is not None:
        return data_folder
    elif DATA_FOLDER is not None:
        return DATA_FOLDER
    else:
        raise ValueError(
            "data_folder must be provided as an argument or set as an environment variable 'PPREC_DATA_FOLDER'"
        )
