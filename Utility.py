import os
import yaml
import logging

from dotenv import load_dotenv

def get_config():
    load_dotenv()
    # Load base.yaml configuration
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_dir, "configs", "base.yaml")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("No config file provided")

    return config


def get_raw_data_directory()->str:
    """
    Returns the directory where raw source files are located
    Returns:
        raw data directory
    Raises:
        ValueError: If raw_data value not exist in base.yaml
    """
    config = get_config()
    data_dir = config.get("paths",{}).get("raw_data_dir","")
    if data_dir == "":
        raise ValueError("No data_dir provided")
    return data_dir


def get_base_dir():
    config = get_config()
    base_dir = config.get("paths",{}).get("base_dir","")
    if base_dir == "":
        raise ValueError("No base_dir provided")
    os.makedirs(os.path.join(base_dir, base_dir), exist_ok=True)
    return base_dir


def get_log_dir():
    """
    Returns the directory where logs are stored and create log directory if it doesn't exist
    Returns:
        log directory
    Raises:
        ValueError: If log_dir value not exist in base.yaml
    """
    config = get_config()
    base_dir = get_base_dir()
    log_dir = config.get("paths",{}).get("log_dir","")
    if log_dir != "":
        os.makedirs(os.path.join(base_dir, log_dir), exist_ok=True)
    else:
        raise ValueError("No log_dir provided")
    return log_dir


def set_logger(log_file:str):
    """
    Set log file with consistent logging structure
    Args:
        log_file(str): log file name
    Returns:
        None
    Raises:
        None
    """
    base_dir = get_base_dir()
    log_dir = get_log_dir()
    log_file = os.path.join(base_dir,log_dir,log_file)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def setup_logger(log_file:str):
    """
    Set log file with consistent logging structure for API
    Args:
        log_file(str): log file name
    Returns:
        logger
    Raises:
        None
    """
    base_dir = get_base_dir()
    log_dir = get_log_dir()

    log_file = os.path.join(base_dir,log_dir,log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (important with reload)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def get_splitter():
    config = get_config()
    splitter = config.get("ingestion",{}).get("splitter","")
    if splitter == "":
        raise ValueError("No splitter provided")
    return splitter


def get_chunk_size(splitter:str):
    config = get_config()
    chunk_size = config.get("ingestion",{}).get(splitter,"").get("chunk_size",100)
    return chunk_size


def get_chunk_overlap(splitter:str):
    config = get_config()
    chunk_overlap = config.get("ingestion",{}).get(splitter,"").get("chunk_overlap",10)
    return chunk_overlap


def get_embedding_type():
    config = get_config()
    embedding_type = config.get("embeddings",{}).get("provider","")
    if embedding_type == "":
        raise ValueError("No embedding_type provided")
    return embedding_type


def get_model_name(embedding_type):
    config = get_config()
    model_name = config.get("embeddings",{}).get(embedding_type,{}).get("model_name","")
    if model_name == "":
        raise ValueError("No model_name provided")
    return model_name


def get_vector_db():
    config = get_config()
    vector_db = config.get("vector_store",{}).get("backend","")
    if vector_db == "":
        raise ValueError("No vector_store provided")
    return vector_db


def get_vector_dir(dbType:str):
    config = get_config()
    base_dir = get_base_dir()
    vector_dir = config.get("vector_store",{}).get(dbType).get("persist_directory","")
    if vector_dir == "":
        raise ValueError("No vector_dir provided")
    return os.path.join(base_dir,vector_dir)


def get_llm_provider():
    config = get_config()
    llm_model = config.get("generation",{}).get("provider","")
    if llm_model == "":
        raise ValueError("No llm provider configured.")
    return llm_model


def get_temperature(llm_provider:str):
    config = get_config()
    temperature = config.get("generation",{}).get(llm_provider,{}).get("temperature",0)
    return temperature


def get_llm_model(llm_provider:str):
    config = get_config()
    llm_model = config.get("generation",{}).get(llm_provider,{}).get("model_name","")
    if llm_model == "":
        raise ValueError("No llm model provided")
    return llm_model


def get_llm_max_retries(llm_provider:str):
    config = get_config()
    max_retries = config.get("generation",{}).get(llm_provider,{}).get("max_retries",1)
    return max_retries


