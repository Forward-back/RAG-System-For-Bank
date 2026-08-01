import logging
import sys

def get_logger(name:str)->logging.Logger:
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    
    if not logger.handlers:
        handlers=logging.StreamHandler(sys.stdout)
        
    
        formatter=logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )
        
        handlers.setFormatter(formatter)
        logger.addHandler(handlers)
        
    return logger


        