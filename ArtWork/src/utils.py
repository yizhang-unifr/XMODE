
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
import re
import errno
import os
import signal
import functools
import json
import datetime


import logging
import json

def _get_db_schema(db_path, tables=None, sample_rows_in_table_info=5):
    # Initialize the database connection
    engine = create_engine(f'sqlite:///{db_path}')  # replace with your actual database connection string
    database = SQLDatabase(engine, sample_rows_in_table_info=sample_rows_in_table_info)

    # List of tables you want the schema for
    if tables is None: # get all tables
        db_schema = database.get_table_info()
    db_schema = database.get_table_info(tables)
    return db_schema


def correct_malformed_json(malformed_json_string):
    # Step 1: Replace escaped quotes with actual quotes
    corrected_json_string = malformed_json_string.replace('\\"', '"')
    
    # Step 2: Ensure all keys and values are properly quoted
    # This regular expression will find unquoted strings and put quotes around them
    # It skips already quoted values and datetime formats
    def quote_value(match):
        value = match.group(1)
        if not re.match(r'^".*"$', value) and not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', value):
            value = f'"{value}"'
        return f':{value}'

    corrected_json_string = re.sub(r':(\w+)', quote_value, corrected_json_string)
    
    # Step 3: Handle duplicate keys by making them unique
    # Use a set to track seen keys and a counter for making keys unique
    seen_keys = set()
    def make_unique(match):
        key = match.group(1)
        if key in seen_keys:
            counter = 2
            new_key = f"{key}{counter}"
            while new_key in seen_keys:
                counter += 1
                new_key = f"{key}{counter}"
            key = new_key
        seen_keys.add(key)
        return f'"{key}"'
    
    corrected_json_string = re.sub(r'"(\w+)"(?=:)', make_unique, corrected_json_string)
    
    # Step 4: Add missing closing brace if needed
    if corrected_json_string.count('{') > corrected_json_string.count('}'):
        corrected_json_string += '}'
    
    return corrected_json_string



class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle generators
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
            return list(obj)
        
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        
        # Handle other non-serializable objects by converting to string
        try:
            return str(obj)
        except Exception:
            pass
        
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
    
    



def configure_logging(state, log_name=None, log_filename=None):
    configurable = state['configurable']
    # check logger is already configured
    if configurable.get('logger') is None:
        # set up logger
        if log_name is None:
            log_name = 'default_logger'
        if log_filename is None:
            log_filename = 'default_log.log'
        configurable['logger'] = {
            'log_filename': log_filename,
            'log_name': log_name 
        }
    log_name = configurable['logger']['log_name']
    log_filename = configurable['logger']['log_filename']
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the handlers to logger
    if not logger.hasHandlers():
        logger.addHandler(fh)
    return logger
