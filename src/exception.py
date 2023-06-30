import os, sys
from logger import logging

def error_message_detailed(error, error_detailed:sys):
    _, _, exc_tb = error_detailed.exc_info() #exc_tb = execution from try block, exc_info = execution information
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    #tb_frame used to read code line by line
    #f_code find from where the error is occur
    #co_filename read all the file from current directory
    
    error_message = 'error occurred in python script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name, exc_tb.tb_lineno, str(error)
        )
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detailed:sys):
        super().__init__(error_message)
        self.error_detailed = error_detailed
        self.error_message = error_message_detailed(error_message, error_detailed)
        
    def __str__(self):
        return self.error_message
