import os

OPENAI_API_KEY = "... "

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value

# from config import set_environment
# set_environment()