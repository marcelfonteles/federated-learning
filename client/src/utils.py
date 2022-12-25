# utils.py
from datetime import datetime
import os


def logging(message, write_to_file, filepath):
    print(message)
    if write_to_file:
        try:
            if os.path.exists(filepath):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'
            f = open(filepath, append_write)
            f.write('[%s] %s \n' % (datetime.now(), message))
            f.close()
        except Exception as e:
            print(e)
