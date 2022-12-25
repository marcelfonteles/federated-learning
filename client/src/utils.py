# utils.py
from datetime import datetime


def logging(message, write_to_file, filepath):
    print(message)
    if write_to_file:
        try:
            f = open(filepath, 'a')
            f.write('[%s] %s \n' % (datetime.now(), message))
            f.close()
        except Exception as e:
            print(e)
