"""
Created on January 28, 2024
@author: Lance A. Endres
"""
from   lendres.io.ConsoleHelper                                      import ConsoleHelper


class IOSingleton(object):
    ConsoleHelper = None

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(IOSingleton, cls).__new__(cls)
            cls.ConsoleHelper = ConsoleHelper()
        return cls.instance


IO = IOSingleton()