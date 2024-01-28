"""
Created on January 28, 2024
@author: Lance A. Endres
"""
from   lendres.io.ConsoleHelper                                      import ConsoleHelper


class IOSingleton(object):
    consoleHelper = None

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(IOSingleton, cls).__new__(cls)
            cls.consoleHelper = ConsoleHelper()
        return cls.instance


IO = IOSingleton()