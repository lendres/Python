"""
Created on September 28, 2022
@author: Lance A. Endres
"""
import nltk
import os


def RunInstall():
   nltk.download("stopwords")
   
   
def Test():
    os.system("python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    Test()