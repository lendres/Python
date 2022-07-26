"""
Created on March 6, 2022
@author: Lance A. Endres
"""
import numpy                          as np
import os

from lendres.ConsoleHelper            import ConsoleHelper
from lendres.DataHelper               import DataHelper

def MakeDataHelper(inputFile, verboseLevel):
    inputFile          = os.path.join("Data", inputFile)

    consoleHelper      = ConsoleHelper(verboseLevel=verboseLevel)
    dataHelper         = DataHelper(consoleHelper=consoleHelper)
    dataHelper.LoadAndInspectData(inputFile)

    return dataHelper


def GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropFirst=True):
    inputFile               = "credit.csv"
    dependentVariable       = "default"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    FixCreditData(dataHelper, dropFirst)

    return dataHelper, dependentVariable


def FixCreditData(dataHelper, dropFirst):
    replaceStruct = {
        "checking_balance"    : {"< 0 DM" : 1, "1 - 200 DM" : 2,"> 200 DM" : 3, "unknown" : -1},
        "credit_history"      : {"critical" : 1, "poor" : 2, "good" : 3, "very good" : 4, "perfect" : 5},
        "savings_balance"     : {"< 100 DM" : 1, "100 - 500 DM" : 2, "500 - 1000 DM" : 3, "> 1000 DM" : 4, "unknown" : -1},
        "employment_duration" : {"unemployed" : 1, "< 1 year" : 2, "1 - 4 years" : 3, "4 - 7 years" : 4, "> 7 years" : 5},
        "phone"               : {"no" : 1, "yes" : 2 },
        "default"             : {"no" : 0, "yes" : 1 }
    }
    oneHotCols = ["purpose", "housing", "other_credit", "job"]

    dataHelper.ChangeAllObjectColumnsToCategories()
    dataHelper.data = dataHelper.data.replace(replaceStruct)
    dataHelper.EncodeCategoricalColumns(columns=oneHotCols, dropFirst=dropFirst)


def GetLoanModellingData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropExtra=True):
    inputFile               = "Loan_Modelling.csv"
    dependentVariable       = "Personal_Loan"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    FixLoanModellingData(dataHelper, dropExtra)

    return dataHelper, dependentVariable


def FixLoanModellingData(dataHelper, dropExtra):
    dataHelper.data.drop(["ID"], axis=1, inplace=True)
    dataHelper.RemoveRowsWithValueOutsideOfCriteria("Experience", 0, "dropbelow", inPlace=True)
    dataHelper.EncodeCategoricalColumns(["Family", "Education"])

    if dropExtra:
        dataHelper.data.drop(["ZIPCode"], axis=1, inplace=True)
        dataHelper.DropOutliers("Income")
        dataHelper.DropOutliers("CCAvg")
        dataHelper.DropOutliers("Mortgage")


def GetInsuranceData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, encode=True):
    inputFile               = "insurance.csv"
    dependentVariable       = "charges"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    if encode:
        dataHelper.EncodeCategoricalColumns(["region", "sex", "smoker"])

    return dataHelper, dependentVariable


def GetDataWithErrors(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "datawitherrors.csv"
    dependentVariable       = "charges"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    return dataHelper, dependentVariable


def GetBackPainData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "backpain.csv"
    dependentVariable       = "Status"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    dataHelper.ConvertCategoryToNumeric(dependentVariable, "Abnormal")

    return dataHelper, dependentVariable


def GetUsedCarsData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "used_cars_data.csv"
    dependentVariable       = "Price"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    return dataHelper, dependentVariable


def GetCardioGoodFitnessData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "CardioGoodFitness.csv"
    dependentVariable       = "Product"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    categoryDataNames       = ["Product", "Gender", "MaritalStatus", "Usage", "Fitness"]
    dataHelper.ChangeToCategoryType(categoryDataNames)

    return dataHelper, dependentVariable


def GetCardiacData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "Cardiac.csv"
    dependentVariable       = "UnderRisk"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    dataHelper.ConvertCategoryToNumeric("Gender", "Male")
    dataHelper.ConvertCategoryToNumeric("UnderRisk", "yes")

    return dataHelper, dependentVariable


def GetTechnicalSupportData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "technical_support_data.csv"
    dependentVariable       = None

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    return dataHelper, dependentVariable


def GetCustomerSpendData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "Cust_Spend_Data.csv"
    dependentVariable       = None

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    return dataHelper, dependentVariable


def GetCarMpgData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "car-mpg.csv"
    dependentVariable       = "mpg"

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    dataHelper.data         = dataHelper.data.replace("?", np.nan)
    dataHelper.data["hp"]   = dataHelper.data["hp"].astype("float64")

    dataHelper.data["hp"]   = dataHelper.data["hp"].fillna(dataHelper.data["hp"].median())

    return dataHelper, dependentVariable


def GetCreditCardCustomerData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
    inputFile               = "Credit Card Customer Data.csv"
    dependentVariable       = None

    dataHelper              = MakeDataHelper(inputFile, verboseLevel)

    return dataHelper, dependentVariable