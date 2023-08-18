"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import DataSetLoading

from   lendres.ConsoleHelper                     import ConsoleHelper

import unittest


skipTests = False

class TestDataHelper(unittest.TestCase):
    #verboseLevel = ConsoleHelper.VERBOSENONE
    verboseLevel = ConsoleHelper.VERBOSETESTING
    #verboseLevel = ConsoleHelper.VERBOSEREQUESTED
    #verboseLevel = ConsoleHelper.VERBOSEIMPORTANT

    @classmethod
    def setUpClass(cls):

        cls.insuranceDataHelper, cls.insuranceDependentVariable = DataSetLoading.GetInsuranceData(verboseLevel=cls.verboseLevel, encode=False)

        cls.loanData, cls.loanDependentVariable = DataSetLoading.GetLoanModellingData(verboseLevel=cls.verboseLevel, dropExtra=False)
        cls.loanData.ChangeToCategoryType(["CreditCard", "Online"])

        cls.dataWithErrors, dependentVariable   = DataSetLoading.GetDataWithErrors(verboseLevel=cls.verboseLevel)

        cls.usedCarData, dependentVariable      = DataSetLoading.GetUsedCarsData(verboseLevel=cls.verboseLevel)

        cls.boundaries      = [0,     90000,   91000,   92000,   93000,   94000,   95000,   96000,   99999]
        cls.labels          = ["Os", "90000", "91000", "92000", "93000", "94000", "95000", "96000", "99999"]


    def setUp(self):
        self.insuranceDataHelper = self.insuranceDataHelper.Copy()
        self.loanData            = self.loanData.Copy()
        self.dataWithErrors      = self.dataWithErrors.Copy()
        self.usedCarData         = self.usedCarData.Copy()


    def testValueCounts(self):
        newColumnName = self.loanData.MergeNumericalDataByRange("ZIPCode", self.labels, self.boundaries);
        self.assertEqual(self.loanData.data[newColumnName].value_counts()["96000"], 40)


    def testNotAvailableCounts(self):
        # Test getting the not available counts with data missing.
        notAvailableCounts, totalNotAvailable = self.dataWithErrors.GetNotAvailableCounts()
        self.assertEqual(totalNotAvailable, 1)

        # Remove the missing data and recheck to make sure it was removed.
        self.dataWithErrors.DropRowsWhereDataNotAvailable(["children"])
        notAvailableCounts, totalNotAvailable = self.dataWithErrors.GetNotAvailableCounts()
        self.assertEqual(totalNotAvailable, 0)


    def testGetMinAndMaxValues(self):
        result = self.loanData.GetMinAndMaxValues("Income", 5, method="quantity")
        self.assertEqual(result["Largest"].iloc[-1], 224)

        solution = self.loanData.data.shape[0] * 0.05
        result   = self.loanData.GetMinAndMaxValues("Income", 5, method="percent")
        self.assertAlmostEqual(len(result["Largest"]), solution, 0)


    @unittest.skipIf(skipTests, "Skipped displaying test.")
    def testDisplaying(self):
        self.loanData.consoleHelper.PrintNewLine(2, ConsoleHelper.VERBOSEREQUESTED)
        self.loanData.DisplayAllCategoriesValueCounts()
        self.loanData.consoleHelper.PrintNewLine(2, ConsoleHelper.VERBOSEREQUESTED)
        self.loanData.DisplayUniqueValues(["Online", "CreditCard"])


    def testSplitComparisons(self):
        self.loanData.SplitData(self.loanDependentVariable, 0.2, 0.3, stratify=False)

        result = self.loanData.GetSplitComparisons()
        self.loanData.consoleHelper.PrintNewLine(2, ConsoleHelper.VERBOSEREQUESTED)
        self.loanData.consoleHelper.PrintTitle("Split Comparisons", ConsoleHelper.VERBOSEREQUESTED)
        self.loanData.consoleHelper.Print(result.T, ConsoleHelper.VERBOSEREQUESTED)

        self.loanData.CreateSplitComparisonPlot()

        self.loanData.DisplayDataShapes()


    @unittest.skipIf(skipTests, "Skipped string extraction test.")
    def testStringExtraction(self):
        columns = ["Mileage", "Engine", "Power"]
        # For this data, the not available rows need to be removed.
        self.usedCarData.DropAllRowsWhereDataNotAvailable()

        result = self.usedCarData.ExtractLastStringTokens(columns)
        result = result.nunique()

        self.loanData.consoleHelper.PrintNewLine(2, ConsoleHelper.VERBOSEREQUESTED)
        self.usedCarData.consoleHelper.PrintTitle("Extracted String Token Counts", ConsoleHelper.VERBOSEREQUESTED)
        self.usedCarData.consoleHelper.Display(result, ConsoleHelper.VERBOSEREQUESTED)

        self.assertEqual(result.loc["Mileage"], 2)
        self.assertEqual(result.loc["Engine"], 1)


    def testCategoryConversion(self):
        self.insuranceDataHelper.data["smoker"] = self.insuranceDataHelper.data["smoker"].astype("category")
        self.insuranceDataHelper.ConvertCategoryToNumeric("smoker", "yes")


    @unittest.skipIf(skipTests, "Skipped print final test.")
    def testPrintFinal(self):
        dataHelper, dependentVariable = DataSetLoading.GetCreditCardCustomerData(verboseLevel=self.verboseLevel)
        dataHelper.consoleHelper.PrintNewLine(2, ConsoleHelper.VERBOSEREQUESTED)
        dataHelper.PrintFinalDataSummary()


if __name__ == "__main__":
    unittest.main()