# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 12:28:33 2022

@author: Lance
"""

minAndMaxKilometersDriven = GetMinAndMaxValues(data, "Kilometers_Driven", 20, method="quantity")
display(minAndMaxKilometersDriven)

minAndMaxKilometersDriven = GetMinAndMaxValues(data, "Kilometers_Driven", 0.25, method="percent")
display(minAndMaxKilometersDriven)