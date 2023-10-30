"""
Created on Thu Sep 21 08:34:24 2023
@author: Lance A. Endres
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from   lendres.demonstration.FunctionGenerator                  import FunctionGenerator

# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.random.random(20)
y = np.random.random(20)


def PlotOneAxisExample():
    sinX, sinY = FunctionGenerator.GetSineWave(magnitude=2, frequency=10, yOffset=0.5, slope=0, steps=500, timeLength=1.0)

    plt.figure()
    plt.subplot(211)
    lines = plt.plot(sinX, sinY, lw=6, c='y')
    print("Sine z order:", lines[0].get_zorder())
    lines = plt.plot(x, y, 'C3', lw=3)
    print("Lines z order:", lines[0].get_zorder())
    lines = plt.scatter(x, y, s=120)
    print("Scatter z order:", lines.get_zorder())
    plt.title('Lines on top of dots')

    # Scatter plot on top of lines.
    print()
    plt.subplot(212)
    axes = plt.gca()
    lines = axes.plot(sinX, sinY, zorder=3, lw=6, c='y')
    print("Sine z order:", lines[0].get_zorder())
    lines = axes.plot(x, y, 'C3', zorder=2, lw=3)
    print("Lines z order:", lines[0].get_zorder())
    lines = axes.scatter(x, y, s=120, zorder=2)
    print("Scatter z order:", lines.get_zorder())
    plt.title('Dots on top of lines')
    plt.tight_layout()
    plt.show()


def PlotTwoAxisExample():
    sine1a = FunctionGenerator.GetSineWaveAsDataFrame(magnitude=5, frequency=4, yOffset=0, slope=0, steps=1000)
    sine1a.rename({"y" : "Sine a"}, axis="columns", inplace=True)
    sine1b = FunctionGenerator.GetSineWaveAsDataFrame(magnitude=10, frequency=6, yOffset=0, slope=10, steps=1000)
    sine1b.rename({"y" : "Sine b"}, axis="columns", inplace=True)
    sine1b.drop("x", axis=1, inplace=True)
    sine1 = pd.concat([sine1a, sine1b], axis=1)
    sine1.name = "Data 1"

    # Generate the second data set of 2 sine waves.
    sine2a = FunctionGenerator.GetSineWaveAsDataFrame(magnitude=8, frequency=2, yOffset=12, slope=-6, steps=500)
    sine2a.rename({"y" : "Sine a"}, axis="columns", inplace=True)
    sine2b = FunctionGenerator.GetSineWaveAsDataFrame(magnitude=2, frequency=1, yOffset=20, slope=0, steps=500)
    sine2b.rename({"y" : "Sine b"}, axis="columns", inplace=True)
    sine2b.drop("x", axis=1, inplace=True)
    sine2 = pd.concat([sine2a, sine2b], axis=1)
    sine2.name = "Data 2"

    figure = plt.figure()
    axeses = [figure.gca()]
    axeses.append(axeses[0].twinx())
    axeses[0].set_zorder(2)
    axeses[0].patch.set_alpha(0)
    axeses[1].set_zorder(1)
    axeses[0].plot(sine1["x"], sine1["Sine a"], label="1 - Data 1 Sine a", linewidth="5.0", c="b", zorder=5)
    axeses[1].plot(sine1["x"], sine1["Sine b"], label="2 - Data 1 Sine b", linewidth="5.0", c="m", zorder=4)
    axeses[0].plot(sine2["x"], sine2["Sine a"], label="3 - Data 2 Sine a", linewidth="5.0", c="g", zorder=3)
    axeses[1].plot(sine2["x"], sine2["Sine b"], label="4 - Data 2 Sine b", linewidth="5.0", c="r", zorder=2)
    axeses[0].set(ylabel="Left (a)")
    axeses[1].set(ylabel="Right (b)")
    figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axeses[0].transAxes)
    plt.show()


PlotTwoAxisExample()