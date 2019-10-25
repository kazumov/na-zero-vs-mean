#!/usr/bin/env python3

import os, sys
from typing import Dict
import argparse

from model import ModelDDDD
from data import Data
from fitter import Fitter
from plot import FittingAccuracy, FittingLossFunction
import base64


def parameters() -> Dict:
    parser = argparse.ArgumentParser(
        description="Fits model with data set and produces an accuracy and a loss function plots.",
        prefix_chars="--",
    )

    parser.add_argument("--data", dest="data_file_url", help="sets a data file url")

    parser.add_argument(
        "--signature", dest="data_signature", help="sets a data signature"
    )

    # parser.add_argument("--model", dest="model_file_url", help="sets a model file url")

    args = parser.parse_args()

    return args


def main():
    params = parameters()

    dataFile = params.data_file_url

    dataSignature = params.data_signature

    data = Data().read(dataFile)

    fitter = Fitter()

    history = fitter.fit(data, epochs=300)

    FittingAccuracy()

    plotAcc = FittingAccuracy(dataSignature=dataSignature)

    plotLoss = FittingLossFunction(dataSignature=dataSignature)

    plotAcc.plot(history)

    plotLoss.plot(history)


if __name__ == "__main__":
    main()
