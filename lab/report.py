from __future__ import annotations

"""Markdown output"""

import os, sys, uuid

from datetime import datetime

from typing import List, Union

from abc import ABC

import numpy as np

from icons import Icon

from data import Data


class Markdown(ABC):
    """Text file in markdown format"""

    def __init__(self):
        self.buffer = ""

    def save(self, path: Union(str, None) = None) -> Markdown:
        """Saves file"""
        name = str(uuid.uuid4()) + ".md"

        if os.path.exists(path) == False:
            raise Exception("Directory " + path + " does not exist.")

        url = path + os.sep + name

        with open(url, "w") as f:
            f.write(self.buffer)

        return self

    def p(self, text: str = "") -> Markdown:
        """Writes paragraph"""
        self.buffer = self.buffer + text + os.linesep * 2
        return self

    def header(self, level: int = 1, text: str = "") -> Markdown:
        """Writes header"""
        self.buffer = self.buffer + "#" * level + text + os.linesep * 2
        return self

    def table(self, columns: List = [], align: List = [], *cells) -> Markdown:
        """Writes table"""
        if len(columns) > len(align):
            align = align + [":---"] * (len(columns) - len(align))

        if len(columns) < len(align):
            align = align[: len(columns)]

        obs = int(len(cells) / len(columns))

        body = np.reshape(list(cells), (obs, len(columns)))

        self.buffer = self.buffer + "| " + " | ".join(columns) + " |" + os.linesep

        self.buffer = self.buffer + "| " + " | ".join(align) + " |" + os.linesep

        for line in body:
            self.buffer = self.buffer + "| " + " | ".join(line) + " |" + os.linesep

        self.buffer = self.buffer + os.linesep * 2
        return self

    def line(self) -> Markdown:
        """Draws a line"""
        self.buffer = self.buffer + "----" + os.linesep * 2
        return self

    def image(self, imagePath: str = "", description: str = "Plot...") -> Markdown:
        """Inserts image"""
        self.buffer = (
            self.buffer + "![ " + description + " ](" + imagePath + ")" + os.linesep * 2
        )
        return self

    def pre(self, text: str = "") -> Markdown:
        """Code block"""
        pre = "```"

        self.buffer = (
            self.buffer + pre + os.linesep + text + os.linesep + pre + os.linesep * 2
        )
        return self


class Report(Markdown):
    """Experiment report in markdown format."""

    def __init__(self):
        super().__init__()

    def reportHeader(self) -> Report:
        return self

    def ndarray(self, data: ndarray) -> Report:
        import io
        from contextlib import redirect_stdout
        from pandas import DataFrame, option_context

        with io.StringIO() as buf, redirect_stdout(buf):
            with option_context("display.max_rows", 10, "display.max_columns", 10):
                print(DataFrame(data))
            self.pre(buf.getvalue())

        return self

    def data(self, data: Data) -> Report:

        try:
            self.p(text="-- x:")
            self.ndarray(data.x)
        except AttributeError:
            self.pre("Empty.")

        try:
            self.p(text="-- xTrain:")
            self.ndarray(data.xTrain)
        except AttributeError:
            self.pre("Empty.")

        try:
            self.p(text="-- yTrain:")
            self.ndarray(data.xTrain)
        except AttributeError:
            self.pre("Empty.")

        try:
            self.p(text="-- xTest:")
            self.ndarray(data.xTest)
        except AttributeError:
            self.pre("Empty.")

        try:
            self.p(text="-- yTest:")
            self.ndarray(data.yTest)
        except AttributeError:
            self.pre("Empty.")

        return self

