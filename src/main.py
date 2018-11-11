#!/usr/bin/env python3

import numpy as np
import copy as cp
import os
import pickle
import sys
import warnings
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn

from lib import ising_1d as ising
from lib import regression as reg
from lib import metrics
from lib import bootstrap as bs
from lib import cross_validation as cv
from lib import logistic_regression

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from taskb import task1b, task1b_bias_variance_analysis
from taskc import task1c
from taskd import task1d
from taske import task1e


def main():
    # Initiating parsers
    prog_desc = ("FYS-STK4155 Project 2 command-line utility for using "
                 "Machine Learning on Ising models data.")
    parser = argparse.ArgumentParser(prog="Project 2 ML Analyser",
                                     description=(prog_desc))
    parser.add_argument(
        "-figf", "--figure_folder", default="../fig", type=str,
        help="output path for figures.")
    parser.add_argument("-latex", "--latex", default=False,
                        action="store_true",
                        help=("Intended for production runs. Will ensure "
                              "proper LaTeX font."))

    # Sets up some subparser for each task.
    subparser = parser.add_subparsers(dest="subparser")

    # Task b
    taskb_parser = subparser.add_parser(
        "b", help=("Runs task b: finds the coupling constant for 1d Ising, "
                   "using Linear, Ridge and Lasso regression"))
    taskb_parser.add_argument("-pk", "--pickle_filename",
                              default="bs_kf_data_1b.pkl",
                              type=str,
                              help=("Filename for storing task b analysis "
                                    "output."))
    taskb_parser.add_argument("-use_pk", "--use_pickled_data", default=False,
                              action="store_true",
                              help="If given, will use stored pickled data.")
    taskb_parser.add_argument("-N", "-N_samples", default=1000, type=int,
                              help="N 1D Ising samples to generate")

    # N_samples=1000, training_size=0.1, N_bs=200,
    #        L_system_size=20

    # Task c
    taskc_parser = subparser.add_parser(
        "c", help=("Runs task c: finds the phase of Ising matrices at "
                   "different temperatures using logisitc regression"))
    taskc_parser.add_argument("-pk", "--pickle-filename")

    # Task d
    taskd_parser = subparser.add_parser(
        "d", help=("Runs task d: uses a neural net to perform the regression"
                   " from b"))
    taskd_parser.add_argument("-pk", "--pickle-filename")

    # Task e
    taske_parser = subparser.add_parser(
        "e", help=("Runs task e: uses a neural net to perform the "
                   "classification from c"))
    taske_parser.add_argument("-pk", "--pickle-filename")

    taskall_parser = subparser.add_parser("all", help=("Runs all tasks."))

    # args = parser.parse_args()

    if len(sys.argv) < 2:
        # Manually specify your arguments if you prefer that
        # args = parser.parse_args(["b", "-use_pk"])
        # args = parser.parse_args(["b"])
        args = parser.parse_args(["c"])
        args = parser.parse_args(["d"])
        args = parser.parse_args(["e"])
    else:
        # Or run from terminal
        args = parser.parse_args()

    if args.latex:
        # Proper LaTeX font
        import matplotlib as mpl
        mpl.rc("text", usetex=True)
        mpl.rc("font", **{"family": "sans-serif",
                          "serif": ["Computer Modern"]})
        mpl.rcParams["font.family"] += ["serif"]

    if args.subparser == "b":
        if args.use_pickled_data:
            task1b_bias_variance_analysis(
                args.pickle_filename, figure_folder=args.figure_folder)
        else:
            task1b(args.pickle_filename, figure_folder=args.figure_folder)
    elif args.subparser == "c":
        task1c(figure_folder=args.figure_folder)
    elif args.subparser == "d":
        task1d(figure_folder=args.figure_folder)
    elif args.subparser == "e":
        task1e(figure_folder=args.figure_folder)
    elif args.subparser == "all":
        task1b(args.pickle_filename, figure_folder=args.figure_folder)
        task1b_bias_variance_analysis(args.pickle_filename,
                                      figure_folder=args.figure_folder)
        task1c(figure_folder=args.figure_folder)
        task1d(figure_folder=args.figure_folder)
        task1e(figure_folder=args.figure_folder)
    else:
        exit("Parse error: {} \n--> exiting".format(args))


if __name__ == '__main__':
    main()
