# Helper module to parse example output for plots.
import re
import myInfos as info
import sys

# # some helpers for below
# def errorIfNone(arg, name)
#     if arg is Nonwk

#------------------------------------------------------------------------------
# FUNCTION TO PARSE OUTPUT (Basic functions)
def getOverallTimeFromOutput(output):
    match = re.search(" overall\s+\|[^|]+\|([^|]*)ms", output)
    # overallTime = float(match.group(1))
    overallTime = float(match.group(1))
    return overallTime

def getNamedAbsoluteTimeFromOutput(name, output):
    match = re.search(name + "[^|]+\|[^|]+\|([^|]*)ms", output)
    # overallTime = float(match.group(1))
    overallTime = float(match.group(1))
    return overallTime

def getNamedAverageTimeFromOutput(name, output):
    match = re.search(name + "[^|]+\|[^|]+\|([^|]*)ms\|([^|]*)ms", output)
    # overallTime = float(match.group(1))
    if match is None:
        errMsg = """
        getNamedAverageTimeFromOutput():
        Could not find timing for

        {0}

        in output. Output was writting to ./tmp.log for inspection.
        """.format(name)

        with open("./tmp.log", "w") as f:
            f.write(output)

        sys.exit(errMsg)

    overallTime = float(match.group(2))
    return overallTime


# variant for multi-threaded code.
# get e.g. computeCost_W_H_total. For the "loop" costs, use vtune.
def getNamedAverageTimeFromOutputMT(name, output):
    specialnames = ['linear iteration', 'JT alloc', 'J_transpose',
            'Jp', 'J\^T', 'JTJ multiply', 'J\^TJp', 'J\^TJ alloc']

    if name in specialnames:
        match = re.search(name + "[^|]+\|[^|]+\|([^|]*)ms\|([^|]*)ms", output)
    else:
        match = re.search(name + "[^|]+total[^|]+\|[^|]+\|([^|]*)ms\|([^|]*)ms", output)

    if match is None:
        errMsg = """
        getNamedAverageTimeFromOutputMT():
        Could not find timing for

        {0}

        in output. Output was writting to ./tmp.log for inspection.
        """.format(name)

        with open("./tmp.log", "w") as f:
            f.write(output)

        sys.exit(errMsg)

    overallTime = float(match.group(2))
    return overallTime

def getFinalCostFromRawOutput(output):
    match = re.search("final cost=(.*)", output)
    finalCost = float(match.group(1))
    return finalCost

def getUnknownSizeFromOutput(output):
    theRegex = "12 tmp vars of TUnknownType use (.*) bytes each"
    match = re.search(theRegex, output)
    if match is None:
        errMsg = """
        getUnknownSizeFromOutput():
        Could not find regex

        {0}

        in output. Output was writting to ./tmp.log for inspection.
        """.format(theRegex)
        sys.exit(errMsg)

    unknownSizeInBytes = int(match.group(1))
    return unknownSizeInBytes


def getPlanDataSizeFromOutput(output):
    theRegex = "total usage of PlanData: (.*) bytes"
    match = re.search(theRegex, output)
    if match is None:
        errMsg = """
        getPlanDataSizeFromOutput():
        Could not find regex

        {0}

        in output. Output was writting to ./tmp.log for inspection.
        """.format(theRegex)
        sys.exit(errMsg)

    pdSizeInBytes = int(match.group(1))
    return pdSizeInBytes


# -----------------------------------------------------------------------------
# some more advanced helpers
def getAvgCategoryTimeOpt(output, category, materialization, homedir):
    # check input and get appropriate dictionary
    if materialization not in ['matfree', 'JTJ', 'fusedJTJ']:
        errMsg = """
        outParse.getAvgCategoryTimeOpt(): invalid materialization

        {0}

        valid string options are 'matfree', 'JTJ', 'fusedJTJ'.
        """.format(materialization)
        sys.exit(errMsg)

    if category not in [1,2,3]:
        errMsg = """
        outParse.getAvgCategoryTimeOpt(): invalid materialization

        {0}

        valid integer options are 1(=perSolve), 2(=perNewton), 3(=perLinear).
        """.format(category)
        sys.exit(errMsg)

    theDict = {}
    if category == 1:
        theDict  = info.optPerSolveNames[materialization][homedir]
    if category == 2:
        theDict  = info.optPerNewtonNames[materialization][homedir]
    if category == 3:
        theDict  = info.optPerLinIterNames[materialization][homedir]

    print("The dict is {0}".format(theDict))
    
    t = 0.0
    for name in theDict:
        tmp = getNamedAverageTimeFromOutput(name, output)
        t += tmp

    return t


# multi-threaded version
def getAvgCategoryTimeOptMT(output, category, materialization, homedir):
    # check input and get appropriate dictionary
    if materialization not in ['matfree', 'JTJ', 'fusedJTJ']:
        errMsg = """
        outParse.getAvgCategoryTimeOptMT(): invalid materialization

        {0}

        valid string options are 'matfree', 'JTJ', 'fusedJTJ'.
        """.format(materialization)
        sys.exit(errMsg)

    if category not in [1,2,3]:
        errMsg = """
        outParse.getAvgCategoryTimeOptMT(): invalid materialization

        {0}

        valid integer options are 1(=perSolve), 2(=perNewton), 3(=perLinear).
        """.format(category)
        sys.exit(errMsg)

    theDict = {}
    if category == 1:
        theDict  = info.optPerSolveNames[materialization][homedir]
    if category == 2:
        theDict  = info.optPerNewtonNames[materialization][homedir]
    if category == 3:
        theDict  = info.optPerLinIterNames[materialization][homedir]

    print("The dict is {0}".format(theDict))
    
    t = 0.0
    for name in theDict:
        t += getNamedAverageTimeFromOutputMT(name, output)

    return t

# -----------------------------------------------------------------------------
# ceres stuff basics
def getAvgTimeLinIterCeres(output):
    match = re.search("Cost per linear solver iteration: (.*)ms", output)
    finalCost = float(match.group(1))
    return finalCost


def getAvgTimePerNewtonCeres(output):
    numIterMatch = re.search("Minimizer iterations (.*)", output)
    numIter = int(numIterMatch.group(1))

    match = re.search("Jacobian evaluation (.*)", output)
    finalCost = 1000*float(match.group(1))/numIter
    return finalCost
#------------------------------------------------------------------------------
