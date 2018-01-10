# Helper module to parse example output for plots.
import re

#------------------------------------------------------------------------------
# FUNCTION TO PARSE OUTPUT
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
    overallTime = float(match.group(2))
    return overallTime

def getFinalCostFromRawOutput(output):
    match = re.search("final cost=(.*)", output)
    finalCost = float(match.group(1))
    return finalCost


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
