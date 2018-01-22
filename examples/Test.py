from __future__ import print_function
import pdb

import os
import subprocess as sp
import re
from TimingInfo import TimingInfoAll



# class represents one run of an example and contains information such as input
# parameters, the raw text-output of the examples and some information that is
# required to build the example and run it (e.g. the folder-name etc.)
class ExampleRun:
    def __init__(self, homedir):
        self._args = []
        self._hasfinished = False
        self._homedir = homedir # root dir of test, eg. Opt/examples/image_warping
        self._output = "Test hasn't been run yet!"
        self._execCommand = ""
        self._printOutput = False # if true, self._output is printed after test completion


    def run(self):
        os.chdir(self._homedir)

        sp.Popen("make").communicate()

        callcommand = [self._execCommand] + self._args
        print(*callcommand, sep=' ')
        process = sp.Popen(callcommand, stdout=sp.PIPE)
        self._output = process.communicate()[0]
        if self._printOutput == True:
            print(self._output)

        os.chdir("..")

        self._hasfinished = True

        # self.parseOutput()

        # if self._checkResult:
        #     relerror = abs(float(self._referenceCost) - self._finalCost)/self._finalCost
        #     print'asdf'
        #     print relerror
        #     print'asdf'
        #     if relerror > self._tolerance:
        #         print "ERROR={0} IN FINAL COST IS LARGER THAN TOL={1}".format(relerror, self._tolerance)
        #     else:
        #         print "final cost is OK"

    def printOutput(self):
        print(self._output)

    def getOutput(self):
        return self._output

    def getCallCommand(self):
        return [self._execCommand] + self._args


    def printInfoStart(self):
        print("\n\n")
        print("START Test: " + self._homedir)
        print("commando was TODO")


    def printInfoEnd(self):
        print("simulation finished by testrunner")
        print("END Test")
        print("\n\n")

   
    def parseOutput(self):
        pass
        # self.parseFinalCost()
        # ti = TimingInfoAll()
        # ti.parseOutput(self._output)
        # self._timeInfoAll = ti

    def parseFinalCost(self):
        pass
        # match = re.search("final cost=(.*)", self._output)
        # self._finalCost = float(match.group(1))


    def parseTime(self, kernelName):
        pass
        # match = makeKernelRegex(kernelName).search(self._output)
        # self._timingValue[kernelName] = float(match.group(2))
        # self._timingUnit[kernelName] = match.group(3)

