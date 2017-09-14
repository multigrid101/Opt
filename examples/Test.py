import pdb

import os
import subprocess as sp
import re
from TimingInfo import TimingInfoAll



class Test:
    def __init__(self, homedir):
        self._optionslist = []
        self._backend = "backend_cpu"
        self._numthreads = 1
        self._hasfinished = False
        self._homedir = homedir # root dir of test, eg. Opt/examples/image_warping
        self._output = "Test hasn't been run yet!"
        self._finalCost = None
        self._timeInfoAll = None
        self._referenceCost = None
        self._tolerance = None # relative tolerance
        self._checkResult = False

    def setBackend(self, backend):
        self._backend = backend

    def getBackend(self):
        return self._backend

    def setNumthreads(self, numthreads):
        self._numthreads = int(numthreads)

    def getNumthreads(self):
        return self._numthreads

    def setReferenceCost(self, cost):
        self._referenceCost = cost

    def setTolerance(self, tol):
        self._tolerance = tol

    def getKernelNames(self):
        return self._timeInfoAll

    def getTimeForKernel(self, kernelName):
        return self._timeInfoAll.getTimeForKernel(kernelName)

    def run(self):
        os.chdir(self._homedir)

        sp.Popen("make").communicate()

        callcommand = ["./"+self._homedir,
                "--backend", self._backend,
                "--numthreads", str(self._numthreads),
                ]
        process = sp.Popen(callcommand, stdout=sp.PIPE)
        self._output = process.communicate()[0]
        # print self._output

        os.chdir("..")

        self._hasfinished = True
        self.parseOutput()

        if self._checkResult:
            relerror = abs(float(self._referenceCost) - self._finalCost)/self._finalCost
            print'asdf'
            print relerror
            print'asdf'
            if relerror > self._tolerance:
                print "ERROR={0} IN FINAL COST IS LARGER THAN TOL={1}".format(relerror, self._tolerance)
            else:
                print "final cost is OK"

    def printOutput(self):
        print self._output

    def getOutput(self):
        return self._output

    def printInfo(self):
        print("\n\n")
        self.printInfoStart()
        self.printInfoEnd()
        print("\n\n")

    def printInfoStart(self):
        print("\n\n")
        print("START Test: " + self._homedir)
        print "backend: " + self._backend
        print "numthreads: " + str(self._numthreads)


    def printInfoEnd(self):
        print "simulation finished by testrunner"
        print "final cost was " + str(self._finalCost)
        print self._timeInfoAll
        print("END Test")
        print("\n\n")

   
    def parseOutput(self):
        self.parseFinalCost()
        ti = TimingInfoAll()
        ti.parseOutput(self._output)
        self._timeInfoAll = ti

    def parseFinalCost(self):
        match = re.search("final cost=(.*)", self._output)
        self._finalCost = float(match.group(1))


    def parseTime(self, kernelName):
        match = makeKernelRegex(kernelName).search(self._output)
        self._timingValue[kernelName] = float(match.group(2))
        self._timingUnit[kernelName] = match.group(3)


class TestGroup:
    def __init__(self):
        self._tests = []

    def addTest(self, test):
        self._tests.append(test)

    def printScalingInfo(self):
        testlist = self.getTestsWithBackend('backend_cpu_mt')
        numthreadslist = [t.getNumthreads() for t in testlist]
        numthreadslist = list(set(numthreadslist)) # eliminate duplicates and sort
        numthreadslist.sort()
        numthreadslist = [str(n) for n in numthreadslist] # convert to string values

        numthreadsline = "{:50}".format("numthreads") + "".join(["{:10}".format(str(n)) for n in numthreadslist])
        print numthreadsline

        # get kernelnames
        # pdb.set_trace()
        kernelNames = testlist[0]._timeInfoAll.getKernelNames()
        for name in kernelNames:
            line = ""
            absline = line + ("{:50}".format(name + " (abs)"))
            relline = line + ("{:50}".format(name + " (rel)"))
            abstimes = []
            for num in numthreadslist:
                t = self.getTestWithBackendAndNumthreads('backend_cpu_mt', num)
                abstimes.append(t.getTimeForKernel(name))

            reltimes = [1.0/(t/abstimes[0]) for t in abstimes]
            absline = absline + "".join(["{:10.5}".format(t) for t in abstimes])
            relline = relline + "".join(["{:10.5}".format(t) for t in reltimes])
            print absline
            print relline
            print ""

    def getTestsWithBackend(self, backend):
        l = [t for t in self._tests if t.getBackend() == backend]
        return l
        
    def getTestsWithNumthreads(self, num): # only from 'backend_cpu_mt'
        l = [t for t in self._tests if t.getNumthreads() == num and t.getBackend() == 'backend_cpu_mt']
        return l

    def getTestWithBackendAndNumthreads(self, backend, num):
        for t in self._tests:
            if t.getBackend() == backend and t.getNumthreads() == int(num):
                return t

    def runAll(self):
        for test in self._tests:
            if (test.getBackend() is not "backend_cpu_mt") and (test.getNumthreads() is not 1):
                pass
            else:
                test.printInfoStart()       
                test.run()
                test.printInfoEnd()       

        self.printScalingInfo()

