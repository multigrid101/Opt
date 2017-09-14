import pyparsing as pp

import pdb

floating = pp.Combine(pp.Word(pp.nums) + pp.Literal(".") + pp.Word(pp.nums))
colSep = pp.Literal("|")
sepLine = pp.Literal("----------------------+----------+-----------+----------") | pp.Literal("--------------------------------------------------------")
kernelName = pp.Word(pp.alphanums + "_").setResultsName("kernelName")
numRuns = pp.Word(pp.nums).setResultsName("numRuns")

timingValue = floating.setResultsName("timingValue")
timingUnit = pp.Literal("ms")
timing = timingValue + timingUnit
timingTotal = timing.setResultsName("timingTotal")
timingAverage = timing.setResultsName("timingAverage")

timeInfoLine = pp.Group(kernelName + colSep + numRuns + colSep + timingTotal + colSep + timingAverage + pp.LineEnd() + sepLine)
timeInfoAll = pp.OneOrMore(timeInfoLine).setResultsName("timeInfoAll")

def getTimingLines(outputstring):
    return timeInfoAll.scanString(outputstring)



class TimingInfoAll:
    def __init__(self):
        self._timeInfos = {}

    def __str__(self):
        return str(self._timeInfos)

    def getKernelNames(self):
        return [ti._kernelName for key, ti in self._timeInfos.iteritems()]

    def getTimeForKernel(self, kernelName):
        return self._timeInfos[kernelName].getTimingAverage()

    def parseOutput(self, output):
        rs = timeInfoAll.scanString(output)

        # get [(parseResults, start, end), ... ]
        parseResult = [r for r in rs] # this list should only have one element

        # get first (and only) element from list and  then the 'parseResult' from that tuple and timInfoAll from there (now we have a list of timeInfoLine)
        parseResult = parseResult[0][0].timeInfoAll

        tinfos = [(r.kernelName, int(r.numRuns), float(r.timingAverage.timingValue)) for r in parseResult]
        for t in tinfos:
            ti = TimingInfoKernel()
            self._timeInfos[t[0]] = ti
            ti.setKernelName(t[0])
            ti.setNumRuns(t[1])
            ti.setTimingAverage(t[2])




class TimingInfoKernel:
    def __init(self):
        self._kernelName = ""
        self._numRuns = None
        self._timingTotal = None
        self._timingAverage = None


    def __repr__(self):
        return "TimingInfoKernel(numRuns={0}, timingAverage={1})".format(str(self._numRuns), self._timingAverage)


    def setKernelName(self, kernelName):
        self._kernelName = kernelName


    def setTimingTotal(self, timingTotal):
        self._timingTotal = timingTotal


    def setTimingAverage(self, timingAverage):
        self._timingAverage = timingAverage


    def getTimingAverage(self):
        return self._timingAverage


    def setNumRuns(self, numRuns):
        self._numRuns = numRuns



# ti = TimingInfoAll()
# ti.parseOutput(output)
# print ti._totaltime
