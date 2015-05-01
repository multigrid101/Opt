#include "main.h"
#include <OptImage.h>

/** \file App.cpp */
#include "App.h"


int App::launchG3DVisualizer() {
    GApp::Settings settings;
    settings.window.framed = false;
    // Change the window and other startup parameters by modifying the
    // settings class.  For example:
    settings.window.width = 1280;
    settings.window.height = 720;
    settings.window.alwaysOnTop = true;
    _g3dVisualizer = new G3DVisualizer(settings);
    return _g3dVisualizer->run();
}

void App::init()
{
    _g3dVisualizer = NULL;
}

UINT32 App::moveWindow(int x, int y, int width, int height) {
    if (notNull(_g3dVisualizer) && _g3dVisualizer->initialized()) {
        _g3dVisualizer->sendMoveMessage(x, y, width, height);
    }
    return 0;
}

UINT32 App::processCommand(const string &command)
{
	vector<string> words = ml::util::split(command, "\t");
	//while(words.Length() < 5) words.PushEnd("");
    _errorString = " ";
    if (words[0] == "load") {
        _terraFile = words[1];
    } else if (words[0] == "run") {
        string optimizationMethod = words[1];
        _g3dVisualizer->sendRunOptMessage(_terraFile, optimizationMethod);
    } 

	return 0;
}


int App::getIntegerByName(const string &s)
{
	if(s == "layerCount")
	{
		return 1;
	}
	else
	{
		MLIB_ERROR("Unknown integer");
		return -1;
	}
}

float App::getFloatByName(const string &s)
{
    OptimizationTimingInfo timingInfo = _g3dVisualizer->acquireTimingInfo();
    if (s == "defineTime")
    {
        return timingInfo.optDefineTime;
    }
    else if (s == "planTime")
    {
        return timingInfo.optPlanTime;
    }
    else if (s == "solveTime")
    {
        return timingInfo.optSolveTime;
    }
    else if (s == "solveTimeGPU")
    {
        return timingInfo.optSolveTimeGPU;
    }
    else
    {
        MLIB_ERROR("Unknown integer");
        return -1;
    }
}

const char* App::getStringByName(const string &s)
{

	if(s == "terraFilename")
	{
        _queryString = _terraFile;
	}
    else if (s == "error") {
        _queryString = _errorString;
    } else {
        MLIB_ERROR("Unknown string");
		return NULL;
	}
    return  _queryString.c_str();
}
