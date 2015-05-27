#pragma once

#ifndef _PATCH_SOLVER_Stereo_UTIL_
#define _PATCH_SOLVER_Stereo_UTIL_

#include "SolverUtil.h"

#define PATCH_SIZE				  (16)
#define SHARED_MEM_SIZE_PATCH	  ((PATCH_SIZE+2)*(PATCH_SIZE+2))
#define SHARED_MEM_SIZE_VARIABLES ((PATCH_SIZE)*(PATCH_SIZE))

/////////////////////////////////////////////////////////////////////////
// Helper
/////////////////////////////////////////////////////////////////////////

__inline__ __device__ unsigned int getLinearThreadId(int tId_i, int tId_j)
{
	return tId_i*PATCH_SIZE+tId_j;
}

__inline__ __device__ unsigned int getLinearThreadIdCache(int tId_i, int tId_j)
{
	return (tId_i + 1)*(PATCH_SIZE + 2) + (tId_j + 1);
}

__inline__ __device__ float readValueFromCache2D(volatile float* cache, int tId_i, int tId_j)
{
	return cache[getLinearThreadIdCache(tId_i, tId_j)];
}


__inline__ __device__ bool isValid(float& v)
{
	return v != MINF;
}

__inline__ __device__ void loadVariableToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	if (gId_i >= 0 && gId_j >= 0 && gId_i < H && gId_j < W) { cache[getLinearThreadIdCache(tId_i, tId_j)] = data[gId_i*W + gId_j];}
	else													{ cache[getLinearThreadIdCache(tId_i, tId_j)] = MINF;	}
}

__inline__ __device__ void loadPatchToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	if (tId_i == 0)				 loadVariableToCache(cache, data, tId_i - 1, tId_j	  , gId_i - 1, gId_j	, W, H);
	if (tId_i == PATCH_SIZE - 1) loadVariableToCache(cache, data, tId_i + 1, tId_j	  , gId_i + 1, gId_j	, W, H);
								 loadVariableToCache(cache, data, tId_i	   , tId_j	  , gId_i,	   gId_j	, W, H);
	if (tId_j == 0)				 loadVariableToCache(cache, data, tId_i    , tId_j - 1, gId_i,	   gId_j - 1, W, H);
	if (tId_j == PATCH_SIZE - 1) loadVariableToCache(cache, data, tId_i    , tId_j + 1, gId_i,	   gId_j + 1, W, H);

	if (tId_i == 0 && tId_j == 0)							loadVariableToCache(cache, data, tId_i - 1, tId_j - 1, gId_i - 1, gId_j - 1, W, H);
	if (tId_i == PATCH_SIZE - 1 && tId_j == 0)			    loadVariableToCache(cache, data, tId_i + 1, tId_j - 1, gId_i + 1, gId_j - 1, W, H);
	if (tId_i == 0 && tId_j == PATCH_SIZE - 1)				loadVariableToCache(cache, data, tId_i - 1, tId_j + 1, gId_i - 1, gId_j + 1, W, H);
	if (tId_i == PATCH_SIZE - 1 && tId_j == PATCH_SIZE - 1) loadVariableToCache(cache, data, tId_i + 1, tId_j + 1, gId_i + 1, gId_j + 1, W, H);
}

__inline__ __device__ void setZero(volatile float* cache, int tId_i, int tId_j)
{
	cache[getLinearThreadIdCache(tId_i, tId_j)] = 0;
}

__inline__ __device__ void setPatchToZero(volatile float* cache, int tId_i, int tId_j)
{
	if (tId_i == 0)				 setZero(cache, tId_i - 1, tId_j	);
	if (tId_i == PATCH_SIZE - 1) setZero(cache, tId_i + 1, tId_j	);
								 setZero(cache, tId_i	 , tId_j	);
	if (tId_j == 0)				 setZero(cache, tId_i	 , tId_j - 1);
	if (tId_j == PATCH_SIZE - 1) setZero(cache, tId_i	 , tId_j + 1);

	if (tId_i == 0 && tId_j == 0)							setZero(cache, tId_i - 1, tId_j - 1);
	if (tId_i == PATCH_SIZE - 1 && tId_j == 0)			    setZero(cache, tId_i + 1, tId_j - 1);
	if (tId_i == 0 && tId_j == PATCH_SIZE - 1)				setZero(cache, tId_i - 1, tId_j + 1);
	if (tId_i == PATCH_SIZE - 1 && tId_j == PATCH_SIZE - 1) setZero(cache, tId_i + 1, tId_j + 1);
}

/*
__inline__ __device__ void loadVariableToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	if (gId_i >= 0 && gId_j >= 0 && gId_i < H && gId_j < W) cache[getLinearThreadIdCache(tId_i, tId_j)] = data[gId_i*W + gId_j];
	else												    cache[getLinearThreadIdCache(tId_i, tId_j)] = MINF;
}

__inline__ __device__ void loadPatchToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	if (tId_i == 0)				 loadVariableToCache(cache, data, tId_i - 1, tId_j, gId_i - 1, gId_j, W, H);
	if (tId_i == PATCH_SIZE - 1) loadVariableToCache(cache, data, tId_i + 1, tId_j, gId_i + 1, gId_j, W, H);
								 loadVariableToCache(cache, data, tId_i, tId_j, gId_i, gId_j, W, H);
	if (tId_j == 0)				 loadVariableToCache(cache, data, tId_i, tId_j - 1, gId_i, gId_j - 1, W, H);
	if (tId_j == PATCH_SIZE - 1) loadVariableToCache(cache, data, tId_i, tId_j + 1, gId_i, gId_j + 1, W, H);

	if (tId_i == 0 && tId_j == 0)							loadVariableToCache(cache, data, tId_i - 1, tId_j - 1, gId_i - 1, gId_j - 1, W, H);
	if (tId_i == PATCH_SIZE - 1 && tId_j == 0)			    loadVariableToCache(cache, data, tId_i + 1, tId_j - 1, gId_i + 1, gId_j - 1, W, H);
	if (tId_i == 0 && tId_j == PATCH_SIZE - 1)				loadVariableToCache(cache, data, tId_i - 1, tId_j + 1, gId_i - 1, gId_j + 1, W, H);
	if (tId_i == PATCH_SIZE - 1 && tId_j == PATCH_SIZE - 1) loadVariableToCache(cache, data, tId_i + 1, tId_j + 1, gId_i + 1, gId_j + 1, W, H);
}
*/

#endif