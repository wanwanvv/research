/*
An Experimental Study on Hub Labeling based Shortest Path Algorithms [Experiments and Analyses]

Authors: Ye Li, Leong Hou U, Man Lung Yiu, Ngai Meng Kou
Contact: yb47438@umac.mo
Affiliation: University of Macau

The MIT License (MIT)

Copyright (c) 2016 University of Macau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#pragma once
#ifndef PARAS_H
#define PARAS_H
#include<cstdint> 
#include <limits>

namespace SP_Constants {
	bool DIRECTED_FLAG = false;
	bool WEIGHTED_FLAG = false;
	int numOfVertices = 0;
	int numOfEdges = 0;
	const extern int INF_WEIGHT = std::numeric_limits<int>::max() / 3;
}

namespace Processing{
        template<typename T>
	    struct  calcCoefficient//struct to store the relative coefficients of the importance of a node
        {
            T deg_mult=0;//coefficient of node degree
            T freq_mult=0;//coefficient of node query frequency
            T cov_mult=0;//coefficient of coverage
            T dep_mult=0;//coefficient of the distance from current node to last choose node denote the uniformity
            bool is_deg_mult=false;
            bool is_freq_mult=false;
            bool is_cov_mult=false;
            bool is_dep_mult=false;
            calcCoefficient():deg_mult(0),freq_mult(0),cov_mult(0),dep_mult(0),is_deg_mult(false),is_freq_mult(false),is_cov_mult(false),is_dep_mult(false){}
            calcCoefficient(T de,T frq,T cov,T dep,bool is_deg, bool is_freq, bool is_cov, bool is_dep):deg_mult(de),freq_mult(frq),cov_mult(cov),dep_mult(dep),is_deg_mult(is_deg),is_freq_mult(is_freq),is_cov_mult(is_cov),is_dep_mult(is_dep){}
        };
}

#endif