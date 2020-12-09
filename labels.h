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

#ifndef LABELS_H
#define LABELS_H


#include <limits>
#include <climits>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include "graph.h"
#include "paras.h" 
#include <malloc.h>
#include <xmmintrin.h>
//typedef unsigned __int64 BPSeed;
#include <omp.h>
#include<bitset>
#include <sstream>

#define numOfVertices SP_Constants::numOfVertices
#define numOfEdges SP_Constants::numOfEdges
#define INF_WEIGHT SP_Constants::INF_WEIGHT
#define AVG_LABEL_SIZE 200

struct index_pivot {
	int* pointNo;
	int pointNum;
};//每个pivot对应的所有结点编号的数据结构 modified by wanjingyi

struct index_t {
	vector<NodeID> spt_v;
	vector<EdgeWeight> spt_d;

	NodeID size() {
		return spt_v.size();
	}

};

struct index_t_p {
	NodeID* spt_v;
	EdgeWeight* spt_d;
}__attribute__((aligned(64)));  // Aligned for cache lines;


struct two_index_t_p {
	NodeID* spt_v;
	EdgeWeight* spt_d;
	uint8_t* spt_lv;
	EdgeWeight* spt_ld;
}__attribute__((aligned(64)));  // Aligned for cache lines;

struct index_t_path {
	vector<NodeID> spt_v;
	vector<NodeID> spt_p;//parent nodes
	vector<EdgeWeight> spt_d;

	NodeID size() {
		return spt_v.size();
	}

};

struct index_t_path_p {
	NodeID* spt_v;
	NodeID* spt_p;
	EdgeWeight* spt_d;
};

struct query_info {
	NodeID meet_node;
	NodeID search_len;
	double time_cost;
	EdgeWeight distance;
};

template<int kNumBitParallelRoots = 50>
struct index_t_bp {
	NodeID* spt_v;
	EdgeWeight* spt_d;
	EdgeWeight bpspt_d[kNumBitParallelRoots];
	uint64_t bpspt_s[kNumBitParallelRoots][2];
}__attribute__((aligned(64)));  // Aligned for cache lines;


struct token_t {
	NodeID* sptc_v; // sptc_v[0] is the root
	EdgeWeight* sptc_d;	 // |*| = k + 1, sptc_d[0] is the number of children - k
	unsigned char* sptc_fbv; // first-level bit vector
	unsigned char* sptc_sbv; // second-level bit vector
	NodeID* sptc_pathv; // intermediate point for a path
}__attribute__((aligned(64)));

//*************written by wanjingyi***************
struct queryPair{
	NodeID s,t;
	int freq;
	queryPair(NodeID s1,NodeID t1,int f) : s(s1), t(t1), freq(f){ }
	bool operator < (const queryPair& qp){
		if(qp.freq!=freq) return freq<qp.freq;
		else if(qp.s!=s) return s<qp.s;
		else return t<qp.t;
	}
};

bool cmp(pair<NodeID, int> a, pair<NodeID, int> b)
{
	if(a.second!=b.second) return a.second<b.second;//根据second的值升序排序
	else return a.first<b.first;
}

class CLabel {

public:
	token_t* supertokenindex_p;
	token_t* tokenindex_p;
	NodeID* anchor_p;
	NodeID numOfTokens;
	long total_children;
	
	
	token_t* r_supertokenindex_p;
	token_t* r_tokenindex_p;
	NodeID* r_anchor_p;
	NodeID r_numOfTokens;
	long r_total_children;
	
	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&anchor_p[v], sizeof(anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		ofs.write((const char*)&numOfTokens, sizeof(numOfTokens));
		for (NodeID t = 0; t < numOfTokens; ++t) {
			token_t& tt = tokenindex_p[t];
			EdgeWeight tsize = tt.sptc_d[0];
			ofs.write((const char*)&tt.sptc_v[0], sizeof(tt.sptc_v[0]));
			ofs.write((const char*)&tsize, sizeof(tsize));
			for(NodeID c = 0; c < tsize; ++c){				
				ofs.write((const char*)&tt.sptc_v[1 + c], sizeof(tt.sptc_v[1 + c]));
				ofs.write((const char*)&tt.sptc_d[1 + c], sizeof(tt.sptc_d[1 + c]));
			} 
		} 
		
		ofs.close();
	}
	
	
	void save_labels_path(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&anchor_p[v], sizeof(anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		ofs.write((const char*)&numOfTokens, sizeof(numOfTokens));
		for (NodeID t = 0; t < numOfTokens; ++t) {
			token_t& tt = tokenindex_p[t];
			EdgeWeight tsize = tt.sptc_d[0];
			ofs.write((const char*)&tt.sptc_v[0], sizeof(tt.sptc_v[0]));
			ofs.write((const char*)&tsize, sizeof(tsize));
			for(NodeID c = 0; c < tsize; ++c){				
				ofs.write((const char*)&tt.sptc_v[1 + c], sizeof(tt.sptc_v[1 + c]));
				ofs.write((const char*)&tt.sptc_d[1 + c], sizeof(tt.sptc_d[1 + c]));
				ofs.write((const char*)&tt.sptc_pathv[1 + c], sizeof(tt.sptc_pathv[1 + c]));
			} 
		} 
		
		ofs.close();
	}
	
	void save_labels_d(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&anchor_p[v], sizeof(anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&r_anchor_p[v], sizeof(r_anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		ofs.write((const char*)&numOfTokens, sizeof(numOfTokens));
		for (NodeID t = 0; t < numOfTokens; ++t) {
			token_t& tt = tokenindex_p[t];
			EdgeWeight tsize = tt.sptc_d[0];
			ofs.write((const char*)&tt.sptc_v[0], sizeof(tt.sptc_v[0]));
			ofs.write((const char*)&tsize, sizeof(tsize));
			for(NodeID c = 0; c < tsize; ++c){				
				ofs.write((const char*)&tt.sptc_v[1 + c], sizeof(tt.sptc_v[1 + c]));
				ofs.write((const char*)&tt.sptc_d[1 + c], sizeof(tt.sptc_d[1 + c]));
			} 
		}
		
		ofs.write((const char*)&r_numOfTokens, sizeof(r_numOfTokens));
		for (NodeID t = 0; t < r_numOfTokens; ++t) {
			token_t& tt = r_tokenindex_p[t];
			EdgeWeight tsize = tt.sptc_d[0];
			ofs.write((const char*)&tt.sptc_v[0], sizeof(tt.sptc_v[0]));
			ofs.write((const char*)&tsize, sizeof(tsize));
			for(NodeID c = 0; c < tsize; ++c){				
				ofs.write((const char*)&tt.sptc_v[1 + c], sizeof(tt.sptc_v[1 + c]));
				ofs.write((const char*)&tt.sptc_d[1 + c], sizeof(tt.sptc_d[1 + c]));
			} 
		}
		
		ofs.close();
	}
	
	
	void load_labels_path(const char* load_filename) {
		
		total_children = 0;
		
		tokenindex_p = NULL;
		anchor_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		NodeID anchor_id;
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				anchor_p[v] = anchor_id;
		}
		
		ifs.read((char*)&isize, sizeof(isize));
		numOfTokens = isize;		
		tokenindex_p = (token_t*)memalign(64, numOfTokens * sizeof(token_t));
		
		EdgeWeight csize;
		NodeID cid;
		EdgeWeight cd;
		for (NodeID v = 0; v < numOfTokens; ++v) {
			token_t& tt = tokenindex_p[v];
		
			ifs.read((char*)&cid, sizeof(cid));
			ifs.read((char*)&csize, sizeof(csize));
			
			tt.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
			
			total_children += (csize + 1);
			
			tt.sptc_v[0] = cid;
			tt.sptc_d[0] = csize;
			for (NodeID i = 0; i < csize; ++i) {
				ifs.read((char*)&cid, sizeof(cid));
				ifs.read((char*)&cd, sizeof(cd));
				tt.sptc_v[i + 1] = cid;
				tt.sptc_d[i + 1] = cd;			
			}
		}
		ifs.close();
	}
	
	void load_labels(const char* load_filename) {
		
		total_children = 0;
		
		tokenindex_p = NULL;
		anchor_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		NodeID anchor_id;
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				anchor_p[v] = anchor_id;
		}
		
		ifs.read((char*)&isize, sizeof(isize));
		numOfTokens = isize;		
		tokenindex_p = (token_t*)memalign(64, numOfTokens * sizeof(token_t));
		
		EdgeWeight csize;
		NodeID cid;
		EdgeWeight cd;
		for (NodeID v = 0; v < numOfTokens; ++v) {
			token_t& tt = tokenindex_p[v];
		
			ifs.read((char*)&cid, sizeof(cid));
			ifs.read((char*)&csize, sizeof(csize));
			
			tt.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
			
			total_children += (csize + 1);
			
			tt.sptc_v[0] = cid;
			tt.sptc_d[0] = csize;
			
			for (NodeID i = 0; i < csize; ++i) {
				ifs.read((char*)&cid, sizeof(cid));
				ifs.read((char*)&cd, sizeof(cd));
				tt.sptc_v[i + 1] = cid;
				tt.sptc_d[i + 1] = cd;		
			}
		}
		ifs.close();
	}
	
	
	void load_labels_d(const char* load_filename) {
		
		total_children = 0;
		r_total_children = 0;
		
		tokenindex_p = NULL;
		anchor_p = NULL;

		r_tokenindex_p = NULL;
		r_anchor_p = NULL;
		
		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		r_anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		NodeID anchor_id;
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				anchor_p[v] = anchor_id;
		}
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				r_anchor_p[v] = anchor_id;
		}
		 
		ifs.read((char*)&isize, sizeof(isize));
		numOfTokens = isize;		
		tokenindex_p = (token_t*)memalign(64, numOfTokens * sizeof(token_t));
		
		EdgeWeight csize;
		NodeID cid;
		EdgeWeight cd;
		for (NodeID v = 0; v < numOfTokens; ++v) {
			token_t& tt = tokenindex_p[v];
		
			ifs.read((char*)&cid, sizeof(cid));
			ifs.read((char*)&csize, sizeof(csize));
			
			tt.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
			
			total_children += (csize + 1);
			
			tt.sptc_v[0] = cid;
			tt.sptc_d[0] = csize;
			for (NodeID i = 0; i < csize; ++i) {
				ifs.read((char*)&cid, sizeof(cid));
				ifs.read((char*)&cd, sizeof(cd));
				tt.sptc_v[i + 1] = cid;
				tt.sptc_d[i + 1] = cd;			
			}
		}
		
		ifs.read((char*)&isize, sizeof(isize));
		r_numOfTokens = isize;		
		r_tokenindex_p = (token_t*)memalign(64, r_numOfTokens * sizeof(token_t));
		
		for (NodeID v = 0; v < r_numOfTokens; ++v) {
			token_t& tt = r_tokenindex_p[v];
		 
			ifs.read((char*)&cid, sizeof(cid));
			ifs.read((char*)&csize, sizeof(csize));
			
			tt.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
			
			r_total_children += (csize + 1);
			
			tt.sptc_v[0] = cid;
			tt.sptc_d[0] = csize;
			for (NodeID i = 0; i < csize; ++i) {
				ifs.read((char*)&cid, sizeof(cid));
				ifs.read((char*)&cd, sizeof(cd));
				tt.sptc_v[i + 1] = cid;
				tt.sptc_d[i + 1] = cd;			
			}
		}
		cout << "finish loading" << endl;
		ifs.close();
	}
	
	 
	void print_stat() {
		cout << "Total Token #: " << numOfTokens << endl;
		cout << "Average Children (Super) Token #: " << (double)total_children/(double)numOfTokens << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}
	
	void print_stat_d() {
		cout << "Total Token #: " << numOfTokens << endl;
		cout << "Total r_Token #: " << r_numOfTokens << endl;
		cout << "Average Children (Super) Token #: " << (double)total_children/(double)numOfTokens << endl;
		cout << "Average Children (Super) Token #: " << (double)r_total_children/(double)r_numOfTokens << endl;
	//	cout << "Maximum Label Size: " << max_size() << endl;
	} 
	
	EdgeWeight query_p(NodeID s, NodeID t, long ts, vector<NodeID>& dis_vec, vector<long>& ts_vec, vector<NodeID>& que, vector<EdgeWeight>& que_d) {
		if(s==t) return 0;
		
		EdgeWeight distance = INF_WEIGHT;
		
		NodeID anchor_s = anchor_p[s];
		NodeID anchor_t = anchor_p[t];

		NodeID que_t0 = 0, que_t1 = 0, que_h = 0;
		
		que_d[que_h] = 0;
		que[que_h++] = anchor_s;
		que_t1 = que_h;
		
		if(anchor_s < numOfVertices){
			if(ts_vec[anchor_s] != ts){
				ts_vec[anchor_s] = ts;
				dis_vec[anchor_s] = 0;
			}			
		}
		else{
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					NodeID r = token_v.sptc_v[0];
					EdgeWeight csize = token_v.sptc_d[0];
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] != ts){
						ts_vec[r] = ts;
						dis_vec[r] = tdis;
					}
					
					for (EdgeWeight i = 0; i < csize; ++i){
						NodeID w = token_v.sptc_v[i+1];
						EdgeWeight w_d = token_v.sptc_d[i+1] + tdis;
						if( w < numOfVertices){// hashing, can be replaced by 1024 linear probing for efficiency.
							if(ts_vec[w] != ts){
								ts_vec[w] = ts;
								dis_vec[w] = w_d;
							}
						}else{
							que_d[que_h] = w_d;
							que[que_h++] = w;
						}
					}
				}
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}
		
		
		que_t0 = 0, que_t1 = 0, que_h = 0;
		que_d[que_h] = 0;
		que[que_h++] = anchor_t;
		
		if(anchor_t < numOfVertices){
			if(ts_vec[anchor_t] == ts){
				EdgeWeight current_dis = dis_vec[anchor_t] + 0;
				if(current_dis < distance)
					distance = current_dis;
			}
		}else{
			que_t1 = que_h;
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					NodeID r = token_v.sptc_v[0];
					EdgeWeight csize = token_v.sptc_d[0];
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] == ts){
						EdgeWeight current_dis = dis_vec[r] + tdis;
						if(current_dis < distance)
							distance = current_dis;
					}
					
					for (EdgeWeight i = 0; i < csize; ++i){
						NodeID w = token_v.sptc_v[i+1];
						EdgeWeight w_d = token_v.sptc_d[i+1] + tdis;
						if( w < numOfVertices){
						// hashing, can be replaced by 1024 linear probing for efficiency.
							if(ts_vec[w] == ts){
								EdgeWeight current_dis = dis_vec[w] + w_d;
								if(current_dis < distance)
									distance = current_dis;
							}
						}else{							
							que_d[que_h] = w_d;
							que[que_h++] = w;
						}
					}
				}
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}		
		return distance;
	}
	
	EdgeWeight query_p_d(NodeID s, NodeID t, long ts, vector<NodeID>& dis_vec, vector<long>& ts_vec, vector<NodeID>& que, vector<EdgeWeight>& que_d) {
		if(s==t) return 0;
		
		EdgeWeight distance = INF_WEIGHT;
		
		NodeID anchor_s = anchor_p[s];
		NodeID anchor_t = r_anchor_p[t];

		NodeID que_t0 = 0, que_t1 = 0, que_h = 0;
		
		que_d[que_h] = 0;
		que[que_h++] = anchor_s;
		que_t1 = que_h;
		
		if(anchor_s < numOfVertices){
			if(ts_vec[anchor_s] != ts){
				ts_vec[anchor_s] = ts;
				dis_vec[anchor_s] = 0;
			}			
		}
		else{
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					NodeID r = token_v.sptc_v[0];
					EdgeWeight csize = token_v.sptc_d[0];
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] != ts){
						ts_vec[r] = ts;
						dis_vec[r] = tdis;
					}
					
					for (EdgeWeight i = 0; i < csize; ++i){
						NodeID w = token_v.sptc_v[i+1];
						EdgeWeight w_d = token_v.sptc_d[i+1] + tdis;
						if( w < numOfVertices){// hashing, can be replaced by 1024 linear probing for efficiency.
							if(ts_vec[w] != ts){
								ts_vec[w] = ts;
								dis_vec[w] = w_d;
							}
						}else{
							que_d[que_h] = w_d;
							que[que_h++] = w;
						}
					}
				}
				que_t0 = que_t1;
				que_t1 = que_h;
			} 
		}
		
		
		que_t0 = 0, que_t1 = 0, que_h = 0;
		que_d[que_h] = 0;
		que[que_h++] = anchor_t;
		
		if(anchor_t < numOfVertices){
			if(ts_vec[anchor_t] == ts){
				EdgeWeight current_dis = dis_vec[anchor_t] + 0;
				if(current_dis < distance)
					distance = current_dis;
			}
		}else{
			que_t1 = que_h;
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = r_tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					NodeID r = token_v.sptc_v[0];
					EdgeWeight csize = token_v.sptc_d[0];
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] == ts){
						EdgeWeight current_dis = dis_vec[r] + tdis;
						if(current_dis < distance)
							distance = current_dis;
					}
					
					for (EdgeWeight i = 0; i < csize; ++i){
						NodeID w = token_v.sptc_v[i+1];
						EdgeWeight w_d = token_v.sptc_d[i+1] + tdis;
						if( w < numOfVertices){
						// hashing, can be replaced by 1024 linear probing for efficiency.
							if(ts_vec[w] == ts){
								EdgeWeight current_dis = dis_vec[w] + w_d;
								if(current_dis < distance)
									distance = current_dis;
							}
						}else{							
							que_d[que_h] = w_d;
							que[que_h++] = w;
						}
					}
				}
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}
		
		
		
		return distance;
	}
	
	
	void save_two_level_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&anchor_p[v], sizeof(anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		// Store supertokens 
		for (NodeID v = 0; v < numOfVertices; ++v) {
			token_t& supertoken_v = supertokenindex_p[v];
			
			NodeID isize = supertoken_v.sptc_v[0];			
			ofs.write((const char*)&isize, sizeof(isize));
			
			for(NodeID i = 0; i < isize; ++i){
				NodeID tid = supertoken_v.sptc_v[i + 1];
				EdgeWeight ew = supertoken_v.sptc_d[i + 1];	
				
				ofs.write((const char*)&tid, sizeof(tid));
				ofs.write((const char*)&ew, sizeof(ew));
				
			}			
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		// Store normal tokens
		ofs.write((const char*)&numOfTokens, sizeof(numOfTokens));
		for (NodeID t = 0; t < numOfTokens; ++t) {
			token_t& tt = tokenindex_p[t];
			NodeID sid = tt.sptc_v[0];
			EdgeWeight ssize = tt.sptc_d[0];
			EdgeWeight fsize = supertokenindex_p[sid].sptc_d[0];
			ofs.write((const char*)&sid, sizeof(sid));			
			ofs.write((const char*)&ssize, sizeof(ssize));	
			if(ssize == 0) continue;
			//ofs.write((const char*)&fsize, sizeof(fsize));
			//if(t < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			for(NodeID c = 0; c < fsize; ++c){		
				//char a = tt.sptc_fbv[c];
				//ofs.write((const char*)&a, sizeof(a));
				ofs.write((const char*)&tt.sptc_fbv[c], sizeof(tt.sptc_fbv[c]));
			//	if(t < 10){
		//			bitset<8> s(tt.sptc_fbv[c]);
			//		cout << s;
			//	}
			} 
			//if(t < 10)
			//	cout << endl;
					
			for(NodeID c = 0; c < ssize; ++c){			
				//char a = tt.sptc_sbv[c];
				//ofs.write((const char*)&a, sizeof(a));	
				ofs.write((const char*)&tt.sptc_sbv[c], sizeof(tt.sptc_sbv[c]));
			//	if(t < 10){
			//		bitset<8> s(tt.sptc_sbv[c]);
			//		cout << s;
			//	}
			} 
			//if(t < 10)
			//	cout << endl;
		}
		
		ofs.close();
	}
	 
	void load_two_level_labels(const char* load_filename) {
		total_children = 0;
		
		tokenindex_p = NULL;
		anchor_p = NULL;
		supertokenindex_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		NodeID anchor_id;
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				anchor_p[v] = anchor_id;
		}
		
		//load supertokens
		NodeID cid;
		EdgeWeight cd;
		supertokenindex_p = (token_t*)memalign(64, numOfVertices * sizeof(token_t));
		for (NodeID v = 0; v < numOfVertices; ++v) {
				
				token_t& supertoken_v = supertokenindex_p[v];					
			
				NodeID csize;
				ifs.read((char*)&csize, sizeof(csize));
				
				supertoken_v.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
				supertoken_v.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
				
				supertoken_v.sptc_v[0] = csize;
				
				NodeID intsize = ceil((double)ceil((double)csize / (double)8) / (double)8);				
				supertoken_v.sptc_d[0] = intsize;
				
				total_children += csize;
				for(EdgeWeight i = 0; i < csize; ++i){
					ifs.read((char*)&cid, sizeof(cid));
					ifs.read((char*)&cd, sizeof(cd));
					supertoken_v.sptc_v[i + 1] = cid;
					supertoken_v.sptc_d[i + 1] = cd;	
				}				
		}
		cout << "loaded supertokens" << endl;
		cout << "Average Children Super Token #: " << (double)total_children/(double)numOfVertices << endl;
		
		ifs.read((char*)&isize, sizeof(isize));
		numOfTokens = isize;			
		NodeID sid;
		EdgeWeight ssize;
		EdgeWeight fsize;
		cout<< numOfTokens << " tokens in total." << endl;
		
		tokenindex_p = (token_t*)memalign(64, numOfTokens * sizeof(token_t));	
		for (NodeID v = 0; v < numOfTokens; ++v) {
			token_t& tt = tokenindex_p[v];
		
			ifs.read((char*)&sid, sizeof(sid));
			ifs.read((char*)&ssize, sizeof(ssize));
			
			tt.sptc_v = (NodeID*)memalign(64, 1 * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, 1 * sizeof(EdgeWeight));
			
			tt.sptc_v[0] = sid;
			tt.sptc_d[0] = ssize;			
			fsize = supertokenindex_p[sid].sptc_d[0];			
			
			if(ssize == 0) continue;
			//if(v < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			tt.sptc_fbv = (unsigned char*)memalign(64, fsize * sizeof(unsigned char));		
		//	unsigned char fb;
			char fb;
			for (NodeID i = 0; i < fsize; ++i) {
				ifs.read((char*)&(tt.sptc_fbv[i]), sizeof(tt.sptc_fbv[i]));
				//ifs.read((char*)&fb, sizeof(fb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_fbv[i]);
			//		cout << s;
			//	}
			}
			//if(v < 10)
			//	cout << endl;
			
			  
			tt.sptc_sbv = (unsigned char*)memalign(64, ssize * sizeof(unsigned char));		
			//unsigned char sb;
			char sb;
			for (NodeID i = 0; i < ssize; ++i) {
				ifs.read((char*)&(tt.sptc_sbv[i]), sizeof(tt.sptc_sbv[i]));
				//ifs.read((char*)&sb, sizeof(sb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_sbv[i]);
			//		cout << s;
				//}
			}
			//if(v < 10)
			//	cout << endl;
			//
		}
		cout << "loaded standard tokens" << endl;
		ifs.close();
	} 
	
	void save_two_level_labels_path(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&anchor_p[v], sizeof(anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		// Store supertokens 
		for (NodeID v = 0; v < numOfVertices; ++v) {
			token_t& supertoken_v = supertokenindex_p[v];
			
			NodeID isize = supertoken_v.sptc_v[0];			
			ofs.write((const char*)&isize, sizeof(isize));
			
			for(NodeID i = 0; i < isize; ++i){
				NodeID tid = supertoken_v.sptc_v[i + 1];
				EdgeWeight ew = supertoken_v.sptc_d[i + 1];	
				NodeID pid = supertoken_v.sptc_pathv[i + 1];
				
				ofs.write((const char*)&tid, sizeof(tid));
				ofs.write((const char*)&ew, sizeof(ew));
				ofs.write((const char*)&pid, sizeof(pid));
				
			}			
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		
		// Store normal tokens
		ofs.write((const char*)&numOfTokens, sizeof(numOfTokens));
		for (NodeID t = 0; t < numOfTokens; ++t) {
			token_t& tt = tokenindex_p[t];
			NodeID sid = tt.sptc_v[0];
			EdgeWeight ssize = tt.sptc_d[0];
			EdgeWeight fsize = supertokenindex_p[sid].sptc_d[0];
			ofs.write((const char*)&sid, sizeof(sid));			
			ofs.write((const char*)&ssize, sizeof(ssize));	
			if(ssize == 0) continue;
			//ofs.write((const char*)&fsize, sizeof(fsize));
			//if(t < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			for(NodeID c = 0; c < fsize; ++c){		
				//char a = tt.sptc_fbv[c];
				//ofs.write((const char*)&a, sizeof(a));
				ofs.write((const char*)&tt.sptc_fbv[c], sizeof(tt.sptc_fbv[c]));
			//	if(t < 10){
		//			bitset<8> s(tt.sptc_fbv[c]);
			//		cout << s;
			//	}
			} 
			//if(t < 10)
			//	cout << endl;
					
			for(NodeID c = 0; c < ssize; ++c){			
				//char a = tt.sptc_sbv[c];
				//ofs.write((const char*)&a, sizeof(a));	
				ofs.write((const char*)&tt.sptc_sbv[c], sizeof(tt.sptc_sbv[c]));
			//	if(t < 10){
			//		bitset<8> s(tt.sptc_sbv[c]);
			//		cout << s;
			//	}
			} 
			//if(t < 10)
			//	cout << endl;
		}
		
		ofs.close();
	}
	
	
	void load_two_level_labels_path(const char* load_filename) {
		total_children = 0;
		
		tokenindex_p = NULL;
		anchor_p = NULL;
		supertokenindex_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		NodeID anchor_id;
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				anchor_p[v] = anchor_id;
		}
		
		//load supertokens
		NodeID cid;
		EdgeWeight cd;
		supertokenindex_p = (token_t*)memalign(64, numOfVertices * sizeof(token_t));
		for (NodeID v = 0; v < numOfVertices; ++v) {
				
				token_t& supertoken_v = supertokenindex_p[v];					
			
				NodeID csize;
				ifs.read((char*)&csize, sizeof(csize));
				
				supertoken_v.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
				supertoken_v.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
				supertoken_v.sptc_pathv = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
				
				supertoken_v.sptc_v[0] = csize;
				
				NodeID intsize = ceil((double)ceil((double)csize / (double)8) / (double)8);				
				supertoken_v.sptc_d[0] = intsize;
				
				supertoken_v.sptc_pathv[0] = numOfVertices;
				
				total_children += csize;
				for(EdgeWeight i = 0; i < csize; ++i){
					ifs.read((char*)&cid, sizeof(cid));
					ifs.read((char*)&cd, sizeof(cd));
					supertoken_v.sptc_v[i + 1] = cid;
					supertoken_v.sptc_d[i + 1] = cd;	
					ifs.read((char*)&cid, sizeof(cid));
					supertoken_v.sptc_pathv[i + 1] = cid;
				}				
		}
		cout << "loaded supertokens" << endl;
		cout << "Average Children Super Token #: " << (double)total_children/(double)numOfVertices << endl;
		
		ifs.read((char*)&isize, sizeof(isize));
		numOfTokens = isize;			
		NodeID sid;
		EdgeWeight ssize;
		EdgeWeight fsize;
		cout<< numOfTokens << " tokens in total." << endl;
		
		tokenindex_p = (token_t*)memalign(64, numOfTokens * sizeof(token_t));	
		for (NodeID v = 0; v < numOfTokens; ++v) {
			token_t& tt = tokenindex_p[v];
		
			ifs.read((char*)&sid, sizeof(sid));
			ifs.read((char*)&ssize, sizeof(ssize));
			
			tt.sptc_v = (NodeID*)memalign(64, 1 * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, 1 * sizeof(EdgeWeight));
			
			tt.sptc_v[0] = sid;
			tt.sptc_d[0] = ssize;			
			fsize = supertokenindex_p[sid].sptc_d[0];			
			
			if(ssize == 0) continue;
			//if(v < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			tt.sptc_fbv = (unsigned char*)memalign(64, fsize * sizeof(unsigned char));		
		//	unsigned char fb;
			char fb;
			for (NodeID i = 0; i < fsize; ++i) {
				ifs.read((char*)&(tt.sptc_fbv[i]), sizeof(tt.sptc_fbv[i]));
				//ifs.read((char*)&fb, sizeof(fb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_fbv[i]);
			//		cout << s;
			//	}
			}
			//if(v < 10)
			//	cout << endl;
			
			  
			tt.sptc_sbv = (unsigned char*)memalign(64, ssize * sizeof(unsigned char));		
			//unsigned char sb;
			char sb;
			for (NodeID i = 0; i < ssize; ++i) {
				ifs.read((char*)&(tt.sptc_sbv[i]), sizeof(tt.sptc_sbv[i]));
				//ifs.read((char*)&sb, sizeof(sb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_sbv[i]);
			//		cout << s;
				//}
			}
			//if(v < 10)
			//	cout << endl;
			//
		}
		cout << "loaded standard tokens" << endl;
		ifs.close();
	} 
	
	void save_two_level_labels_d(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);
		//cout << "1" << endl;
		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&anchor_p[v], sizeof(anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs.write((const char*)&r_anchor_p[v], sizeof(r_anchor_p[v]));
		//	ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
		}
				
		
		// Store supertokens 
	//	cout << "2" << endl;
		for (NodeID v = 0; v < numOfVertices; ++v) {
			token_t& supertoken_v = supertokenindex_p[v];
			
			NodeID isize = supertoken_v.sptc_v[0];			
			ofs.write((const char*)&isize, sizeof(isize));
			
			for(NodeID i = 0; i < isize; ++i){
				NodeID tid = supertoken_v.sptc_v[i + 1];
				EdgeWeight ew = supertoken_v.sptc_d[i + 1];	
				
				ofs.write((const char*)&tid, sizeof(tid));
				ofs.write((const char*)&ew, sizeof(ew));
				
			}			
		}
		
		for (NodeID v = 0; v < numOfVertices; ++v) {
			token_t& supertoken_v = r_supertokenindex_p[v];
			
			NodeID isize = supertoken_v.sptc_v[0];			
			ofs.write((const char*)&isize, sizeof(isize));
			
			for(NodeID i = 0; i < isize; ++i){
				NodeID tid = supertoken_v.sptc_v[i + 1];
				EdgeWeight ew = supertoken_v.sptc_d[i + 1];	
				
				ofs.write((const char*)&tid, sizeof(tid));
				ofs.write((const char*)&ew, sizeof(ew));
				
			}			
		}
		
		 
		// Store normal tokens
		//cout << "3" << endl;
		ofs.write((const char*)&numOfTokens, sizeof(numOfTokens));
		for (NodeID t = 0; t < numOfTokens; ++t) {
			
		//	cout << "31:" << t << endl;
			token_t& tt = tokenindex_p[t]; 
			NodeID sid = tt.sptc_v[0];
			EdgeWeight ssize = tt.sptc_d[0];
			EdgeWeight fsize = supertokenindex_p[sid].sptc_d[0];
			ofs.write((const char*)&sid, sizeof(sid));			
			ofs.write((const char*)&ssize, sizeof(ssize));	
			
			
		//	cout << "32:" << t << endl;
			if(ssize == 0) continue;
			//ofs.write((const char*)&fsize, sizeof(fsize));
			//if(t < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			
		//	cout << "33:" << t << endl;
			for(NodeID c = 0; c < fsize; ++c){		
				//char a = tt.sptc_fbv[c];
				//ofs.write((const char*)&a, sizeof(a));
				ofs.write((const char*)&tt.sptc_fbv[c], sizeof(tt.sptc_fbv[c]));
			//	if(t < 10){
		//			bitset<8> s(tt.sptc_fbv[c]);
			//		cout << s;
			//	}
			} 
			//if(t < 10) 
			//	cout << endl;
					
		//	cout << "34:" << t << endl;
			for(NodeID c = 0; c < ssize; ++c){			
				//char a = tt.sptc_sbv[c];
				//ofs.write((const char*)&a, sizeof(a));	
				ofs.write((const char*)&tt.sptc_sbv[c], sizeof(tt.sptc_sbv[c]));
			//	if(t < 10){
			//		bitset<8> s(tt.sptc_sbv[c]);
			//		cout << s;
			//	}
			} 
			//if(t < 10)
			//	cout << endl;
		} 
		
		//cout << "4" << endl;
		ofs.write((const char*)&r_numOfTokens, sizeof(r_numOfTokens));
		for (NodeID t = 0; t < r_numOfTokens; ++t) {
					
			//cout << "41:" << t << endl;
			token_t& tt = r_tokenindex_p[t];
			NodeID sid = tt.sptc_v[0];
			EdgeWeight ssize = tt.sptc_d[0];
			EdgeWeight fsize = r_supertokenindex_p[sid].sptc_d[0];
			ofs.write((const char*)&sid, sizeof(sid));			
			ofs.write((const char*)&ssize, sizeof(ssize));	
			if(ssize == 0) continue; 
			//ofs.write((const char*)&fsize, sizeof(fsize));
			//if(t < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			//cout << "42:" << t << "," << fsize <<  endl;
			for(NodeID c = 0; c < fsize; ++c){		
				ofs.write((const char*)&tt.sptc_fbv[c], sizeof(tt.sptc_fbv[c]));
			} 

					
			//cout << "43:" << t << "," << ssize << endl;
			for(NodeID c = 0; c < ssize; ++c){		
				ofs.write((const char*)&tt.sptc_sbv[c], sizeof(tt.sptc_sbv[c]));	
			} 
		}	
		
		ofs.close();
	}
	 
	void load_two_level_labels_d(const char* load_filename) {
		total_children = 0;		
		tokenindex_p = NULL;
		anchor_p = NULL;
		supertokenindex_p = NULL;
		
		r_total_children = 0;		
		r_tokenindex_p = NULL;
		r_anchor_p = NULL;
		r_supertokenindex_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		r_anchor_p = (NodeID*)memalign(64, numOfVertices * sizeof(NodeID));
		NodeID anchor_id;
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				anchor_p[v] = anchor_id;
		}
		for (NodeID v = 0; v < numOfVertices; ++v) {
				ifs.read((char*)&anchor_id, sizeof(anchor_id));
				r_anchor_p[v] = anchor_id;
		}
		
		//load supertokens
		NodeID cid;
		EdgeWeight cd;
		supertokenindex_p = (token_t*)memalign(64, numOfVertices * sizeof(token_t));
		for (NodeID v = 0; v < numOfVertices; ++v) {
				
				token_t& supertoken_v = supertokenindex_p[v];					
			
				NodeID csize;
				ifs.read((char*)&csize, sizeof(csize));
				
				supertoken_v.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
				supertoken_v.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
				
				supertoken_v.sptc_v[0] = csize;
				
				NodeID intsize = ceil((double)ceil((double)csize / (double)8) / (double)8);				
				supertoken_v.sptc_d[0] = intsize;
				
				total_children += csize;
				for(EdgeWeight i = 0; i < csize; ++i){
					ifs.read((char*)&cid, sizeof(cid));
					ifs.read((char*)&cd, sizeof(cd));
					supertoken_v.sptc_v[i + 1] = cid;
					supertoken_v.sptc_d[i + 1] = cd;	
				}				
		}	
		
		r_supertokenindex_p = (token_t*)memalign(64, numOfVertices * sizeof(token_t));
		for (NodeID v = 0; v < numOfVertices; ++v) {
				
				token_t& supertoken_v = r_supertokenindex_p[v];					
			
				NodeID csize;
				ifs.read((char*)&csize, sizeof(csize));
				
				supertoken_v.sptc_v = (NodeID*)memalign(64, (csize + 1) * sizeof(NodeID));
				supertoken_v.sptc_d = (EdgeWeight*)memalign(64, (csize + 1 ) * sizeof(EdgeWeight));
				
				supertoken_v.sptc_v[0] = csize;
				
				NodeID intsize = ceil((double)ceil((double)csize / (double)8) / (double)8);				
				supertoken_v.sptc_d[0] = intsize;
				
				r_total_children += csize;
				for(EdgeWeight i = 0; i < csize; ++i){
					ifs.read((char*)&cid, sizeof(cid));
					ifs.read((char*)&cd, sizeof(cd));
					supertoken_v.sptc_v[i + 1] = cid;
					supertoken_v.sptc_d[i + 1] = cd;	
				}				
		}
		cout << "loaded supertokens" << endl;
		cout << "Average Children Super Token #: " << (double)total_children/(double)numOfVertices << endl;
		cout << "Average Children Super Token #: " << (double)r_total_children/(double)numOfVertices << endl;
	
	
		ifs.read((char*)&isize, sizeof(isize));
		numOfTokens = isize;			
		NodeID sid;
		EdgeWeight ssize;
		EdgeWeight fsize;
		cout<< numOfTokens << " tokens in total." << endl;		
		tokenindex_p = (token_t*)memalign(64, numOfTokens * sizeof(token_t));	
		for (NodeID v = 0; v < numOfTokens; ++v) {
			token_t& tt = tokenindex_p[v];
		
			ifs.read((char*)&sid, sizeof(sid));
			ifs.read((char*)&ssize, sizeof(ssize));
			
			tt.sptc_v = (NodeID*)memalign(64, 1 * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, 1 * sizeof(EdgeWeight));
			
			tt.sptc_v[0] = sid;
			tt.sptc_d[0] = ssize;			
			fsize = supertokenindex_p[sid].sptc_d[0];			
			
			if(ssize == 0) continue;
			//if(v < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			tt.sptc_fbv = (unsigned char*)memalign(64, fsize * sizeof(unsigned char));		
		//	unsigned char fb;
			char fb;
			for (NodeID i = 0; i < fsize; ++i) {
				ifs.read((char*)&(tt.sptc_fbv[i]), sizeof(tt.sptc_fbv[i]));
				//ifs.read((char*)&fb, sizeof(fb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_fbv[i]);
			//		cout << s;
			//	}
			}
			//if(v < 10)
			//	cout << endl;
			
			  
			tt.sptc_sbv = (unsigned char*)memalign(64, ssize * sizeof(unsigned char));		
			//unsigned char sb;
			char sb;
			for (NodeID i = 0; i < ssize; ++i) {
				ifs.read((char*)&(tt.sptc_sbv[i]), sizeof(tt.sptc_sbv[i]));
				//ifs.read((char*)&sb, sizeof(sb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_sbv[i]);
			//		cout << s;
				//}
			}
			//if(v < 10)
			//	cout << endl;
			//
		}
		
		ifs.read((char*)&isize, sizeof(isize));
		r_numOfTokens = isize;			
		cout<< r_numOfTokens << " tokens in total." << endl;		
		r_tokenindex_p = (token_t*)memalign(64, r_numOfTokens * sizeof(token_t));	
		for (NodeID v = 0; v < r_numOfTokens; ++v) {
			token_t& tt = r_tokenindex_p[v];
		
			ifs.read((char*)&sid, sizeof(sid));
			ifs.read((char*)&ssize, sizeof(ssize));
			
			tt.sptc_v = (NodeID*)memalign(64, 1 * sizeof(NodeID));
			tt.sptc_d = (EdgeWeight*)memalign(64, 1 * sizeof(EdgeWeight));
			
			tt.sptc_v[0] = sid;
			tt.sptc_d[0] = ssize;			
			fsize = r_supertokenindex_p[sid].sptc_d[0];			
			
			if(ssize == 0) continue;
			//if(v < 10) 
			//	cout << sid << "vs" << fsize << "vs" << ssize << endl;
			
			tt.sptc_fbv = (unsigned char*)memalign(64, fsize * sizeof(unsigned char));		
		//	unsigned char fb;
			char fb;
			for (NodeID i = 0; i < fsize; ++i) {
				ifs.read((char*)&(tt.sptc_fbv[i]), sizeof(tt.sptc_fbv[i]));
				//ifs.read((char*)&fb, sizeof(fb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_fbv[i]);
			//		cout << s;
			//	}
			}
			//if(v < 10)
			//	cout << endl;
			
			  
			tt.sptc_sbv = (unsigned char*)memalign(64, ssize * sizeof(unsigned char));		
			//unsigned char sb;
			char sb;
			for (NodeID i = 0; i < ssize; ++i) {
				ifs.read((char*)&(tt.sptc_sbv[i]), sizeof(tt.sptc_sbv[i]));
				//ifs.read((char*)&sb, sizeof(sb));
			//	if(v < 10){
			//		bitset<8> s(tt.sptc_sbv[i]);
			//		cout << s;
				//}
			}
			//if(v < 10)
			//	cout << endl;
			//
		}
		 
		cout << "loaded standard tokens" << endl;
		ifs.close();
	}
	 
	 

	EdgeWeight query_p_two_level(NodeID s, NodeID t, long ts, vector<NodeID>& dis_vec, vector<long>& ts_vec, vector<NodeID>& que, vector<EdgeWeight>& que_d) {
		if(s==t) return 0;
		
		EdgeWeight distance = INF_WEIGHT;
		
		NodeID anchor_s = anchor_p[s];
		NodeID anchor_t = anchor_p[t];

		NodeID que_t0 = 0, que_t1 = 0, que_h = 0;
		
		que_d[que_h] = 0;
		que[que_h++] = anchor_s;
		que_t1 = que_h;
		
		if(anchor_s < numOfVertices){
			if(ts_vec[anchor_s] != ts){
				ts_vec[anchor_s] = ts;
				dis_vec[anchor_s] = 0;
			}			
		}
		else{
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					NodeID r = token_v.sptc_v[0];					
					EdgeWeight ssize = token_v.sptc_d[0];
					
					token_t& supertoken_r = supertokenindex_p[r];
					EdgeWeight fsize = supertoken_r.sptc_d[0];
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] != ts){
						ts_vec[r] = ts;
						dis_vec[r] = tdis;
					}
					
					EdgeWeight spos = 0;
					
					for(EdgeWeight i = 0; i < fsize; ++i){
						unsigned char fmask = token_v.sptc_fbv[i];						
						bitset<8> fbs(fmask);
						for(NodeID j = 0; j < 8; ++j){							
							if(fbs[ 7 - j]){
								unsigned char smask = token_v.sptc_sbv[spos++];
								bitset<8> sbs(smask);
								for(NodeID k = 0; k < 8; ++k){
									if(sbs[7 - k]){
										NodeID w = supertoken_r.sptc_v[ (i * 8 + j) * 8 + k  +  1];
										EdgeWeight w_d = supertoken_r.sptc_d[(i * 8 + j) * 8 + k  +  1] + tdis;											
										if( w < numOfVertices){// hashing, can be replaced by 1024 linear probing for efficiency.
											if(ts_vec[w] != ts){
												ts_vec[w] = ts;
												dis_vec[w] = w_d;
											}
										}else{
											que_d[que_h] = w_d;
											que[que_h++] = w;
										}										
									}
								}
								//if(spos == ssize) break;
							}							
						}
						//if(spos == ssize) break;						
					}					
				}
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}
		
		
		que_t0 = 0, que_t1 = 0, que_h = 0;
		que_d[que_h] = 0;
		que[que_h++] = anchor_t;
		
		if(anchor_t < numOfVertices){
			if(ts_vec[anchor_t] == ts){
				EdgeWeight current_dis = dis_vec[anchor_t] + 0;
				if(current_dis < distance)
					distance = current_dis;
			}
		}else{
			que_t1 = que_h;
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					
					
					NodeID r = token_v.sptc_v[0];					
					EdgeWeight ssize = token_v.sptc_d[0];					
					token_t& supertoken_r = supertokenindex_p[r];
					EdgeWeight fsize = supertoken_r.sptc_d[0];
					
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] == ts){
						EdgeWeight current_dis = dis_vec[r] + tdis;
						if(current_dis < distance)
							distance = current_dis;
					}
					
					EdgeWeight spos = 0;
					
					for(EdgeWeight i = 0; i < fsize; ++i){
						unsigned char fmask = token_v.sptc_fbv[i];						
						bitset<8> fbs(fmask);
						for(NodeID j = 0; j < 8; ++j){							
							if(fbs[7 - j]){
								unsigned char smask = token_v.sptc_sbv[spos++];
								bitset<8> sbs(smask);
								for(NodeID k = 0; k < 8; ++k){
									if(sbs[7 - k]){
										NodeID w = supertoken_r.sptc_v[ (i * 8 + j) * 8 + k  +  1];
										EdgeWeight w_d = supertoken_r.sptc_d[(i * 8 + j) * 8 + k  +  1] + tdis;											
										if( w < numOfVertices){// hashing, can be replaced by 1024 linear probing for efficiency.
											if(ts_vec[w] == ts){
												EdgeWeight current_dis = dis_vec[w] + w_d;
												if(current_dis < distance)
													distance = current_dis;
											}
										}else{
											que_d[que_h] = w_d;
											que[que_h++] = w;
										}										
									}
								}
								//if(spos == ssize) break;
							}							
						}
						//if(spos == ssize) break;						
					}					
				} 
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}
		
		
		
		return distance;
	}
	 
	EdgeWeight query_p_two_level_d(NodeID s, NodeID t, long ts, vector<NodeID>& dis_vec, vector<long>& ts_vec, vector<NodeID>& que, vector<EdgeWeight>& que_d) {
		if(s==t) return 0;
		
		EdgeWeight distance = INF_WEIGHT;
		
		NodeID anchor_s = anchor_p[s];
		NodeID anchor_t = r_anchor_p[t];

		NodeID que_t0 = 0, que_t1 = 0, que_h = 0;
		
		que_d[que_h] = 0;
		que[que_h++] = anchor_s;
		que_t1 = que_h;
		
		if(anchor_s < numOfVertices){
			if(ts_vec[anchor_s] != ts){
				ts_vec[anchor_s] = ts;
				dis_vec[anchor_s] = 0;
			}			
		}
		else{
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					NodeID r = token_v.sptc_v[0];					
					EdgeWeight ssize = token_v.sptc_d[0];
					
					token_t& supertoken_r = supertokenindex_p[r];
					EdgeWeight fsize = supertoken_r.sptc_d[0];
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] != ts){
						ts_vec[r] = ts;
						dis_vec[r] = tdis;
					}
					
					EdgeWeight spos = 0;
					
					for(EdgeWeight i = 0; i < fsize; ++i){
						unsigned char fmask = token_v.sptc_fbv[i];						
						bitset<8> fbs(fmask);
						for(NodeID j = 0; j < 8; ++j){							
							if(fbs[ 7 - j]){
								unsigned char smask = token_v.sptc_sbv[spos++];
								bitset<8> sbs(smask);
								for(NodeID k = 0; k < 8; ++k){
									if(sbs[7 - k]){
										NodeID w = supertoken_r.sptc_v[ (i * 8 + j) * 8 + k  +  1];
										EdgeWeight w_d = supertoken_r.sptc_d[(i * 8 + j) * 8 + k  +  1] + tdis;											
										if( w < numOfVertices){// hashing, can be replaced by 1024 linear probing for efficiency.
											if(ts_vec[w] != ts){
												ts_vec[w] = ts;
												dis_vec[w] = w_d;
											}
										}else{
											que_d[que_h] = w_d;
											que[que_h++] = w;
										}										
									}
								}
								//if(spos == ssize) break;
							}							
						}
						//if(spos == ssize) break;						
					}					
				}
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}
		
		
		que_t0 = 0, que_t1 = 0, que_h = 0;
		que_d[que_h] = 0;
		que[que_h++] = anchor_t;
		
		if(anchor_t < numOfVertices){
			if(ts_vec[anchor_t] == ts){
				EdgeWeight current_dis = dis_vec[anchor_t] + 0;
				if(current_dis < distance)
					distance = current_dis;
			}
		}else{
			que_t1 = que_h;
			for (; que_t0 < que_h;) {
				for (NodeID que_i = que_t0; que_i < que_t1; ++que_i) {
					NodeID tid = que[que_i];
					EdgeWeight tdis = que_d[que_i];
					
					const token_t& token_v = r_tokenindex_p[tid - numOfVertices];
					
					_mm_prefetch(&token_v.sptc_v[0], _MM_HINT_T0);
					_mm_prefetch(&token_v.sptc_d[0], _MM_HINT_T0);
					
					
					
					NodeID r = token_v.sptc_v[0];					
					EdgeWeight ssize = token_v.sptc_d[0];					
					token_t& supertoken_r = r_supertokenindex_p[r];
					EdgeWeight fsize = supertoken_r.sptc_d[0];
					
					
					// hashing, can be replaced by 1024 linear probing for efficiency.
					if(ts_vec[r] == ts){
						EdgeWeight current_dis = dis_vec[r] + tdis;
						if(current_dis < distance)
							distance = current_dis;
					}
					
					EdgeWeight spos = 0;
					
					for(EdgeWeight i = 0; i < fsize; ++i){
						unsigned char fmask = token_v.sptc_fbv[i];						
						bitset<8> fbs(fmask);
						for(NodeID j = 0; j < 8; ++j){							
							if(fbs[7 - j]){
								unsigned char smask = token_v.sptc_sbv[spos++];
								bitset<8> sbs(smask);
								for(NodeID k = 0; k < 8; ++k){
									if(sbs[7 - k]){
										NodeID w = supertoken_r.sptc_v[ (i * 8 + j) * 8 + k  +  1];
										EdgeWeight w_d = supertoken_r.sptc_d[(i * 8 + j) * 8 + k  +  1] + tdis;											
										if( w < numOfVertices){// hashing, can be replaced by 1024 linear probing for efficiency.
											if(ts_vec[w] == ts){
												EdgeWeight current_dis = dis_vec[w] + w_d;
												if(current_dis < distance)
													distance = current_dis;
											}
										}else{
											que_d[que_h] = w_d;
											que[que_h++] = w;
										}										
									}
								}
								//if(spos == ssize) break;
							}							
						}
						//if(spos == ssize) break;						
					}					
				} 
				que_t0 = que_t1;
				que_t1 = que_h;
			}
		}
		
		
		
		return distance;
	}
	
	
};


class Label {

public:
	vector<index_t> index_;	
	index_t_p* index_p;
	two_index_t_p* two_index_p;


	double GetCurrentTimeSec() {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + tv.tv_usec * 1e-6;
	}


	Label() {
		index_.resize(numOfVertices);
	}

	~Label() {
		Free();
	}

	EdgeWeight query_p(NodeID s, NodeID t) {
		//
		//EdgeWeight distance = INF_WEIGHT;

		//NodeID *vs = index_p[s].spt_v;
		//NodeID *vt = index_p[t].spt_v;
		//EdgeWeight* ws = index_p[s].spt_d;
		//EdgeWeight* wt = index_p[t].spt_d;

		//_mm_prefetch(vs, _MM_HINT_T0);
		//_mm_prefetch(vt, _MM_HINT_T0);
		//_mm_prefetch(ws, _MM_HINT_T0);
		//_mm_prefetch(wt, _MM_HINT_T0);

		//for (unsigned i = 0, j = 0; ; ) {
		//	if (*(vs + i) == *(vt + j)) {
		//		if (*(vs + i) == numOfVertices) break;  // Sentinel
		//		EdgeWeight td = *(ws + i) + *(wt + j);
		//		if (td < distance) distance = td;
		//		++i;
		//		++j;
		//	}
		//	else {
		//		i += *(vs + i) < *(vt + j) ? 1 : 0;
		//		j += *(vs + i) > *(vt + j) ? 1 : 0;
		//	}
		//}
		//return distance;

		EdgeWeight distance = INF_WEIGHT;

		const index_t_p &idx_s = index_p[s];
		const index_t_p &idx_t = index_p[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) distance = td;
				++i;
				++j;
			} 
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}
		return distance;
	}

	EdgeWeight two_query_p_sequential(NodeID s, NodeID t) {
		
		EdgeWeight distance = INF_WEIGHT;
		EdgeWeight ldistance = INF_WEIGHT;

		const two_index_t_p &idx_s = two_index_p[s];
		const two_index_t_p &idx_t = two_index_p[t];

		_mm_prefetch(&idx_s.spt_lv[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_lv[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_ld[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_ld[0], _MM_HINT_T0);

		for (uint8_t i = 0, j = 0; ; ) {
			uint8_t uv8_1 = idx_s.spt_lv[i], uv8_2 = idx_t.spt_lv[j];

			if (uv8_1 == UCHAR_MAX) break;  // Sentinel

			if (uv8_1 == uv8_2) {
				EdgeWeight td = idx_s.spt_ld[i] + idx_t.spt_ld[j];
				if (td < ldistance) ldistance = td;
				++i;
				++j;
			}
			else {
				i += uv8_1 < uv8_2 ? 1 : 0;
				j += uv8_1 > uv8_2 ? 1 : 0;
			}
		}
	
		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) distance = td;
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}
		
		if(distance < ldistance) 
			return distance;
		else
			return ldistance;		
	}
	
	EdgeWeight two_query_p_parallel(NodeID s, NodeID t) {
		
		EdgeWeight distance = INF_WEIGHT;
		EdgeWeight ldistance = INF_WEIGHT;

		const two_index_t_p &idx_s = two_index_p[s];
		const two_index_t_p &idx_t = two_index_p[t];

		
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
				_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
				_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
				_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

				for (int i = 0, j = 0; ; ) {
					NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

					if (v1 == numOfVertices) break;  // Sentinel

					if (v1 == v2) {
						EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
						if (td < distance) distance = td;
						++i;
						++j;
					}
					else {
						i += v1 < v2 ? 1 : 0;
						j += v1 > v2 ? 1 : 0;
					}
				}
			}
			
			#pragma omp section
			{
				_mm_prefetch(&idx_s.spt_lv[0], _MM_HINT_T0);
				_mm_prefetch(&idx_t.spt_lv[0], _MM_HINT_T0);
				_mm_prefetch(&idx_s.spt_ld[0], _MM_HINT_T0);
				_mm_prefetch(&idx_t.spt_ld[0], _MM_HINT_T0);

				for (uint8_t i = 0, j = 0; ; ) {
					uint8_t uv8_1 = idx_s.spt_lv[i], uv8_2 = idx_t.spt_lv[j];

					if (uv8_1 == UCHAR_MAX) break;  // Sentinel

					if (uv8_1 == uv8_2) {
						EdgeWeight td = idx_s.spt_ld[i] + idx_t.spt_ld[j];
						if (td < ldistance) ldistance = td;
						++i;
						++j;
					}
					else {
						i += uv8_1 < uv8_2 ? 1 : 0;
						j += uv8_1 > uv8_2 ? 1 : 0;
					}
				}
			}
		}
		if(distance < ldistance) 
			return distance;
		else
			return ldistance;		
	}
	
	EdgeWeight query_p_with_nums(NodeID s, NodeID t, int k) {
		//
		//EdgeWeight distance = INF_WEIGHT;

		//NodeID *vs = index_p[s].spt_v;
		//NodeID *vt = index_p[t].spt_v;
		//EdgeWeight* ws = index_p[s].spt_d;
		//EdgeWeight* wt = index_p[t].spt_d;

		//_mm_prefetch(vs, _MM_HINT_T0);
		//_mm_prefetch(vt, _MM_HINT_T0);
		//_mm_prefetch(ws, _MM_HINT_T0);
		//_mm_prefetch(wt, _MM_HINT_T0);

		//for (unsigned i = 0, j = 0; ; ) {
		//	if (*(vs + i) == *(vt + j)) {
		//		if (*(vs + i) == numOfVertices) break;  // Sentinel
		//		EdgeWeight td = *(ws + i) + *(wt + j);
		//		if (td < distance) distance = td;
		//		++i;
		//		++j;
		//	}
		//	else {
		//		i += *(vs + i) < *(vt + j) ? 1 : 0;
		//		j += *(vs + i) > *(vt + j) ? 1 : 0;
		//	}
		//}
		//return distance;


		EdgeWeight distance = INF_WEIGHT;

		const index_t_p &idx_s = index_p[s];
		const index_t_p &idx_t = index_p[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
		int k1 = k, k2 = k;
		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) distance = td;
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}

			if (i > k1 || j > k2) break;
		}
		return distance;
	}


	EdgeWeight query(NodeID s, NodeID t) {
		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;

		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j]) 
				distance = min(distance, (EdgeWeight)(index_s_d[i++] + index_t_d[j++]));
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		}
		return distance;
	}
	
	

	EdgeWeight query(NodeID s, NodeID t, NodeID& meet, EdgeWeight& dis1, EdgeWeight& dis2) {
		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;
		meet = numeric_limits<NodeID>::max();
		dis1 = numeric_limits<EdgeWeight>::max();
		dis2 = numeric_limits<EdgeWeight>::max();
		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j]) {
				if (distance > (EdgeWeight)(index_s_d[i] + index_t_d[j])) {
					distance = (EdgeWeight)(index_s_d[i] + index_t_d[j]);
					meet = index_s[i];
					dis1 = index_s_d[i];
					dis2 = index_t_d[j];
				}
				++i; ++j;
			}
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		}
		return distance;
	}

	/*EdgeWeight query_new(NodeID s, NodeID t, Ordering& ordering) {
		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;



		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j])
				distance = min(distance, (EdgeWeight)(index_s_d[i++] + index_t_d[j++]));
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		}
		return distance;
	}
	*/
	double avg_size() {
		
			double total = 0;
		if(index_.size()!=0){
			for (int i = 0; i < numOfVertices; ++i) total += index_[i].spt_v.size();

			double avg = total / numOfVertices - 1; // We do not count the trivial label (V, INF_WEIGHT).
			return avg;
		}
		
		total = 0;
		for (int i = 0; i < numOfVertices; ++i) {
			int unit_count = 0;
			const index_t_p &idx_s = index_p[i];
			for(int j = 0; ;){
				NodeID v = idx_s.spt_v[j++];
				++unit_count;
				if( v == numOfVertices) break;
			}
			total += unit_count;
		}

		double avg = total / numOfVertices - 1; // We do not count the trivial label (V, INF_WEIGHT).

		return avg;
	}
	/*
	NodeID max_size() {
		NodeID maxsize = numeric_limits<NodeID>::min();
		for (int i = 0; i < V; ++i) maxsize = max(maxsize, index_[i].spt_v.size());
		return maxsize;
	}*/

	void append(NodeID v, NodeID root, EdgeWeight distance) {
		index_[v].spt_v.push_back(root);
		index_[v].spt_d.push_back(distance);
	}

	void print_stat() {
		cout << "Average Label Size: " << avg_size() << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}

	void Free() {
		if (index_.size() == 0) return;
		for (int v = 0; v < numOfVertices; ++v) {
			index_[v].spt_v.clear();
			index_[v].spt_d.clear();
		}
		index_.clear();
	}

		 /**
		  * function used to serialize vertice's labels to file,format as follows:
		  * label_size v (w0,d(v,w0)) (w1,d(v,w1))...(wn,d(v,wn))
		  * written by wanjingyi
		  * */
		 void write_labels(const char* write_filename)
		 {
			 ofstream ofs(write_filename);
			for (NodeID v = 0; v < numOfVertices; ++v) 
			{
				NodeID isize = index_[v].size();
				ofs <<isize<<" "<<v;
				for (NodeID i = 0; i < index_[v].size(); ++i) {
					ofs<<" "<<'('<<index_[v].spt_v[i]<<","<<index_[v].spt_d[i]<<")";
				}
				ofs<<endl;
			}
			ofs.close();
		 }

	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			NodeID isize = index_[v].size();
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				ofs.write((const char*)&index_[v].spt_v[i], sizeof(index_[v].spt_v[i]));
				ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
			}
		}
		ofs.close();
	}
	
	void load_labels(const char* load_filename) {
	/*	for (NodeID v = 0; v < numOfVertices; ++v) {
			free(index_p[v].spt_v);
			free(index_p[v].spt_d);
		} 
		*/
		//free(index_p);
		index_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		cout<<"numOfVertices = "<<numOfVertices<<endl;

		index_p = (index_t_p*)memalign(64, numOfVertices * sizeof(index_t_p));
		


		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_p &idx = index_p[v];
			ifs.read((char*)&isize, sizeof(isize));
			
			idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));

		//	index_[v].spt_v.resize(isize);
		//	index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_v[i] = hub;
				idx.spt_d[i] = hub_weight;

			}
		}
		ifs.close();

		/*
		index_.clear();
		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		index_.resize(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {

			ifs.read((char*)&isize, sizeof(isize));
			index_[v].spt_v.resize(isize);
			index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < index_[v].size(); ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				index_[v].spt_v[i] = hub;
				index_[v].spt_d[i] = hub_weight;
			}
		}
		ifs.close();
		*/
	}

	void convert_to_fewerbit(){
		
		two_index_p = NULL;
		two_index_p = (two_index_t_p*)memalign(64, numOfVertices * sizeof(two_index_t_p));

		double compressed_size = 0;
		double total_size = 0;
		for (NodeID v = 0; v < numOfVertices; ++v) {
			two_index_t_p &idx = two_index_p[v];
			
			index_t_p &idx_original = index_p[v];
			
			NodeID isize = 0;
			for(NodeID i = 0; idx_original.spt_v[i] < UCHAR_MAX; ++i){
				++isize;
			}
			

			idx.spt_lv = (uint8_t*)memalign(64, (isize + 1) * sizeof(uint8_t));
			idx.spt_ld = (EdgeWeight*)memalign(64, (isize + 1) * sizeof(EdgeWeight));

		//	index_[v].spt_v.resize(isize);
		//	index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < isize; ++i) {
				uint8_t hub;
				EdgeWeight hub_weight;
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_lv[i] = idx_original.spt_v[i];
				idx.spt_ld[i] = idx_original.spt_d[i];

			}
			
			compressed_size += 4 * (isize - 1)  - isize;
			
			idx.spt_lv[isize] = UCHAR_MAX;
			idx.spt_ld[isize] = INF_WEIGHT;
			
			NodeID larger_size = 0;
			for(NodeID i = isize; idx_original.spt_v[i] != numOfVertices; ++i){
				++larger_size;
			}
			
			larger_size++;
			
			idx.spt_v = (NodeID*)memalign(64, larger_size * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, larger_size * sizeof(EdgeWeight));
			
			for (NodeID i = 0; i < larger_size; ++i) {
				uint8_t hub;
				EdgeWeight hub_weight;
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_v[i] = idx_original.spt_v[i + isize];
				idx.spt_d[i] = idx_original.spt_d[i + isize];
			}			
			
			total_size += 4 * (isize - 1 + larger_size) * 2;
			
		}		
		cout << "reduce size :" << compressed_size << " out of " << total_size << " saving " << int(compressed_size * 100 / total_size) << "%" << endl;

	}
	
	void load_labels_with_k(const char* load_filename, int k) {
		/*	for (NodeID v = 0; v < numOfVertices; ++v) {
		free(index_p[v].spt_v);
		free(index_p[v].spt_d);
		}
		*/
		//free(index_p);

		long total_amount = 0;
		long actual_amount = 0;

		index_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		index_p = (index_t_p*)memalign(64, numOfVertices * sizeof(index_t_p));



		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_p &idx = index_p[v];
			ifs.read((char*)&isize, sizeof(isize));
			int actual_isize = k;
			if (isize > k) actual_isize = k;
			else actual_isize = isize;

			total_amount += isize;
			actual_amount += actual_isize;

			idx.spt_v = (NodeID*)memalign(64, actual_isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, actual_isize * sizeof(EdgeWeight));

			//	index_[v].spt_v.resize(isize);
			//	index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < isize; ++i) {
				
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				if (i > actual_isize) continue;
				if (i == actual_isize - 1) {
					idx.spt_v[i] = numOfVertices;
					idx.spt_d[i] = INF_WEIGHT;
				}else {
					idx.spt_v[i] = hub;
					idx.spt_d[i] = hub_weight;
				}
			}
		}
		ifs.close();

		cout << "Total Labels:" << total_amount << endl;
		cout << "Actual Labels:" << actual_amount << endl;

		/*
		index_.clear();
		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		index_.resize(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {

		ifs.read((char*)&isize, sizeof(isize));
		index_[v].spt_v.resize(isize);
		index_[v].spt_d.resize(isize);

		for (NodeID i = 0; i < index_[v].size(); ++i) {
		NodeID hub;
		EdgeWeight hub_weight;
		ifs.read((char*)&hub, sizeof(hub));
		ifs.read((char*)&hub_weight, sizeof(hub_weight));
		index_[v].spt_v[i] = hub;
		index_[v].spt_d[i] = hub_weight;
		}
		}
		ifs.close();
		*/
	}


	void save_labels_iteration_stats(const char* save_filename) {

		vector<NodeID> stat(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {
			for (NodeID i = 0; i < index_[v].size(); ++i)
				stat[index_[v].spt_v[i]]++;
		}

		ofstream ofs(save_filename);

		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs << stat[v] << endl;
		}
		ofs.close();
	}


	EdgeWeight query_with_info(NodeID s, NodeID t, query_info& q_info) {

		double stime = GetCurrentTimeSec();

		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;		

		q_info.meet_node = numOfVertices;
		double meet_distance;

		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j]) {
				meet_distance = (EdgeWeight)(index_s_d[i++] + index_t_d[j++]);
				if ( distance >  meet_distance) {
					distance = meet_distance;
					q_info.meet_node = index_s[i];
				}
			}
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		};

		stime = GetCurrentTimeSec() - stime;

		q_info.time_cost = stime;

		if (index_s.size() < index_t.size())
			q_info.search_len = index_s.size();
		else
			q_info.search_len = index_t.size();

		return distance;
	}

	/**
	 * function used to save label size to file
	 * written by wanjingyi
	 * */
	void save_label_size(const char* label_size_file,const vector<NodeID>& inv) {
		string labelSizefile_prefix(label_size_file);
		string labelSizefile=labelSizefile_prefix.append(".size");
		ofstream ofs(labelSizefile);
		if(!ofs.is_open()) {cerr<<"Cannot open "<<labelSizefile<<endl;}
		for (int i = 0; i < numOfVertices; ++i) {
			ofs << index_[i].size()-1 << endl;
		}
		ofs.close();
		//output the size by descending order,format:label_size
		string labelSizefile_prefix1(label_size_file);
		string labelSizefile1=labelSizefile_prefix1.append("_byOrder.size");
		ofstream out(labelSizefile1.c_str());
		if(!out.is_open()) {cerr<<"Cannot open "<<labelSizefile1<<endl;}
		for (int i = 0; i < numOfVertices; ++i) {
			out<<index_[inv[i]].size()-1 << endl;
		}		
		out.close();
	}

	/**
	 * function used to serialize vertice's labels to file,format as follows:
	 * label_size v (w0,d(v,w0)) (w1,d(v,w1))...(wn,d(v,wn))
	 * written by wanjingyi
	 * */
	void write_labels(const char* write_filename,const vector<NodeID>& inv,bool isOrder=false)
	{
			string write_filename_prefix(write_filename);
			string write_filename1=write_filename_prefix.append(".list");
			ofstream ofs(write_filename1.c_str());
		for (NodeID v = 0; v < numOfVertices; ++v) 
		{
			NodeID isize = index_[v].size();
			ofs <<isize<<" "<<v;
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				ofs<<" "<<'('<<index_[v].spt_v[i]<<","<<index_[v].spt_d[i]<<")";
			}
			ofs<<endl;
		}
		ofs.close();

		//write labels with original NodeId in graph
		string write_filename2=write_filename_prefix.append("_original");
		ofstream out(write_filename2.c_str());
		for (NodeID v = 0; v < numOfVertices; ++v) 
		{
			NodeID isize = index_[v].size();
			out<<isize-1<<" "<<v;
			for (NodeID i = 0; i < index_[v].size()-1; ++i) {
				out<<" "<<'('<<inv[index_[v].spt_v[i]]<<","<<index_[v].spt_d[i]<<")";
			}
			out<<endl;
		}			
		out.close();

		if(isOrder){
			string write_filename_prefix1(write_filename);
			string write_filename3=write_filename_prefix1.append(".list_order");
			ofstream out1(write_filename3.c_str());
			for (NodeID r= 0; r < numOfVertices; ++r) 
			{
				NodeID v=inv[r];
				NodeID isize = index_[v].size();
				out1<<isize-1<<" "<<v;
				for (NodeID i = 0; i < index_[v].size()-1; ++i) {
					out1<<" "<<'('<<inv[index_[v].spt_v[i]]<<","<<index_[v].spt_d[i]<<")";
				}
				out1<<endl;
			}			
			out1.close();
			}
	}


};

/***
 * class HFLabel inherented from Label 
 * written by wanjingyi
 * **/
class HFLabel : public Label{
	public:
		vector<NodeID> HFPoint; //store the high frequency point referred orders
		vector<bool> HFinGraphIndex;//HFinGraphIndex is index of high frequency point in original graph, 0 represents low frequency point，1 represents high frequency point
		vector<NodeID> HFOripoint; //store the high frequency points'ids
		vector<NodeID> LFOripoint; //store the low frequency points'ids
		vector<NodeID> spt_v_num; //store the size of each vertice's labels
		vector<NodeID> rank;//store the verices' rank
		vector<int> queryTime;//store the query time of each point by id
		vector< pair<NodeID,int> > queryTime_hf; //store the query time of each point which is not null
		int numOfHFpoint; //number of High frequency points
		vector< vector<double> > hf_chache;//use to cache the h-h point distance
		vector< vector<int> > queryPairTime;//store the pair query frequency read from file
	
		HFLabel(){
			numOfHFpoint=0;
			HFPoint.clear();
			HFinGraphIndex.clear();
			spt_v_num.clear();
			rank.clear();
			HFOripoint.clear();
			LFOripoint.clear();
			queryTime.clear();
			queryPairTime.clear();
		}
		~HFLabel(){
			Free();
		}

		/*
		 *@description: load real pair query frequencies from file
		 *@return:total query times
		 *@author: wanjingyi
		 *@date: 2020-11-21
		*/
		int_fast32_t load_query_pair_time(char* load_filename){
			queryPairTime.resize(numOfVertices);
			for(size_t i=0;i<numOfVertices;++i) queryPairTime[i].resize(numOfVertices); //initial
			int cnt=0;//count the total times
			ifstream in(load_filename);//input query file to ifstream
			if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			NodeID s,t; 
			int query_time=0; 
			char line[24];
			int query_cnt=0;
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream ls(line);
				 ls>>s>>t>>query_time;
				 queryPairTime[s][t]=queryPairTime[t][s]=query_time;
				 cnt+=query_time;
			 }
			 in.close();
			 cout<<"total query times:"<<cnt<<endl;
			 return cnt;
		}

		/**
		 * function used to read  query time from file
		 * return total query_time
		 * written by wanjingyi
		 * */
		int load_query_time(char* load_filename){
			queryTime.resize(numOfVertices,0);
			ifstream in(load_filename);//input query file to ifstream
			if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			NodeID q_id;
			int query_time=0; //each line representing the label size of each vertice
			char line[24];
			int query_cnt=0;
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream ls(line);
				 ls>>q_id>>query_time;
				 queryTime[q_id]=query_time;
				 query_cnt+=query_time;
				 if(query_time!=0){
					 queryTime_hf.push_back(make_pair(q_id,query_time));
				 }
			 }
			 in.close();
			cout<<"total query time = "<<query_cnt<<endl;
		}

		/**
		 * function used to read  spt_v_num from file
		 * written by wanjingyi
		 * */
		void load_label_size(char* load_filename){
			spt_v_num.resize(numOfVertices,0);
			ifstream in(load_filename);//input HFPoint file to ifstream
			if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			NodeID label_size=0; //each line representing the label size of each vertice
			char line[24];
			int i=0;
			 //read each line representing HFpoint to vector 
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream ls(line);
				 ls>>label_size;
				 spt_v_num[i++]=label_size;
			 }
			 in.close();
			 if(i!=numOfVertices) cout<<"i!=numOfVertices"<<endl;
			 cout<<"load_label_size: numOfVertices="<<spt_v_num.size()<<endl;
		}

		/**
		 * function used to read orders from file
		 * written by wanjingyi
		 * */
		void load_order(char* load_filename)
		{
			rank.resize(numOfVertices,0);
			ifstream in(load_filename);//input orderfile to ifstream
			if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			NodeID id,r=0;
			char line[24];
			 //read each line representing HFpoint to vector 
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream id_ss(line);
				 id_ss>>id;
				 rank[id]=r++;
			 }
			in.close();
			if(r!=numOfVertices) cout<<"r!=numOfVertices"<<endl;
			
		}

		/**
		 * function used to read HFpoint from file
		 * written by wanjingyi
		 * */
		 void load_HFpoint(char* load_filename,int hfRate=50){
			 numOfHFpoint = 0;//first line is the number of HFpoints
			 numOfHFpoint = static_cast<int> ( (double)numOfVertices*hfRate/(double)1000);
			 if(numOfHFpoint<=0) cout<<"error:numOfHFpoint<=0"<<endl;
			 cout<<"numOfHFpoint  = "<<numOfHFpoint <<endl;
			 ifstream in(load_filename);//input HFPoint file to ifstream
			 if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			 NodeID t,id;int i=0;
			 char line[24];
			 //read each line representing HFpoint to vector 
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream hp(line);
				 hp>>t;
				 if(i<numOfHFpoint) HFOripoint.push_back(t);
				 else LFOripoint.push_back(t);
				 ++i;
			 }
			 in.close();
			 if(i!=numOfVertices) cout<<"i!=numOfVertices"<<endl;
			 cout<<"HFOripoint.size()="<<HFOripoint.size()<<endl;
			 cout<<"LFOripoint.size()="<<LFOripoint.size()<<endl;
			cout<<"numOfHFpoint="<<numOfHFpoint<<endl;

			//update HFPoint and HFinGraphIndex by order instead of node id
			HFinGraphIndex.resize(numOfVertices,0);//initial 0
			HFPoint.resize(numOfHFpoint,0);
			for(i=0;i<numOfHFpoint;i++)
			{
				id=HFOripoint[i];
				HFPoint[i]=rank[id]; //fetch the order rank by NodeId
				HFinGraphIndex[id]=true;
			}
		 }

		 /**
		  * function used to find  the minist-index label of current point
		  * modified by wanjingyi
		  * */
		 int findminIndex(vector<pair<NodeID,NodeID>>& index)
		 {
			NodeID min{ INT_MAX };//min代表当前找到的label中结点的最小编号
			int result{ -1 };//表示当前找到最小值的点对应的数组位置
			NodeID currentPoint;//表示当前比较的label结点
			//cout<<"findminIndex:numOfVertices="<<numOfVertices<<endl;
			for (int i = 0; i < numOfHFpoint; i++) {
				//if(i%100==0) cout<<"i="<<i<<" spt_v_num[i]="<<spt_v_num[i]<<";";
				if (index[i].second < spt_v_num[i]) {
					//cout<<"index=("<<index[i].first<<","<<index[i].second<<")"<<";";
					currentPoint = index_p[index[i].first].spt_v[index[i].second];
					//cout<<" currentPoint="<<currentPoint<<endl;
					if (currentPoint < min) {
						min = currentPoint;
						result = i;
					}
				}
			}
			for (int i = 0; i < HFPoint.size(); i++) {
				if (index_p[index[i].first].spt_v[index[i].second] == min) {
					index[i].second++;
				}	
			}
		if(result!=-1)
		index[result].second--;
			if (INT_MAX == min)
				result == -1;
			return result;
		 }

		 /**
		  * function used to serialize vertice's labels to file,format as follows:
		  * label_size v (w0,d(v,w0)) (w1,d(v,w1))...(wn,d(v,wn))
		  * written by wanjingyi
		  * */
		 void write_update_labels(const char* write_filename)
		 {
			 cout<<"write_update_labels begins!"<<endl;
			 string write_filename_prefix(write_filename);
			 string write_filename1=write_filename_prefix.append("_update.list");
			 ofstream ofs(write_filename1.c_str());
			 if(!ofs.is_open()) {cerr<<"Cannot open "<<write_filename1<<endl;}
			for (NodeID v = 0; v < numOfVertices; ++v) 
			{
				NodeID isize = spt_v_num[v]+1;
				ofs <<isize<<" "<<v;
				for (NodeID i = 0; i < isize; ++i) {
					ofs<<" "<<'('<<index_p[v].spt_v[i]<<","<<index_p[v].spt_d[i]<<")";
				}
				ofs<<endl;
			}
			ofs.close();
			cout<<"write_update_labels finished!"<<endl;
		 }

		 /**
		  * function used to output the total and ave size comparation
		  * written by wanjingyi
		  * */
		 void save_anaylysis_size(const char* freqid_filename,const char* write_filename,int hfRate=5)
		 {
			 //load hfPoint
			numOfHFpoint = 0;//first line is the number of HFpoints
			 numOfHFpoint = static_cast<int> ( (double)numOfVertices*hfRate/(double)1000);
			 if(numOfHFpoint<=0) cout<<"error:numOfHFpoint<=0"<<endl;
			 cout<<"numOfHFpoint  = "<<numOfHFpoint <<endl;
			 ifstream in(freqid_filename);//input HFPoint file to ifstream
			 HFOripoint.resize(numOfHFpoint,0);
			 HFinGraphIndex.resize(numOfVertices,0);
			 if(!in.is_open()) {cerr<<"Cannot open "<<freqid_filename<<endl;}
			 NodeID id;int i=0;
			 char line[24];
			 //read each line representing HFpoint to vector 
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream hp(line);
				 hp>>id;
				if(i>=numOfHFpoint) break;
				HFOripoint[i++]=id;
				HFinGraphIndex[id]=true;
			 }
			 in.close();
			//write size analysis to file
			long long total_sum_size=0,hf_sum_size=0;
			double total_ave_size=0,hf_ave_size=0;
			for (NodeID v = 0; v < numOfVertices; ++v) 
			{
				NodeID isize = index_[v].size()-1;
				total_sum_size+=isize;
				if(HFinGraphIndex[v]) hf_sum_size+=isize;
			}
			total_ave_size= (double) total_sum_size/(double) numOfVertices;
			hf_ave_size= (double) hf_sum_size/(double) numOfHFpoint;
			cout<<"numOfVertices = "<<numOfVertices<<" total_sum_size = "<<total_sum_size<<" total_ave_size = "<<total_ave_size<<endl;
			cout<<"numOfHFpoint = "<<numOfHFpoint<<" hf_sum_size = "<<hf_sum_size<<" hf_ave_size = "<<hf_ave_size<<endl;
			string write_filename_prefix(write_filename);
			string asize_filename=write_filename_prefix.append("_analysis.size");
			ofstream ofs(asize_filename.c_str());
			if(!ofs.is_open()) {cerr<<"Cannot open "<<asize_filename<<endl;}
			ofs<<"numOfVertices = "<<numOfVertices<<" total_sum_size = "<<total_sum_size<<" total_ave_size = "<<total_ave_size<<endl;
			ofs<<"numOfHFpoint = "<<numOfHFpoint<<" hf_sum_size = "<<hf_sum_size<<" hf_ave_size = "<<hf_ave_size<<endl;
			ofs.close();	
		 }

		 /**
		  * function used to serialize vertice's labels to file,format as follows:
		  * label_size v (w0,d(v,w0)) (w1,d(v,w1))...(wn,d(v,wn))
		  * written by wanjingyi
		  * */
		 void write_labels(const char* write_filename,const vector<NodeID>& inv)
		 {
			 string write_filename_prefix(write_filename);
			 string write_filename1=write_filename_prefix.append(".list");
			 ofstream ofs(write_filename1.c_str());
			for (NodeID v = 0; v < numOfVertices; ++v) 
			{
				NodeID isize = index_[v].size();
				ofs <<isize<<" "<<v;
				for (NodeID i = 0; i < index_[v].size(); ++i) {
					ofs<<" "<<'('<<index_[v].spt_v[i]<<","<<index_[v].spt_d[i]<<")";
				}
				ofs<<endl;
			}
			ofs.close();

			//write labels with original NodeId in graph
			string write_filename2=write_filename_prefix.append("o");
			ofstream out(write_filename2.c_str());
			for (NodeID v = 0; v < numOfVertices; ++v) 
			{
				NodeID isize = index_[v].size();
				out<<isize-1<<" "<<v;
				for (NodeID i = 0; i < index_[v].size()-1; ++i) {
					out<<" "<<'('<<inv[index_[v].spt_v[i]]<<","<<index_[v].spt_d[i]<<")";
				}
				out<<endl;
			}			
			out.close();
		 }

		/**
		 * function used to quickly updtae all labels
		 * written by wanjingyi
		 * */
		void update_all_labels()
		{

			//数据量小时，注释掉所有omp部分，数据量在50万以上时，恢复所有omp部分，打开VS的openMP并行支持

			//以下部分代码生成pivot的label，即每个pivot对应的结点编号，作为预处理，不计入Update时间，其结果pivot_p作为quickUpdateLabel函数的参数
				vector<vector<int>> pivotLabel(numOfVertices);//存放每个pivot对应的所有结点编号
				for (int i = 0; i < numOfVertices; i++) {
					pivotLabel[i].reserve(AVG_LABEL_SIZE);
				}
				for (int i = 0; i < numOfVertices; i++) {//对于每一个结点
					omp_set_num_threads(16);
			#pragma omp parallel for
					for (int j = 0; j < spt_v_num[i]; j++) {//对于结点中的每一个pivot
						pivotLabel[index_p[i].spt_v[j]].push_back(i);//将当前结点号存入该pivot，遍历过程确保了存储结果有序
					}
				}

				/****************quickUpdate***********/
				//time_t start_quick = clock();
				cout<<"update all point begins!"<<endl;
				double update_time=GetCurrentTimeSec();
				//unsigned __int64 numOfHFpoint{=HFPoint.size() };
				vector<int> HFhopPoint;//存放所有HFPoint中无重复的hop点
				vector<vector<int>> tempLabel(numOfVertices);//用来临时存放更新的所有结点的label
				//为tempLabel中每一个结点预留存储空间,设定预留大小为结点总数的1%
				omp_set_num_threads(16);
			#pragma omp parallel for
				for (int i = 0; i < numOfVertices; i++) {
					tempLabel[i].reserve(AVG_LABEL_SIZE+1);
				}																																			
				vector<bool> HFPivotFlag(numOfVertices, 0);//临时存放高频点pivto标记
				omp_set_num_threads(16);
			#pragma omp parallel for
				for (int i = 0; i < HFOripoint.size(); i++) {//对于每一个高频点
					for (int j = 0; j < spt_v_num[HFOripoint[i]]; j++) {//对于高频点中的每一个pivot
						HFPivotFlag[index_p[HFOripoint[i]].spt_v[j]] = 1;//将HFPivotFlag赋值为1
					}
				}
				for (int i = 0; i < numOfVertices; i++) {//对于HFPivotFlag中的每一个pivot
					if (1 == HFPivotFlag[i]) {//如果标记为1
						HFhopPoint.push_back(i);//将pivot号放入HFhopPoint
					}
				}
				HFPivotFlag.clear();
				//将所有高频点中的pivot对应结点的label按照pivot大小顺序存放在低频点的tempLabel中
				for (int i = 0; i < HFhopPoint.size(); i++) {//对于所有高频点中的每一个pivot
					omp_set_num_threads(16);
			#pragma omp parallel for
					for (int j = 0; j < pivotLabel[HFhopPoint[i]].size(); j++) {//对于pivot对应的每一个结点
						if (HFinGraphIndex[pivotLabel[HFhopPoint[i]][j]] == 0) {//如果当前点是低频点
							tempLabel[pivotLabel[HFhopPoint[i]][j]].push_back(HFhopPoint[i]);//将该pivot存入对应结点的临时label中
						}	
					}
				}
				HFhopPoint.clear();
				//以下代码用于查找HFPoint中共有的pivot，并将计数大于2的pivot按编号由小到大放入HFPivot中，作为更新HFPoint时调用updateHFLabel()的参数
				vector<int> pivotCount(numOfVertices, 0);//用来记录所有高频点中重复的pivot，其值大于2则代表此pivot应排在高频点label前部
				vector<int> HFPivot;//存放所有HFPoint中计数大于2的pivot
				for (int i = 0; i < numOfHFpoint; i++) {//对于每一个高频点
					for (int j = 0; j < spt_v_num[HFOripoint[i]]; j++)//对于高频点中的每个pivot
						pivotCount[index_p[HFOripoint[i]].spt_v[j]]++;//使得pivotCount中对应的pivot计数器+1
				}
				for (int i = 0; i < numOfVertices; i++) {
					if (pivotCount[i] > 1) //如果pivot计数大于1，则将该pivot放入HFpivot中
						HFPivot.push_back(i);
				}
				//将所有高频点共有的pivot放入所有高频点的tempLabel中
				for (int i = 0; i < HFPivot.size(); i++) {//对于所有高频点中的每一个pivot
					omp_set_num_threads(16);
			#pragma omp parallel for
					for (int j = 0; j < pivotLabel[HFPivot[i]].size(); j++) {//对于pivot对应的每一个结点
						if (HFinGraphIndex[pivotLabel[HFPivot[i]][j]] == 1)//如果当前点是高频点
							tempLabel[pivotLabel[HFPivot[i]][j]].push_back(HFPivot[i]);//将该pivot存入对应结点的临时label中
					}
				}
				HFPivot.clear();
				pivotCount.clear();

				vector<int> tempJoinCur(numOfVertices,0);//用于表示在tempLabel中，当前比较的位置
				vector<int> CurPreMax(numOfVertices,0);//用来表示每个结点在合并前tempLabel中包含pivot的最大值
				//将tempLabel中所有结果与原label合并，使得label中前半部分为tempLabel中的pivot，后半部分为其它pivot，前后都按降序排列
				omp_set_num_threads(16);
			#pragma omp parallel for
				for (int i = 0; i < numOfVertices; i++) {//对于每一个结点
					tempJoinCur[i] = 0;
					CurPreMax[i] = tempLabel[i].size();
					index_p[i].spt_v[spt_v_num[i]] = CurPreMax[i];//将后半部分pivot的索引号放在无穷大的位置
					for (int j = 0; j < spt_v_num[i]; j++) {//对于结点中的每一个pivot
						if (tempJoinCur[i] < CurPreMax[i]) {//如果tempLabel中前半部分pivot未比较结束
							if (index_p[i].spt_v[j] != tempLabel[i][tempJoinCur[i]])
								tempLabel[i].push_back(index_p[i].spt_v[j]);
							else
								tempJoinCur[i]++;
						}
						else
							tempLabel[i].push_back(index_p[i].spt_v[j]);//如果前半部分已比较结束，则只需将剩下的结点全部存入tempLabel
					}
					for (int j = 0; j < spt_v_num[i]; j++) {//将更新完后的Label赋值给原label，实现label更新
						index_p[i].spt_v[j] = tempLabel[i][j];
					}
				}
				//time_t end_quick = clock();

				update_time = GetCurrentTimeSec()-update_time;
				cout << " Total update time is : " << update_time* 1e6 << endl;
				double utime=update_time/numOfVertices;
				cout << " Each point update time is : " << utime* 1e6 << endl;
				cout<<"update all point successffully!"<<endl;
				//清除缓存
				tempLabel.clear();
				tempJoinCur.clear();
				CurPreMax.clear();

		}

		/**
		 * function used to updtae labels by low frequency points
		 * written by wanjingyi
		 * */
		 void update_low_labels()
		 {
			 vector<NodeID> HFhopPoint;//存放所有HFPoint中无重复的hop点
			 //建立高频点与遍历到的点的对应关系,第一个点为HFPoint序号，第二个值为HFPoint中当前搜索到的标签位置
			 vector<pair<NodeID,NodeID> > indexHFhopPoint(numOfHFpoint ,make_pair(0,0));
			 int i = numOfHFpoint;//i表示HFPoint各标签中剩余未取出的hop点
			 //用于临时存放更新label时选择的高频点对应的hop以及最终更新后的label
			 pair<vector<NodeID>, vector<NodeID> > tempHOP;
			 tempHOP.first.resize(numOfVertices);//初始化临时存放pair,预留10000储存区
			tempHOP.second.resize(numOfVertices);//初始化临时存放pair,预留10000储存区
			cout<<"tempHOP. initialize size is numOfVertices="<<numOfVertices<<endl;
			int tempcur;//表示tempHOP中当前存储到第几个label
			int hIndex, lIndex;//分别存储更新label时高频点和低频点对应的当前检索的label位置
			//初始化indexHFhopPoint为label对应点	
			for (int i = 0; i < HFOripoint.size(); i++) {
				indexHFhopPoint[i].first = HFOripoint[i];
			}
			cout<<"update_labels initialize successffully!"<<endl;

			//将各HFPoint中的label由小到大无重复地放入HFhopPoint中
			for (int i = 0; i != -1;) {
				i = findminIndex(indexHFhopPoint);
				//cout<<"i = "<<i<<endl;
				if (i != -1) {
					//将查找到的最小Label对应的Hop号放入HFhopPoint
					HFhopPoint.push_back(index_p[indexHFhopPoint[i].first].spt_v[indexHFhopPoint[i].second]);
					indexHFhopPoint[i].second++;
				}
			}
			cout<<"findminIndex successffully!"<<endl;

			double update_time=GetCurrentTimeSec();
			//更新图中所有低频点的label
			for (int i = 0; i < HFinGraphIndex.size(); i++) {
				if (!HFinGraphIndex[i]) {//如果该点不是高频点，则更新其label
					hIndex = 0;//初始化hIndex
					lIndex = 0;//初始化lIndex
					tempcur = 0;//初始化tempcur
					//对于每一个低频点，比对HFhopPoint和低频点中的label，将HFhopPoint中的部分放在低频点label的最前面，更新后的label仍按序排列
					while (hIndex < HFhopPoint.size() && lIndex < spt_v_num[i] ) {
						if (HFhopPoint[hIndex] == index_p[i].spt_v[lIndex]) {
							tempHOP.first[tempcur] = index_p[i].spt_v[lIndex];//将选中的共有hop存入tempHOP
							tempHOP.second[tempcur] = index_p[i].spt_d[lIndex];
							index_p[i].spt_v[lIndex] = -1;//将选中的共有hop标志为-1
							hIndex++;
							lIndex++;
							tempcur++;
						}
						else if (HFhopPoint[hIndex] < index_p[i].spt_v[lIndex])
							hIndex++;
						else
							lIndex++;
					}
					//将当前低频点中剩余的非共有hop点放入tempHOP,未比较部分不用处理
					if (tempcur > 0) {
						index_p[i].spt_v[spt_v_num[i]] = tempcur;//将Label后半部分的起始位置索引放在无穷大位置
						for (int j = 0; j < spt_v_num[i]; j++) {
							if (index_p[i].spt_v[j] != -1) {
								tempHOP.first[tempcur] = index_p[i].spt_v[j];
								tempHOP.second[tempcur++] = index_p[i].spt_d[j];
							}
						}
						//将更新后的tempHOP中的内容赋值给原label，实现原label更新
						for (int j = 0; j < tempcur; j++) {
							index_p[i].spt_v[j] = tempHOP.first[j];
							index_p[i].spt_d[j] = tempHOP.second[j];
						}
					}
				}
			}
			update_time = GetCurrentTimeSec()-update_time;
			cout << " Total update time is : " << update_time* 1e6 << endl;
			double utime=update_time/(numOfVertices-numOfHFpoint);
			cout << " Each LFPoint update time is : " << utime* 1e6 << endl;
			cout<<"update low frequency point successffully!"<<endl;

		 }

		/**
		 * function used to updtae labels by high frequency points
		 * written by wanjingyi
		 * */		 
		void update_high_labels()
		{
			 vector<NodeID> HFhopPoint;//存放所有HFPoint中无重复的hop点
			 //建立高频点与遍历到的点的对应关系,第一个点为HFPoint序号，第二个值为HFPoint中当前搜索到的标签位置
			 vector<pair<NodeID,NodeID> > indexHFhopPoint(numOfHFpoint ,make_pair(0,0));
			 int i = numOfHFpoint;//i表示HFPoint各标签中剩余未取出的hop点
			 //用于临时存放更新label时选择的高频点对应的hop以及最终更新后的label
			 pair<vector<NodeID>, vector<NodeID> > tempHOP;
			 tempHOP.first.resize(numOfVertices);//初始化临时存放pair,预留10000储存区
			tempHOP.second.resize(numOfVertices);//初始化临时存放pair,预留10000储存区
			cout<<"tempHOP. initialize size is numOfVertices="<<numOfVertices<<endl;
			int tempcur;//表示tempHOP中当前存储到第几个label
			int hIndex, lIndex;//分别存储更新label时高频点和低频点对应的当前检索的label位置
			//初始化indexHFhopPoint为label对应点	
			for (int i = 0; i < HFOripoint.size(); i++) {
				indexHFhopPoint[i].first = HFOripoint[i];
			}
			cout<<"update_labels initialize successffully!"<<endl;

			//将各HFPoint中的label由小到大无重复地放入HFhopPoint中
			for (int i = 0; i != -1;) {
				i = findminIndex(indexHFhopPoint);
				//cout<<"i = "<<i<<endl;
				if (i != -1) {
					//将查找到的最小Label对应的Hop号放入HFhopPoint
					HFhopPoint.push_back(index_p[indexHFhopPoint[i].first].spt_v[indexHFhopPoint[i].second]);
					indexHFhopPoint[i].second++;
				}
			}
			cout<<"findminIndex successffully!"<<endl;
			vector<int> pivotCount(numOfVertices, 0);//用来记录所有高频点中重复的pivot，其值大于2则代表此pivot应排在高频点label前部
			vector<int> HFPivot;//存放所有HFPoint中计数大于2的pivot
			for (int i = 0; i < numOfHFpoint; i++) {//对于每一个高频点
				for (int j = 0; j < spt_v_num[HFOripoint[i]]; j++)//对于高频点中的每个pivot
					pivotCount[index_p[HFOripoint[i]].spt_v[j]]++;//使得pivotCount中对应的pivot计数器+1
			}
			for (int i = 0; i < numOfVertices; i++) {
				if (pivotCount[i] > 1) //如果pivot计数大于1，则将该pivot放入HFpivot中
					HFPivot.push_back(i);
			}

			//更新图中所有低频点的label
			for (int i = 0; i < HFinGraphIndex.size(); i++) {
				if (!HFinGraphIndex[i]) {//如果该点不是高频点，则更新其label
					hIndex = 0;//初始化hIndex
					lIndex = 0;//初始化lIndex
					tempcur = 0;//初始化tempcur
					//对于每一个低频点，比对HFhopPoint和低频点中的label，将HFhopPoint中的部分放在低频点label的最前面，更新后的label仍按序排列
					while (hIndex < HFhopPoint.size() && lIndex < spt_v_num[i] ) {
						if (HFhopPoint[hIndex] == index_p[i].spt_v[lIndex]) {
							tempHOP.first[tempcur] = index_p[i].spt_v[lIndex];//将选中的共有hop存入tempHOP
							tempHOP.second[tempcur] = index_p[i].spt_d[lIndex];
							index_p[i].spt_v[lIndex] = -1;//将选中的共有hop标志为-1
							hIndex++;
							lIndex++;
							tempcur++;
						}
						else if (HFhopPoint[hIndex] < index_p[i].spt_v[lIndex])
							hIndex++;
						else
							lIndex++;
					}
					//将当前低频点中剩余的非共有hop点放入tempHOP,未比较部分不用处理
					if (tempcur > 0) {
						for (int j = 0; j < spt_v_num[i]; j++) {
							if (index_p[i].spt_v[j] != -1) {
								tempHOP.first[tempcur] = index_p[i].spt_v[j];
								tempHOP.second[tempcur++] = index_p[i].spt_d[j];
							}
						}
						//将更新后的tempHOP中的内容赋值给原label，实现原label更新
						for (int j = 0; j < tempcur; j++) {
							index_p[i].spt_v[j] = tempHOP.first[j];
							index_p[i].spt_d[j] = tempHOP.second[j];
						}
					}
				}else{//更新高频点的label，使得所有高频点的共有pivot在label前按序排列，原label大小不变
					updateHFLabel(HFPivot);
				}
			}
		}

//更新高频点的label，使得所有高频点的共有pivot在label前部按序排列，原label大小不变
void updateHFLabel(const vector<int> mutualPivot) {
	int hIndex{ 0 };//高频点label索引
	int pivotIndex{ 0 };//mutualPivot索引
	vector<pair<int, int>> tempLabel(numOfVertices, make_pair(0, 0));//临时存放更新过程中的高频点label
	int tempcur{ 0 };//表示tempLabel中当前存储到第几个label
	//对于每一个高频点，比对mutualPivot中的元素和高频点中的label，将包含在pivot中的部分放在高频点label的最前面，更新后的label仍按序排列
	for (int i = 0; i < HFOripoint.size(); i++) {
		int hIndex = 0;//初始化hIndex
		int pivotIndex = 0;//初始化pivotIndex
		int tempcur = 0;//初始化tempcur
		while (hIndex < spt_v_num[HFOripoint[i]] && pivotIndex < mutualPivot.size()) {
			if (mutualPivot[pivotIndex] == index_p[HFOripoint[i]].spt_v[hIndex]) {
				tempLabel[tempcur].first = index_p[HFOripoint[i]].spt_v[hIndex];//将选中的pivot存入tempLabel
				tempLabel[tempcur].second = index_p[HFOripoint[i]].spt_d[hIndex];
				index_p[HFOripoint[i]].spt_v[hIndex] = -1;//将选中的共有hop标志为-1
				hIndex++;
				pivotIndex++;
				tempcur++;
			}
			else if (mutualPivot[pivotIndex] < index_p[HFOripoint[i]].spt_v[hIndex])
				pivotIndex++;
			else
				hIndex++;
		}
		//将当前高频点中剩余的非共有pivot放入tempLabel,未比较部分不用处理
		if (tempcur > 0) {
			for (int j = 0; j < spt_v_num[HFOripoint[i]]; j++) {
				if (index_p[HFOripoint[i]].spt_v[j] != -1) {
					tempLabel[tempcur].first = index_p[HFOripoint[i]].spt_v[j];
					tempLabel[tempcur++].second = index_p[HFOripoint[i]].spt_d[j];
				}
			}
			//将更新后的tempLabel中的内容赋值给原label，实现原label更新
			for (int j = 0; j < tempcur; j++) {
				index_p[HFOripoint[i]].spt_v[j] = tempLabel[j].first;
				index_p[HFOripoint[i]].spt_d[j] = tempLabel[j].second;
			}
		}
	}
	
}
		 

		/**
		 * function used to save label size to file
		 * written by wanjingyi
		 * */
		void save_label_size(const char* label_size_file,const vector<NodeID>& inv) {
			string labelSizefile_prefix(label_size_file);
			string labelSizefile=labelSizefile_prefix.append(".size");
			ofstream ofs(labelSizefile);
			if(!ofs.is_open()) {cerr<<"Cannot open "<<labelSizefile<<endl;}
			for (int i = 0; i < numOfVertices; ++i) {
				ofs << index_[i].size()-1 << endl;
			}
			ofs.close();
			//output the size by descending order,format:label_size
			string labelSizefile_prefix1(label_size_file);
			string labelSizefile1=labelSizefile_prefix1.append("_byOrder.size");
			ofstream out(labelSizefile1.c_str());
			if(!out.is_open()) {cerr<<"Cannot open "<<labelSizefile1<<endl;}
			for (int i = 0; i < numOfVertices; ++i) {
				out<<index_[inv[i]].size()-1 << endl;
			}		
			out.close();
		}

		/**
		 * function used to save query distance to file
		 * written by wanjingyi
		 * */
		void save_query_distance(vector<pair<int, int> > queries,const char* query_distance_file,const vector<EdgeWeight>& distance_p, const vector<EdgeWeight>& distance_hf,int numQuery,int warmup,int queryModel=0,int updateModel=0) {
			string query_distance_file_prefix(query_distance_file);
			string appendix="";
			appendix+="_u"+to_string(updateModel)+"_q"+to_string(queryModel)+"_"+to_string(numQuery)+".dis";
			string query_distance_filename=query_distance_file_prefix.append(appendix);
			ofstream ofs(query_distance_filename);
			if(!ofs.is_open()) {cerr<<"Cannot open "<<query_distance_filename<<endl;}
			for (int i = warmup; i < warmup + numQuery; ++i) {
				ofs <<"("<<queries[i].first<<","<<queries[i].second<<") "<<distance_p[i-warmup];
				if(updateModel!=0) ofs<<" "<<distance_hf[i-warmup];
				ofs<<endl;
			}
			ofs.close();
		}

		/**
		 * function used to save query distance to file
		 * written by wanjingyi
		 * */
		void save_query_distance_data(vector<pair<int, int> > queries,const char* query_distance_file,const vector<EdgeWeight>& distance_p, const vector<EdgeWeight>& distance_hf,int timeQuery,int queryModel=0,int updateModel=0) {
			string query_distance_file_prefix(query_distance_file);
			string appendix="";
			appendix+="_u"+to_string(updateModel)+"_q"+to_string(queryModel)+"_"+to_string(timeQuery)+".dis";
			string query_distance_filename=query_distance_file_prefix.append(appendix);
			ofstream ofs(query_distance_filename);
			if(!ofs.is_open()) {cerr<<"Cannot open "<<query_distance_filename<<endl;}
			for (int i = 0; i < timeQuery; ++i) {
				ofs <<"("<<queries[i].first<<","<<queries[i].second<<") "<<distance_p[i];
				if(updateModel!=0) ofs<<" "<<distance_hf[i];
				ofs<<endl;
			}
			ofs.close();
		}

		EdgeWeight query_lf(NodeID s, NodeID t)
		{
			EdgeWeight distance = INF_WEIGHT;

			const index_t_p &idx_s = index_p[s];
			const index_t_p &idx_t = index_p[t];

			_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
			bool isBreak=false; //judge break or not
			for(int i=0,j=0; ; )
				{
					//if(i==0&&j==0) cout<<" s-t:h-l";
					if(j==idx_t.spt_v[spt_v_num[t]] ) break;
					NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];
					if (v1 == v2) {
						EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
						if (td < distance) distance = td;
						++i;
						++j;
					} 
					else {
						i += v1 < v2 ? 1 : 0;
						j += v1 > v2 ? 1 : 0;
					}
					
				}
				return distance;
		}
		
		/**
		 * function used to query s-t case:h-h including cache and comparison between no cache
		 * written by wanjingyi
		 * */
		EdgeWeight query_h_h(NodeID s,NodeID t)
		{
			EdgeWeight distance = INF_WEIGHT;
			const index_t_p &idx_s = index_p[s];
			const index_t_p &idx_t = index_p[t];

			_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

			//所有label都Update后，高频点之间的查询,无穷大位置存放后半部分起始点
			for (int i = 0, j = 0; ; ) {//i,j从label前半部分开始搜索
				if (i == idx_s.spt_v[spt_v_num[s]] || j == idx_t.spt_v[spt_v_num[t]]) break;//前半部分搜索到分界点位置时结束
				NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];
				if (v1 == v2) {
					EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
					if (td < distance) distance = td;
					++i;
					++j;
				}
				else {
					i += v1 < v2 ? 1 : 0;
					j += v1 > v2 ? 1 : 0;
				}
			}
			return distance;
		}

		/**
		 * function used to query s-t case:h-l
		 * written by wanjingyi
		 * */
		EdgeWeight query_h_l(NodeID s,NodeID t)
		{
			EdgeWeight distance = INF_WEIGHT;
			const index_t_p &idx_s = index_p[s];
			const index_t_p &idx_t = index_p[t];

			_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
			//所有label都update之后，高频点和低频点之间的查询,其中idx_s为高频点，idx_t为低频点
			for (int i = 0, imiddle = idx_s.spt_v[spt_v_num[s]], j = 0; ; )
			 {//imiddle表示高频点后半部分的起始点(该索引号存放在无穷大的位置)
				if (j==idx_t.spt_v[spt_v_num[t]] || (i == idx_s.spt_v[spt_v_num[s]] && imiddle == spt_v_num[s])) break;//低频点当前pivot大于下一个pivot或高频点中前半部分和后半部分都查询结束时，整个查询结束，低频点的无穷大保证了正确性
				NodeID v1 = idx_s.spt_v[i], v1middle = idx_s.spt_v[imiddle], v2 = idx_t.spt_v[j];
				if (v1 == v2) {
					EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
					if (td < distance) distance = td;
					++i;
					++j;
				}else{
					if (v1middle == v2) {
						EdgeWeight td = idx_s.spt_d[imiddle] + idx_t.spt_d[j];
						if (td < distance) distance = td;
						++imiddle;
						++j;
					}else {//高频点前后查询中都未找到与低频点共有的pivot,则pivot编号最小的+1
						if (v1 > v2) {
							if (v2 > v1middle)
								++imiddle;//v1middle最小
							else
								++j;//v2最小
						}
						else {
							if (v1 > v1middle) {
								++imiddle;//v1middle最小
							}
							else
								++i;//v1最小
						}
					}
				}
			 }
			return distance;

		}

		/**
		 * function used to query s-t case:l-h
		 * written by wanjingyi
		 * */
		EdgeWeight query_l_h(NodeID s,NodeID t)
		{
			NodeID tmp=s;s=t;t=tmp; //swap s and t
			EdgeWeight distance = INF_WEIGHT;
			const index_t_p &idx_s = index_p[s];
			const index_t_p &idx_t = index_p[t];

			_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

			//所有label都update之后，高频点和低频点之间的查询,其中idx_s为高频点，idx_t为低频点
			for (int i = 0, imiddle = idx_s.spt_v[spt_v_num[s]], j = 0; ; )
			 {//imiddle表示高频点后半部分的起始点(该索引号存放在无穷大的位置)
				if (j==idx_t.spt_v[spt_v_num[t]] || (i == idx_s.spt_v[spt_v_num[s]] && imiddle == spt_v_num[s])) break;//低频点当前pivot大于下一个pivot或高频点中前半部分和后半部分都查询结束时，整个查询结束，低频点的无穷大保证了正确性
				NodeID v1 = idx_s.spt_v[i], v1middle = idx_s.spt_v[imiddle], v2 = idx_t.spt_v[j];
				if (v1 == v2) {
					EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
					if (td < distance) distance = td;
					++i;
					++j;
				}else{
					if (v1middle == v2) {
						EdgeWeight td = idx_s.spt_d[imiddle] + idx_t.spt_d[j];
						if (td < distance) distance = td;
						++imiddle;
						++j;
					}else {//高频点前后查询中都未找到与低频点共有的pivot,则pivot编号最小的+1
						if (v1 > v2) {
							if (v2 > v1middle)
								++imiddle;//v1middle最小
							else
								++j;//v2最小
						}
						else {
							if (v1 > v1middle) {
								++imiddle;//v1middle最小
							}
							else
								++i;//v1最小
						}
					}
				}
			 }
			return distance;
		}

		/**
		 * function used to query s-t case:l-l
		 * written by wanjingyi
		 * */
		EdgeWeight query_l_l(NodeID s,NodeID t)
		{
			EdgeWeight distance = INF_WEIGHT;
			const index_t_p &idx_s = index_p[s];
			const index_t_p &idx_t = index_p[t];

			_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
			//所有label都Update后两点都为低频点的查询,无穷大位置存放后半部分pivot的起始位置
			for (int i = 0, j = 0; ; ) 
			{//i,j从label前半部分开始搜索
				if (i == idx_s.spt_v[spt_v_num[s]] || j == idx_t.spt_v[spt_v_num[t]]) break;//前半部分搜索到分界点位置时结束
				NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];
				if (v1 == v2) {
					EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
					if (td < distance) distance = td;
					++i;
					++j;
				}
				else {
					i += v1 < v2 ? 1 : 0;
					j += v1 > v2 ? 1 : 0;
				}
			}
			for (int iMiddle = idx_s.spt_v[spt_v_num[s]], jMiddle = idx_t.spt_v[spt_v_num[t]]; ; )
			 {//imiddle和jmiddle从label后半部分开始搜索
				if (iMiddle == spt_v_num[s] || jMiddle == spt_v_num[t]) break;//后半部分搜索到pivot最后一个位置之后结果
				NodeID v1Middle = idx_s.spt_v[iMiddle], v2Middle = idx_t.spt_v[jMiddle];
				if (v1Middle == v2Middle) {
					EdgeWeight tdMiddle = idx_s.spt_d[iMiddle] + idx_t.spt_d[jMiddle];
					if (tdMiddle < distance) distance = tdMiddle;
					++iMiddle;
					++jMiddle;
				}
				else {
					iMiddle += v1Middle < v2Middle ? 1 : 0;
					jMiddle += v1Middle > v2Middle ? 1 : 0;
				}
			 }	
			return distance;
		}

		/**
		 * function used to query all s-t cases
		 * written by wanjingyi
		 * */
		EdgeWeight query_all(NodeID s,NodeID t)
		{
			//0-random(default) 1-(s-t):(h-l),2-(s-t):(l:h),3-(s-t):(h-h),4-(s-t):(l-l)
			if(HFinGraphIndex[s]&&!HFinGraphIndex[t]) return query_h_l(s,t);
			else if(!HFinGraphIndex[s]&&HFinGraphIndex[t]) return query_l_h(s,t);
			else if(HFinGraphIndex[s]&&!HFinGraphIndex[t]) return query_h_h(s,t);
			else return query_l_l(s,t);
		}

		/**
		 * function used to query all s-t cases use cache
		 * written by wanjingyi
		 * */
		EdgeWeight query_all_cached(NodeID s,NodeID t)
		{
			//0-random(default) 1-(s-t):(h-l),2-(s-t):(l:h),3-(s-t):(h-h),4-(s-t):(l-l)
			if(HFinGraphIndex[s]&&!HFinGraphIndex[t]) return query_h_l(s,t);
			else if(!HFinGraphIndex[s]&&HFinGraphIndex[t]) return query_l_h(s,t);
			else if(HFinGraphIndex[s]&&!HFinGraphIndex[t]){
				if(hf_chache[s][t]!=INF_WEIGHT) return hf_chache[s][t];
				double result_distance=query_h_h(s,t);
				hf_chache[s][t]=result_distance;
				return result_distance;
			}
			else return query_l_l(s,t);
		}

		/**
		 * function used to query distinguished by high frequency and low frequency
		 * written by wanjingyi
		 * */
		EdgeWeight query_hf(NodeID s, NodeID t) {
			EdgeWeight distance = INF_WEIGHT;
			if(!HFinGraphIndex[s]&&HFinGraphIndex[t]){ //s-t:l-h
				NodeID tmp=s;
				s=t;
				t=tmp;
			}
			const index_t_p &idx_s = index_p[s];
			const index_t_p &idx_t = index_p[t];

			_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
			_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

			if(HFinGraphIndex[s]&&HFinGraphIndex[t]){ // (s,t)both high 
				for(int i=0,j=0; ; )
				{
					NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];
					if (v1 == numOfVertices|| v2== numOfVertices) break;  // Sentinel
					if(v1>idx_s.spt_v[i+1] || v2>idx_t.spt_v[j+1]){
						break;
					}
					if (v1 == v2) {
						EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
						if (td < distance) distance = td;
						++i;
						++j;
					} 
					else {
						i += v1 < v2 ? 1 : 0;
						j += v1 > v2 ? 1 : 0;
					}

				}
			}else if(HFinGraphIndex[s]&&!HFinGraphIndex[t]){ //s-high,t-low
			//imiddle表示高频点后半部分的起始点(该索引号存放在无穷大的位置)
				for (int i = 0, imiddle = idx_s.spt_v[spt_v_num[s]], j = 0; ; ) {
					//低频点当前pivot大于下一个pivot或高频点中前半部分和后半部分都查询结束时，整个查询结束，低频点的无穷大保证了正确性
					if (idx_t.spt_v[j] > idx_t.spt_v[j + 1] || (i == idx_s.spt_v[spt_v_num[s]] && imiddle == spt_v_num[s])) break;
					NodeID v1 = idx_s.spt_v[i], v1middle = idx_s.spt_v[imiddle], v2 = idx_t.spt_v[j];
					if (v1 == numOfVertices) break;  // Sentinel
					if (i < idx_s.spt_v[spt_v_num[s]]) {//i只查询高频点前半部分
						if (v1 == v2) {
							EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
							if (td < distance) distance = td;
							++i;
							++j;
						}else{//若前半部分当前未查到相同的pivot，则要进入后半部分进行查询
							while (v1middle != v2 && imiddle < spt_v_num[s]) {//imiddle只查询高频点后半部分
								++imiddle;
								v1middle == idx_s.spt_v[imiddle];//无穷大位置用来存放后半部分起始点标记，保证了数组不会越界
							}
							if (v1middle == v2) {
								EdgeWeight td = idx_s.spt_d[imiddle] + idx_t.spt_d[j];
								if (td < distance) distance = td;
								++imiddle;
								++j;
							}else{
								if (v1 > v2) {
									if (v2 > v1middle)
										++imiddle;//v1middle最小
									else
										++j;//v2最小
								}
								else {
									if (v1 > v1middle) {
										++imiddle;//v1middle最小
									}
									else ++i;//v1最小
								}
							}
						}

					}
					else {//前半部分都查询结束了，那么只剩后半部分要查询
						if (v1middle == v2) {
							EdgeWeight td = idx_s.spt_d[imiddle] + idx_t.spt_d[j];
							if (td < distance) distance = td;
							++imiddle;
							++j;
						}
						else {
							imiddle += v1middle < v2 ? 1 : 0;
							j += v1middle > v2 ? 1 : 0;
						}
				 	}
				}
			}else if(!HFinGraphIndex[s]&&HFinGraphIndex[t]){ //s-low,t-high
				cout<<" s-t:l-h"<<endl;
			}else{ //low-low frequency point query
				// for (int i = 0, j = 0; ; ) {
				// NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

				// if (v1 == numOfVertices) break;  // Sentinel

				// 	if (v1 == v2) {
				// 	EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				// 	if (td < distance) distance = td;
				// 	++i;
				// 	++j;
				// 	} 
				// 	else {
				// 		i += v1 < v2 ? 1 : 0;
				// 		j += v1 > v2 ? 1 : 0;
				// 	}
				// }
					int flag{ 1 };
					for (int i = 0, j = 0, iBack = spt_v_num[s]-1, jBack = spt_v_num[t]; ; ) {//i,j从label前向搜索，iBack和jBack从label后向搜索
						if (i == iBack || j == jBack|| 0==flag) break;
						flag = 0;//flag用于排除两个搜索悬空无法相等的特殊情况
						NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j], v1Back = idx_s.spt_v[iBack], v2Back = idx_t.spt_v[jBack];
						if (v1 == numOfVertices || v2 == numOfVertices) break;  // Sentinel
						if (v1 < idx_s.spt_v[i + 1] && v2 < idx_t.spt_v[j + 1]) {//进入前向搜索的条件是label中当前点序号小于下一点序号
							flag = 1;
							if (v1 == v2) {
								EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
								if (td < distance) distance = td;
								++i;
								++j;
							}
							else {
								i += v1 < v2 ? 1 : 0;
								j += v1 > v2 ? 1 : 0;
							}
						}
						if (v1Back > idx_s.spt_v[iBack - 1] && v2Back > idx_t.spt_v[jBack - 1]) {//进入后向搜索的条件是label中当前点序号大于下一点序号
							flag = 1;
							if (v1Back == v2Back) {
								EdgeWeight tdBack = idx_s.spt_d[iBack] + idx_t.spt_d[jBack];
								if (tdBack < distance) distance = tdBack;
								--iBack;
								--jBack;
							}
							else {
								iBack -= v1 > v2 ? 1 : 0;
								jBack -= v1 < v2 ? 1 : 0;
							}
						}
					}

			}
			
			return distance;
	}


};



class PLabel {

public:
	vector<index_t_path> index_;
	index_t_path_p* index_p;


	double GetCurrentTimeSec() {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + tv.tv_usec * 1e-6;
	}


	PLabel() {
		index_.resize(numOfVertices);
	}

	~PLabel() {
		Free();
	}

	EdgeWeight query_p(NodeID s, NodeID t) {

		//EdgeWeight distance = INF_WEIGHT;

		//NodeID *vs = index_p[s].spt_v;
		//NodeID *vt = index_p[t].spt_v;
		//EdgeWeight* ws = index_p[s].spt_d;
		//EdgeWeight* wt = index_p[t].spt_d;

		//_mm_prefetch(vs, _MM_HINT_T0);
		//_mm_prefetch(vt, _MM_HINT_T0);
		//_mm_prefetch(ws, _MM_HINT_T0);
		//_mm_prefetch(wt, _MM_HINT_T0);

		//for (unsigned i = 0, j = 0; ; ) {
		//	if (*(vs + i) == *(vt + j)) {
		//		if (*(vs + i) == numOfVertices) break;  // Sentinel
		//		EdgeWeight td = *(ws + i) + *(wt + j);
		//		if (td < distance) distance = td;
		//		++i;
		//		++j;
		//	}
		//	else {
		//		i += *(vs + i) < *(vt + j) ? 1 : 0;
		//		j += *(vs + i) > *(vt + j) ? 1 : 0;
		//	}
		//}
		//return distance;


		EdgeWeight distance = INF_WEIGHT;
		NodeID meet;

		const index_t_path_p &idx_s = index_p[s];
		const index_t_path_p &idx_t = index_p[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) {
					distance = td;
				}
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}

		return distance;
	}

	EdgeWeight query(NodeID s, NodeID t) {
		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;

		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j])
				distance = min(distance, (EdgeWeight)(index_s_d[i++] + index_t_d[j++]));
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		}
		return distance;
	}

	EdgeWeight query(NodeID s, NodeID t, NodeID& meet, EdgeWeight& dis1, EdgeWeight& dis2) {
		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;
		meet = numeric_limits<NodeID>::max();
		dis1 = numeric_limits<EdgeWeight>::max();
		dis2 = numeric_limits<EdgeWeight>::max();
		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j]) {
				if (distance >(EdgeWeight)(index_s_d[i] + index_t_d[j])) {
					distance = (EdgeWeight)(index_s_d[i] + index_t_d[j]);
					meet = index_s[i];
					dis1 = index_s_d[i];
					dis2 = index_t_d[j];
				}
				++i; ++j;
			}
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		}
		return distance;
	}


	EdgeWeight query_path(NodeID s, NodeID t, vector<NodeID>& rank, vector<NodeID>& inv) {


		EdgeWeight distance = INF_WEIGHT;
		NodeID meetnode = numOfVertices;
		NodeID s_parent;
		NodeID t_parent;

		const index_t_path_p &idx_s = index_p[s];
		const index_t_path_p &idx_t = index_p[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_p[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_p[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) {
					distance = td;
				//	if (v1 < meetnode) {
						meetnode = v1;
						s_parent = idx_s.spt_p[i];
						t_parent = idx_t.spt_p[j];
					//}
				}
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}


		//Next, retrieve path from s - meetnode and meetnode - t.
		vector<NodeID> path_from_s;
		vector<NodeID> path_to_t;
		path_from_s.push_back(s_parent);
		path_to_t.push_back(t_parent);
		
		int operation = 0;

		/*	if (s == 194569 && t == 20072)
		cout << "debug." << " meet: " << meetnode << " sparent:" << s_parent << " tparent:" << t_parent <<  endl;*/

		NodeID inv_meetnode = inv[meetnode];

		while (path_from_s.back() != inv_meetnode) {
			/*if (s == 194569 && t == 20072)
			cout << "s meet:" << path_from_s.back() << endl;*/
			const index_t_path_p &idx_from_s = index_p[path_from_s.back()];

			_mm_prefetch(&idx_from_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_from_s.spt_p[0], _MM_HINT_T0);
			 
		//	vector<NodeID>& index_from_s = index_[path_from_s.back()].spt_v;
			for (int i = 0; ; ++i) {
				operation++;
				if (idx_from_s.spt_v[i] == numOfVertices) break;
				if (idx_from_s.spt_v[i] == meetnode) {
					path_from_s.push_back(idx_from_s.spt_p[i]);
					break;
				}
			}
		}

		while (path_to_t.back() != inv_meetnode) {
			/*if (s == 194569 && t == 20072)
			cout << "t meet:" << path_to_t.back() << endl;*/
		//	vector<NodeID>& index_to_t = index_[path_to_t.back()].spt_v;
			const index_t_path_p &idx_to_t = index_p[path_to_t.back()];
			
			_mm_prefetch(&idx_to_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_to_t.spt_p[0], _MM_HINT_T0);
			for (int i = 0; ; ++i) {
				operation++;
				if (idx_to_t.spt_v[i] == numOfVertices) break;
				if (idx_to_t.spt_v[i] == meetnode) {
					path_to_t.push_back(idx_to_t.spt_p[i]);
					break;
				}
			}
		}

		distance = 0;
		distance += path_from_s.size() + path_to_t.size();

//		return distance;
		return distance;
		//EdgeWeight distance = INF_WEIGHT;
		//vector<NodeID>& index_s = index_[s].spt_v;
		//vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		//vector<NodeID>& bindex_t = index_[t].spt_v;
		//vector<EdgeWeight>& bindex_t_d = index_[t].spt_d;


		//NodeID meetnode = numOfVertices;
		//int s_parent;
		//int t_parent;
		//for (int i = 0, j = 0; i < index_s.size(), j < bindex_t.size(); ) {
		//	if (index_s[i] == bindex_t[j]) {
		//		if (distance >(EdgeWeight)(index_s_d[i] + bindex_t_d[j])) {
		//			distance = (EdgeWeight)(index_s_d[i] + bindex_t_d[j]);
		//			if (index_s[i] < meetnode) {
		//				meetnode = index_s[i];
		//				s_parent = index_[s].spt_p[i];
		//				t_parent = index_[t].spt_p[j];
		//			}
		//		}
		//		//distance = min(distance, (EdgeWeight)(index_s_d[i] + bindex_t_d[j]));
		//		++i;
		//		++j;
		//	}
		//	else {
		//		if (index_s[i] < bindex_t[j])
		//			++i;
		//		else
		//			++j;
		//	}
		//}

		////Next, retrieve path from s - meetnode and meetnode - t.
		//vector<NodeID> path_from_s;
		//vector<NodeID> path_to_t;
		//path_from_s.push_back(s_parent);
		//path_to_t.push_back(t_parent);

		///*	if (s == 194569 && t == 20072)
		//cout << "debug." << " meet: " << meetnode << " sparent:" << s_parent << " tparent:" << t_parent <<  endl;*/

		//while (path_from_s.back() != inv[meetnode]) {
		//	/*if (s == 194569 && t == 20072)
		//	cout << "s meet:" << path_from_s.back() << endl;*/
		//	vector<NodeID>& index_from_s = index_[path_from_s.back()].spt_v;
		//	for (int i = 0; i < index_from_s.size(); ++i) {
		//		if (index_from_s[i] == meetnode) {
		//			path_from_s.push_back(index_[path_from_s.back()].spt_p[i]);
		//			break;
		//		}
		//	}
		//}

		//while (path_to_t.back() != inv[meetnode]) {
		//	/*if (s == 194569 && t == 20072)
		//	cout << "t meet:" << path_to_t.back() << endl;*/
		//	vector<NodeID>& index_to_t = index_[path_to_t.back()].spt_v;
		//	for (int i = 0; i < index_to_t.size(); ++i) {
		//		if (index_to_t[i] == meetnode) {
		//			path_to_t.push_back(index_[path_to_t.back()].spt_p[i]);
		//			break;
		//		}
		//	}
		//}

		////for (int i = 0; i < path_from_s.size(); ++i)
		////	path_from_s[i] = inv[path_from_s[i]];
		////for (int i = 0; i < path_to_t.size(); ++i)
		////	path_to_t[i] = inv[path_to_t[i]];

		//return path_from_s.size() + path_to_t.size();
	}

	EdgeWeight query_path_check(NodeID s, NodeID t, vector<NodeID>& rank, vector<NodeID>& inv) {


		EdgeWeight distance = INF_WEIGHT;
		NodeID meetnode = numOfVertices;
		NodeID s_parent;
		NodeID t_parent;

		const index_t_path_p &idx_s = index_p[s];
		const index_t_path_p &idx_t = index_p[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_p[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_p[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) {
					distance = td;
				//	if (v1 < meetnode) {
						meetnode = v1;
						s_parent = idx_s.spt_p[i];
						t_parent = idx_t.spt_p[j];
					//}
				}
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}
		
		NodeID inv_meetnode = inv[meetnode];
		//Next, retrieve path from s - meetnode and meetnode - t.
		vector<NodeID> path_from_s;
		vector<NodeID> path_to_t;
		if(s !=inv_meetnode)
			path_from_s.push_back(s);
		path_from_s.push_back(s_parent);
		if (t != inv_meetnode)	
			path_to_t.push_back(t);

		path_to_t.push_back(t_parent);

		/*	if (s == 194569 && t == 20072)
		cout << "debug." << " meet: " << meetnode << " sparent:" << s_parent << " tparent:" << t_parent <<  endl;*/


		while (path_from_s.back() != inv_meetnode) {
			/*if (s == 194569 && t == 20072)
			cout << "s meet:" << path_from_s.back() << endl;*/
			const index_t_path_p &idx_from_s = index_p[path_from_s.back()];

			_mm_prefetch(&idx_from_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_from_s.spt_p[0], _MM_HINT_T0);

			//	vector<NodeID>& index_from_s = index_[path_from_s.back()].spt_v;
			for (int i = 0; ; ++i) {
				if (idx_from_s.spt_v[i] == numOfVertices) break;
				if (idx_from_s.spt_v[i] == meetnode) {
					path_from_s.push_back(idx_from_s.spt_p[i]);
					break;
				}
			}
		}

		while (path_to_t.back() != inv_meetnode) {
			/*if (s == 194569 && t == 20072)
			cout << "t meet:" << path_to_t.back() << endl;*/
			//	vector<NodeID>& index_to_t = index_[path_to_t.back()].spt_v;
			const index_t_path_p &idx_to_t = index_p[path_to_t.back()];
			_mm_prefetch(&idx_to_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_to_t.spt_p[0], _MM_HINT_T0);
			for (int i = 0; ; ++i) {
				if (idx_to_t.spt_v[i] == numOfVertices) break;
				if (idx_to_t.spt_v[i] == meetnode) {
					path_to_t.push_back(idx_to_t.spt_p[i]);
					break;
				}
			}
		}

		//return distance;

		EdgeWeight alldis = 0;

		if (path_from_s.size() == 1)
			if (s != inv_meetnode)
				alldis += query_p(s, inv_meetnode);

		if (path_to_t.size() == 1)
			if (t != inv_meetnode)
				alldis += query_p(t, inv_meetnode);

		for (int i = 0; i < path_from_s.size() - 1; ++i) {
			alldis += query_p(path_from_s[i], path_from_s[i + 1]);
			//cout << "s " << path_from_s[i] << "," << path_from_s[i + 1] << endl;
		}
		for (int i = 0; i < path_to_t.size() - 1; ++i) {
			alldis += query_p(path_to_t[i], path_to_t[i + 1]);

			//cout <<"t " <<  path_to_t[i] << "," << path_to_t[i + 1] << endl;
		}
		/*if (distance != alldis)
			cout << "a?" << endl;*/
		//cout << distance << "," << alldis << "," << path_from_s.size() + path_to_t.size() << endl;
//		cout << s << "," << t << "," << inv_meetnode << "   " << distance << "vs." << alldis << endl;

		return distance;
	}


	//EdgeWeight query_path_check(NodeID s, NodeID t, vector<NodeID>& rank, vector<NodeID>& inv) {


	//	EdgeWeight distance = INF_WEIGHT;
	//	NodeID meetnode = numOfVertices;
	//	NodeID s_parent;
	//	NodeID t_parent;

	//	const index_t_path_p &idx_s = index_p[s];
	//	const index_t_path_p &idx_t = index_p[t];

	//	_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
	//	_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
	//	_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
	//	_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
	//	_mm_prefetch(&idx_s.spt_p[0], _MM_HINT_T0);
	//	_mm_prefetch(&idx_t.spt_p[0], _MM_HINT_T0);

	//	for (int i = 0, j = 0; ; ) {
	//		NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

	//		if (v1 == numOfVertices) break;  // Sentinel

	//		if (v1 == v2) {
	//			EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
	//			if (td < distance) {
	//				distance = td;
	//				if (v1 < meetnode) {
	//					meetnode = v1;
	//					s_parent = idx_s.spt_p[i];
	//					t_parent = idx_t.spt_p[j];
	//				}
	//			}
	//			++i;
	//			++j;
	//		}
	//		else {
	//			i += v1 < v2 ? 1 : 0;
	//			j += v1 > v2 ? 1 : 0;
	//		}
	//	}

	//	//Next, retrieve path from s - meetnode and meetnode - t.
	//	vector<NodeID> path_from_s;
	//	vector<NodeID> path_to_t;
	//	path_from_s.push_back(s_parent);
	//	path_to_t.push_back(t_parent);

	//	/*	if (s == 194569 && t == 20072)
	//	cout << "debug." << " meet: " << meetnode << " sparent:" << s_parent << " tparent:" << t_parent <<  endl;*/

	//	NodeID inv_meetnode = inv[meetnode];

	//	while (path_from_s.back() != inv_meetnode) {
	//		/*if (s == 194569 && t == 20072)
	//		cout << "s meet:" << path_from_s.back() << endl;*/
	//		const index_t_path_p &idx_from_s = index_p[path_from_s.back()];

	//		_mm_prefetch(&idx_from_s.spt_v[0], _MM_HINT_T0);
	//		_mm_prefetch(&idx_from_s.spt_p[0], _MM_HINT_T0);

	//		//	vector<NodeID>& index_from_s = index_[path_from_s.back()].spt_v;
	//		for (int i = 0; ; ++i) {
	//			if (idx_from_s.spt_v[i] == numOfVertices) break;
	//			if (idx_from_s.spt_v[i] == meetnode) {
	//				path_from_s.push_back(idx_from_s.spt_p[i]);
	//				break;
	//			}
	//		}
	//	}

	//	while (path_to_t.back() != inv_meetnode) {
	//		/*if (s == 194569 && t == 20072)
	//		cout << "t meet:" << path_to_t.back() << endl;*/
	//		//	vector<NodeID>& index_to_t = index_[path_to_t.back()].spt_v;
	//		const index_t_path_p &idx_to_t = index_p[path_to_t.back()];
	//		_mm_prefetch(&idx_to_t.spt_v[0], _MM_HINT_T0);
	//		_mm_prefetch(&idx_to_t.spt_p[0], _MM_HINT_T0);
	//		for (int i = 0; ; ++i) {
	//			if (idx_to_t.spt_v[i] == numOfVertices) break;
	//			if (idx_to_t.spt_v[i] == meetnode) {
	//				path_to_t.push_back(idx_to_t.spt_p[i]);
	//				break;
	//			}
	//		}
	//	}

	//	EdgeWeight path_from_s = 0;
	//	for (int i = 0; i < path_from_s.size(); ++i) {

	//	}

	//	

	//	return distance;

	//	
	//}


	/*EdgeWeight query_new(NodeID s, NodeID t, Ordering& ordering) {
	EdgeWeight distance = INF_WEIGHT;
	vector<NodeID>& index_s = index_[s].spt_v;
	vector<EdgeWeight>& index_s_d = index_[s].spt_d;

	vector<NodeID>& index_t = index_[t].spt_v;
	vector<EdgeWeight>& index_t_d = index_[t].spt_d;



	for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
	if (index_s[i] == index_t[j])
	distance = min(distance, (EdgeWeight)(index_s_d[i++] + index_t_d[j++]));
	else {
	if (index_s[i] < index_t[j])
	++i;
	else
	++j;
	}
	}
	return distance;
	}
	*/
	double avg_size() {
		double total = 0;
		for (int i = 0; i < numOfVertices; ++i) total += index_[i].spt_v.size();

		double avg = total / numOfVertices - 1; // We do not count the trivial label (V, INF_WEIGHT).

		return avg;
	}
	/*
	NodeID max_size() {
	NodeID maxsize = numeric_limits<NodeID>::min();
	for (int i = 0; i < V; ++i) maxsize = max(maxsize, index_[i].spt_v.size());
	return maxsize;
	}*/

	void append(NodeID v, NodeID root, EdgeWeight distance) {
		index_[v].spt_v.push_back(root);
		index_[v].spt_d.push_back(distance);
	}

	void print_stat() {
		cout << "Average Label Size: " << avg_size() << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}

	void Free() {
		if (index_.size() == 0) return;
		for (int v = 0; v < numOfVertices; ++v) {
			index_[v].spt_v.clear();
			index_[v].spt_d.clear();
		}
		index_.clear();
	}

	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			NodeID isize = index_[v].size();
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				ofs.write((const char*)&index_[v].spt_v[i], sizeof(index_[v].spt_v[i]));
				ofs.write((const char*)&index_[v].spt_p[i], sizeof(index_[v].spt_p[i]));
				ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
			}
		}
		ofs.close();
	}

	void load_labels(const char* load_filename) {
		/*	for (NodeID v = 0; v < numOfVertices; ++v) {
		free(index_p[v].spt_v);
		free(index_p[v].spt_d);
		}
		*/
		//free(index_p);
		index_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		index_p = (index_t_path_p*)memalign(64, numOfVertices * sizeof(index_t_path_p));

		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_path_p &idx = index_p[v];
			ifs.read((char*)&isize, sizeof(isize));

			idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_p = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));

			//	index_[v].spt_v.resize(isize);
			//	index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				NodeID hub_parent;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_parent, sizeof(hub_parent));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;
				idx.spt_v[i] = hub;
				idx.spt_p[i] = hub_parent;
				idx.spt_d[i] = hub_weight;
			}
		}
		ifs.close();

		/*
		index_.clear();
		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		index_.resize(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {

		ifs.read((char*)&isize, sizeof(isize));
		index_[v].spt_v.resize(isize);
		index_[v].spt_d.resize(isize);

		for (NodeID i = 0; i < index_[v].size(); ++i) {
		NodeID hub;
		EdgeWeight hub_weight;
		ifs.read((char*)&hub, sizeof(hub));
		ifs.read((char*)&hub_weight, sizeof(hub_weight));
		index_[v].spt_v[i] = hub;
		index_[v].spt_d[i] = hub_weight;
		}
		}
		ifs.close();
		*/
	}

	void save_labels_iteration_stats(const char* save_filename) {

		vector<NodeID> stat(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {
			for (NodeID i = 0; i < index_[v].size(); ++i)
				stat[index_[v].spt_v[i]]++;
		}

		ofstream ofs(save_filename);

		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs << stat[v] << endl;
		}
		ofs.close();
	}

	EdgeWeight query_with_info(NodeID s, NodeID t, query_info& q_info) {

		double stime = GetCurrentTimeSec();

		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& index_t = index_[t].spt_v;
		vector<EdgeWeight>& index_t_d = index_[t].spt_d;

		q_info.meet_node = numOfVertices;
		double meet_distance;

		for (int i = 0, j = 0; i < index_s.size(), j < index_t.size(); ) {
			if (index_s[i] == index_t[j]) {
				meet_distance = (EdgeWeight)(index_s_d[i++] + index_t_d[j++]);
				if (distance >  meet_distance) {
					distance = meet_distance;
					q_info.meet_node = index_s[i];
				}
			}
			else {
				if (index_s[i] < index_t[j])
					++i;
				else
					++j;
			}
		};

		stime = GetCurrentTimeSec() - stime;

		q_info.time_cost = stime;

		if (index_s.size() < index_t.size())
			q_info.search_len = index_s.size();
		else
			q_info.search_len = index_t.size();

		return distance;
	}


};


class DLabel : public Label {
public:
	vector<index_t> bindex_; // Backward labels.

	index_t_p* bindex_p;
	
	
	two_index_t_p* b_two_index_p;


	DLabel() {
		index_.resize(numOfVertices);
		bindex_.resize(numOfVertices);
	}
	
	~DLabel() {  
		Free();
	}

	EdgeWeight query(NodeID s, NodeID t) {

		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& bindex_t = bindex_[t].spt_v;
		vector<EdgeWeight>& bindex_t_d = bindex_[t].spt_d;

		for (int i = 0, j = 0; i < index_s.size(), j < bindex_t.size(); ) {
			if (index_s[i] == bindex_t[j]) {
				distance = min(distance, (EdgeWeight)(index_s_d[i] + bindex_t_d[j]));
				++i;
				++j;
			}
			else {
				if (index_s[i] < bindex_t[j])
					++i;
				else
					++j;
			}
		}

		return distance;
	}

	EdgeWeight query(NodeID s, NodeID t, NodeID& meet, EdgeWeight& dis1, EdgeWeight& dis2) {
		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& bindex_t = bindex_[t].spt_v;
		vector<EdgeWeight>& bindex_t_d = bindex_[t].spt_d;

		meet = numeric_limits<NodeID>::max();
		dis1 = numeric_limits<EdgeWeight>::max();
		dis2 = numeric_limits<EdgeWeight>::max();
		for (int i = 0, j = 0; i < index_s.size(), j < bindex_t.size(); ) {
			if (index_s[i] == bindex_t[j]) {
				if (distance > (EdgeWeight)(index_s_d[i] + bindex_t_d[j])) {
					distance = (EdgeWeight)(index_s_d[i] + bindex_t_d[j]);
					meet = index_s[i];
					dis1 = index_s_d[i];
					dis2 = bindex_t_d[j];
				}
				++i;
				++j;
			}
			else {
				if (index_s[i] < bindex_t[j])
					++i;
				else
					++j;
			}
		}

		return distance;
	}
	
	inline EdgeWeight query_p(NodeID s, NodeID t) {

		//EdgeWeight distance = INF_WEIGHT;
		//
		////const index_t_p &idx_s = index_p[s];
		////const index_t_p &idx_t = bindex_p[t];

		//NodeID *vs = index_p[s].spt_v;
		//NodeID *vt = bindex_p[t].spt_v;
		//EdgeWeight* ws = index_p[s].spt_d;
		//EdgeWeight* wt = bindex_p[t].spt_d;


		//_mm_prefetch(vs, _MM_HINT_T0);
		//_mm_prefetch(vt, _MM_HINT_T0);
		//_mm_prefetch(ws, _MM_HINT_T0); 
		//_mm_prefetch(wt, _MM_HINT_T0);

		//for (unsigned i = 0, j = 0; ; ) {
		//	if (*(vs + i) == *(vt + j)) {
		//		if (*(vs + i) == numOfVertices) break;  // Sentinel
		//		EdgeWeight td = *(ws + i) + *(wt + j);
		//		if (td < distance) distance = td;
		//		++i;
		//		++j;
		//	}
		//	else {
		//		i += *(vs + i) < *(vt + j) ? 1 : 0;
		//		j += *(vs + i) > *(vt + j) ? 1 : 0;
		//	}
		//}
		//return distance;

		
		EdgeWeight distance = INF_WEIGHT;

		const index_t_p &idx_s = index_p[s];
		const index_t_p &idx_t = bindex_p[t];


		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == v2) {
				if (v1 == numOfVertices) break;  //two_index_t_p Sentinel
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) distance = td;
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}
		return distance; 
	}

	EdgeWeight query_with_info(NodeID s, NodeID t, query_info& q_info) {

		double stime = GetCurrentTimeSec();

		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

	//	vector<NodeID>& index_t = index_[t].spt_v;
	//	vector<EdgeWeight>& index_t_d = index_[t].spt_d;	
		vector<NodeID>& bindex_t = bindex_[t].spt_v;
		vector<EdgeWeight>& bindex_t_d = bindex_[t].spt_d;

		q_info.meet_node = numOfVertices;
		double meet_distance;

		for (int i = 0, j = 0; i < index_s.size(), j < bindex_t.size(); ) {
			if (index_s[i] == bindex_t[j]) {
				meet_distance = (EdgeWeight)(index_s_d[i++] + bindex_t[j++]);
				if (distance >  meet_distance) {
					distance = meet_distance;
					q_info.meet_node = index_s[i];
				}
			}
			else {
				if (index_s[i] < bindex_t[j])
					++i;
				else
					++j;
			}
		};

		stime = GetCurrentTimeSec() - stime;

		q_info.time_cost = stime;

		if (index_s.size() < bindex_t.size())
			q_info.search_len = index_s.size();
		else
			q_info.search_len = bindex_t.size();

		return distance;
	}


	void append(NodeID v, NodeID root, EdgeWeight distance, bool forward) { // forward(backward) search from root to vertex v.
		if (forward) { // forward search from root to vertex v, hence append (root, distance) to backward index of vertex v.
			bindex_[v].spt_v.push_back(root);
			bindex_[v].spt_d.push_back(distance);
		}
		else { // backward search from root to vertex v, hence append (root, distance) to forward index of vertex v.
			index_[v].spt_v.push_back(root);
			index_[v].spt_d.push_back(distance);
		}
	}

	void Free() {
		if (index_.size() == 0 || bindex_.size() == 0) return;
		for (int v = 0; v < numOfVertices; ++v) {
			index_[v].spt_v.clear();
			index_[v].spt_d.clear();
			if (DIRECTED_FLAG == true) {
				bindex_[v].spt_v.clear();
				bindex_[v].spt_d.clear();
			}
		}
		index_.clear();
		bindex_.clear();
	}

	double avg_size() {
		double total = 0;
		for (int i = 0; i < numOfVertices; ++i) {
			total += index_[i].spt_v.size() ;
			total += bindex_[i].spt_v.size();
		}

		double avg = total / numOfVertices / 2 - 1; // We do not count the trivial labels (V, INF_WEIGHT).

		return avg;
	}

	void print_stat() {
		cout << "Average Label Size: " << avg_size() << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}

	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			int isize = index_[v].size();
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				ofs.write((const char*)&index_[v].spt_v[i], sizeof(index_[v].spt_v[i]));
				ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
			}
			int bisize = bindex_[v].size();
			ofs.write((const char*)&bisize, sizeof(bisize));
			for (NodeID i = 0; i < bindex_[v].size(); ++i) {
				ofs.write((const char*)&bindex_[v].spt_v[i], sizeof(bindex_[v].spt_v[i]));
				ofs.write((const char*)&bindex_[v].spt_d[i], sizeof(bindex_[v].spt_d[i]));
			}
		}
		ofs.close();
	}

	void load_labels(const char* load_filename) {
		cout << "Loading Labels" << endl;
/*
		for (NodeID v = 0; v < numOfVertices; ++v) {
			free(index_p[v].spt_v);
			free(index_p[v].spt_d);
		}*/

		//free(index_p);
		index_p = NULL;
		bindex_p = NULL;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		index_p = (index_t_p*)memalign(64, numOfVertices * sizeof(index_t_p));
		bindex_p = (index_t_p*)memalign(64, numOfVertices * sizeof(index_t_p));

		cout << numOfVertices << " vertices." << endl;

		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_p &idx = index_p[v];
			ifs.read((char*)&isize, sizeof(isize));

			idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));

			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_v[i] = hub;
				idx.spt_d[i] = hub_weight;

			}

			//	index_[v].spt_v.resize(isize);
			//	index_[v].spt_d.resize(isize);


			index_t_p &bidx = bindex_p[v];
			ifs.read((char*)&isize, sizeof(isize));
			bidx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			bidx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));

			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;
				bidx.spt_v[i] = hub;
				bidx.spt_d[i] = hub_weight;

			}

		}
		ifs.close();

		/*
		index_.clear();
		bindex_.clear();
		ifs.open(load_filename, ios::binary | ios::in);
		
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		index_.resize(numOfVertices);
		bindex_.resize(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {

			ifs.read((char*)&isize, sizeof(isize));
			index_[v].spt_v.resize(isize);
			index_[v].spt_d.resize(isize);
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				index_[v].spt_v[i] = hub;
				index_[v].spt_d[i] = hub_weight;
			}

			ifs.read((char*)&isize, sizeof(isize));
			bindex_[v].spt_v.resize(isize);
			bindex_[v].spt_d.resize(isize); 			
			for (NodeID i = 0; i < bindex_[v].size(); ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				bindex_[v].spt_v[i] = hub;
				bindex_[v].spt_d[i] = hub_weight;
			}
		}
		ifs.close();
		*/
	/*	for (int i = 0; i < numOfVertices; ++i) {
			for (int j = 0; j < index_[i].size(); ++j)
				if (index_[i].spt_v[j] != index_p[i].spt_v[j])
					cout << "warning." << endl;
		}*/
		
	}


	
	void convert_to_fewerbit(){
		
		two_index_p = NULL;
		b_two_index_p = NULL;
		two_index_p = (two_index_t_p*)memalign(64, numOfVertices * sizeof(two_index_t_p));
		b_two_index_p = (two_index_t_p*)memalign(64, numOfVertices * sizeof(two_index_t_p));

		for (NodeID v = 0; v < numOfVertices; ++v) {
			two_index_t_p &idx = two_index_p[v];
			
			index_t_p &idx_original = index_p[v];
			
			NodeID isize = 0;
			for(NodeID i = 0; idx_original.spt_v[i] < UCHAR_MAX; ++i){
				++isize;
			}
			

			idx.spt_lv = (uint8_t*)memalign(64, (isize + 1) * sizeof(uint8_t));
			idx.spt_ld = (EdgeWeight*)memalign(64, (isize + 1) * sizeof(EdgeWeight));

		//	index_[v].spt_v.resize(isize);
		//	index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < isize; ++i) {
				uint8_t hub;
				EdgeWeight hub_weight;
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_lv[i] = idx_original.spt_v[i];
				idx.spt_ld[i] = idx_original.spt_d[i];

			}
			 
			idx.spt_lv[isize] = UCHAR_MAX;
			idx.spt_ld[isize] = INF_WEIGHT;
			
			NodeID larger_size = 0;
			for(NodeID i = isize; idx_original.spt_v[i] != numOfVertices; ++i){
				++larger_size;
			}
			
			
			idx.spt_v = (NodeID*)memalign(64, larger_size * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, larger_size * sizeof(EdgeWeight));
			
			for (NodeID i = 0; i < larger_size; ++i) {
				uint8_t hub;
				EdgeWeight hub_weight;
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_v[i] = idx_original.spt_v[i + isize];
				idx.spt_d[i] = idx_original.spt_d[i + isize];

			}			
			
			two_index_t_p &b_idx = b_two_index_p[v];
			
			index_t_p &b_idx_original = bindex_p[v];
			
			isize = 0;
			for(NodeID i = 0; b_idx_original.spt_v[i] < UCHAR_MAX; ++i){
				++isize;
			}
			

			b_idx.spt_lv = (uint8_t*)memalign(64, (isize + 1) * sizeof(uint8_t));
			b_idx.spt_ld = (EdgeWeight*)memalign(64, (isize + 1) * sizeof(EdgeWeight));

		//	index_[v].spt_v.resize(isize);
		//	index_[v].spt_d.resize(isize);

			for (NodeID i = 0; i < isize; ++i) {
				uint8_t hub;
				EdgeWeight hub_weight;
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				b_idx.spt_lv[i] = b_idx_original.spt_v[i];
				b_idx.spt_ld[i] = b_idx_original.spt_d[i];

			}
			 
			b_idx.spt_lv[isize] = UCHAR_MAX;
			b_idx.spt_ld[isize] = INF_WEIGHT;
			
			larger_size = 0;
			for(NodeID i = isize; b_idx_original.spt_v[i] != numOfVertices; ++i){
				++larger_size;
			}
			
			
			b_idx.spt_v = (NodeID*)memalign(64, larger_size * sizeof(NodeID));
			b_idx.spt_d = (EdgeWeight*)memalign(64, larger_size * sizeof(EdgeWeight));
			
			for (NodeID i = 0; i < larger_size; ++i) {
				uint8_t hub;
				EdgeWeight hub_weight;
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;
 
				b_idx.spt_v[i] = b_idx_original.spt_v[i + isize];
				b_idx.spt_d[i] = b_idx_original.spt_d[i + isize];

			}			
		}		
	}
	
	
	void save_labels_iteration_stats(const char* save_filename) {

		vector<NodeID> stat(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {
			for (NodeID i = 0; i < index_[v].size(); ++i)
				stat[index_[v].spt_v[i]]++;
			for (NodeID i = 0; i < bindex_[v].size(); ++i)
				stat[bindex_[v].spt_v[i]]++;
		}

		ofstream ofs(save_filename);

		for (NodeID v = 0; v < numOfVertices; ++v) {
			ofs << stat[v] << endl;
		}
		ofs.close();
	}

};

class DPLabel{
public:
	vector<index_t_path> index_;
	vector<index_t_path> bindex_; // Backward labels.

	index_t_path_p* index_p;
	index_t_path_p* bindex_p;


	DPLabel() {
		index_.resize(numOfVertices);
		bindex_.resize(numOfVertices);
	}

	~DPLabel() {
		Free();
	}

	inline EdgeWeight query_path(NodeID s, NodeID t, vector<NodeID>& rank, vector<NodeID>& inv) {



		EdgeWeight distance = INF_WEIGHT;
		NodeID meetnode = numOfVertices;
		NodeID s_parent;
		NodeID t_parent;

		const index_t_path_p &idx_s = index_p[s];
		const index_t_path_p &idx_t = bindex_p[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_p[0], _MM_HINT_T0); 
		_mm_prefetch(&idx_t.spt_p[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) {
					distance = td;
					//if (v1 < meetnode) {
						meetnode = v1;
						s_parent = idx_s.spt_p[i];
						t_parent = idx_t.spt_p[j];
				//	}
				}
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}

		//Next, retrieve path from s - meetnode and meetnode - t.
		vector<NodeID> path_from_s;
		vector<NodeID> path_to_t;
		path_from_s.push_back(s_parent);
		path_to_t.push_back(t_parent);

		/*	if (s == 194569 && t == 20072)
		cout << "debug." << " meet: " << meetnode << " sparent:" << s_parent << " tparent:" << t_parent <<  endl;*/

		NodeID inv_meetnode = inv[meetnode];

		while (path_from_s.back() != inv_meetnode) {
			/*if (s == 194569 && t == 20072)
			cout << "s meet:" << path_from_s.back() << endl;*/
			const index_t_path_p &idx_from_s = index_p[path_from_s.back()];

			_mm_prefetch(&idx_from_s.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_from_s.spt_p[0], _MM_HINT_T0);

			//	vector<NodeID>& index_from_s = index_[path_from_s.back()].spt_v;
			for (int i = 0; ; ++i) {
				if (idx_from_s.spt_v[i] == numOfVertices) break;
				if (idx_from_s.spt_v[i] == meetnode) {
					path_from_s.push_back(idx_from_s.spt_p[i]);
					break;
				}
			}
		}

		while (path_to_t.back() != inv_meetnode) {
			/*if (s == 194569 && t == 20072)
			cout << "t meet:" << path_to_t.back() << endl;*/
			//	vector<NodeID>& index_to_t = index_[path_to_t.back()].spt_v;
			const index_t_path_p &idx_to_t = bindex_p[path_to_t.back()];
			_mm_prefetch(&idx_to_t.spt_v[0], _MM_HINT_T0);
			_mm_prefetch(&idx_to_t.spt_p[0], _MM_HINT_T0);
			for (int i = 0; ; ++i) {
				if (idx_to_t.spt_v[i] == numOfVertices) break;
				if (idx_to_t.spt_v[i] == meetnode) {
					path_to_t.push_back(idx_to_t.spt_p[i]);
					break;
				}
			}
		}

		return distance;
		
	}

	EdgeWeight query_path_p(NodeID s, NodeID t, vector<NodeID>& rank, vector<NodeID>& inv) {

		EdgeWeight distance = INF_WEIGHT;
		vector<NodeID>& index_s = index_[s].spt_v;
		vector<EdgeWeight>& index_s_d = index_[s].spt_d;

		vector<NodeID>& bindex_t = bindex_[t].spt_v;
		vector<EdgeWeight>& bindex_t_d = bindex_[t].spt_d;

		
		NodeID meetnode = numOfVertices;
		int s_parent;
		int t_parent;
		for (int i = 0, j = 0; i < index_s.size(), j < bindex_t.size(); ) {
			if (index_s[i] == bindex_t[j]) {
				if (distance >(EdgeWeight)(index_s_d[i] + bindex_t_d[j])) {
					distance = (EdgeWeight)(index_s_d[i] + bindex_t_d[j]);
				//	if (index_s[i] < meetnode) {
						meetnode = index_s[i];
						s_parent = index_[s].spt_p[i];
						t_parent = index_[t].spt_p[j];
				//	}
				}
				//distance = min(distance, (EdgeWeight)(index_s_d[i] + bindex_t_d[j]));
				++i;
				++j;
			}
			else {
				if (index_s[i] < bindex_t[j])
					++i;
				else
					++j;
			}
		}

		//Next, retrieve path from s - meetnode and meetnode - t.
		vector<NodeID> path_from_s;
		vector<NodeID> path_to_t;
		path_from_s.push_back(s_parent);
		path_to_t.push_back(t_parent);

		/*	if (s == 194569 && t == 20072)
		cout << "debug." << " meet: " << meetnode << " sparent:" << s_parent << " tparent:" << t_parent <<  endl;*/

		while (path_from_s.back() != inv[meetnode]) {
			/*if (s == 194569 && t == 20072)
			cout << "s meet:" << path_from_s.back() << endl;*/
			vector<NodeID>& index_from_s = index_[path_from_s.back()].spt_v;
			for (int i = 0; i < index_from_s.size(); ++i) {
				if (index_from_s[i] == meetnode) {
					path_from_s.push_back(index_[path_from_s.back()].spt_p[i]);
					break;
				}
			}
		}

		while (path_to_t.back() != inv[meetnode]) {
			/*if (s == 194569 && t == 20072)
			cout << "t meet:" << path_to_t.back() << endl;*/
			vector<NodeID>& index_to_t = bindex_[path_to_t.back()].spt_v;
			for (int i = 0; i < index_to_t.size(); ++i) {
				if (index_to_t[i] == meetnode) {
					path_to_t.push_back(bindex_[path_to_t.back()].spt_p[i]);
					break;
				}
			}
		}

		//for (int i = 0; i < path_from_s.size(); ++i)
		//	path_from_s[i] = inv[path_from_s[i]];
		//for (int i = 0; i < path_to_t.size(); ++i)
		//	path_to_t[i] = inv[path_to_t[i]];

		return path_from_s.size() + path_to_t.size();
	}


	void Free() {
		if (index_.size() == 0 || bindex_.size() == 0) return;
		for (int v = 0; v < numOfVertices; ++v) {
			index_[v].spt_v.clear();
			index_[v].spt_d.clear();
			if (DIRECTED_FLAG == true) {
				bindex_[v].spt_v.clear();
				bindex_[v].spt_d.clear();
			}
		}
		index_.clear();
		bindex_.clear();
	}

	double avg_size() {
		double total = 0;
		for (int i = 0; i < numOfVertices; ++i) {
			total += index_[i].spt_v.size();
			total += bindex_[i].spt_v.size();
		}
		double avg = total / numOfVertices / 2 - 1; // We do not count the trivial labels (V, INF_WEIGHT).

		return avg;
	}

	void print_stat() {
		cout << "Average Label Size: " << avg_size() << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}

	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		for (NodeID v = 0; v < numOfVertices; ++v) {
			int isize = index_[v].size();
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				ofs.write((const char*)&index_[v].spt_v[i], sizeof(index_[v].spt_v[i]));
				ofs.write((const char*)&index_[v].spt_p[i], sizeof(index_[v].spt_p[i]));
				ofs.write((const char*)&index_[v].spt_d[i], sizeof(index_[v].spt_d[i]));
			}
			int bisize = bindex_[v].size();
			ofs.write((const char*)&bisize, sizeof(bisize));
			for (NodeID i = 0; i < bindex_[v].size(); ++i) {
				ofs.write((const char*)&bindex_[v].spt_v[i], sizeof(bindex_[v].spt_v[i]));
				ofs.write((const char*)&bindex_[v].spt_p[i], sizeof(bindex_[v].spt_p[i]));
				ofs.write((const char*)&bindex_[v].spt_d[i], sizeof(bindex_[v].spt_d[i]));
			}
		}
		ofs.close();
	}

	void load_labels(const char* load_filename) {

		index_p = NULL;
		bindex_p = NULL;

		ifstream ifs(load_filename, ios::binary | ios::in);

		NodeID isize;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;

		index_p = (index_t_path_p*)memalign(64, numOfVertices * sizeof(index_t_path_p));
		bindex_p = (index_t_path_p*)memalign(64, numOfVertices * sizeof(index_t_path_p));

		cout << numOfVertices << " vertices." << endl;

		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_path_p &idx = index_p[v];
			ifs.read((char*)&isize, sizeof(isize));

			idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_p = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));

			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				NodeID hub_parent;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_parent, sizeof(hub_parent));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;

				idx.spt_v[i] = hub;
				idx.spt_d[i] = hub_weight;
				idx.spt_p[i] = hub_parent;
			}

			//	index_[v].spt_v.resize(isize);
			//	index_[v].spt_d.resize(isize);


			index_t_path_p &bidx = bindex_p[v];
			ifs.read((char*)&isize, sizeof(isize));
			bidx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			bidx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));
			bidx.spt_p = (NodeID*)memalign(64, isize * sizeof(NodeID));

			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				NodeID hub_parent;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_parent, sizeof(hub_parent));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				//index_[v].spt_v[i] = hub;
				//index_[v].spt_d[i] = hub_weight;
				bidx.spt_v[i] = hub;
				bidx.spt_d[i] = hub_weight;
				bidx.spt_p[i] = hub_parent;
			}

		}
		ifs.close();

		/*index_.clear();
		bindex_.clear();
		ifstream ifs(load_filename, ios::binary | ios::in);
		NodeID isize;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		index_.resize(numOfVertices);
		bindex_.resize(numOfVertices);
		for (NodeID v = 0; v < numOfVertices; ++v) {
			ifs.read((char*)&isize, sizeof(isize));
			index_[v].spt_v.resize(isize);
			index_[v].spt_p.resize(isize);
			index_[v].spt_d.resize(isize);
			for (NodeID i = 0; i < index_[v].size(); ++i) {
				NodeID hub;
				NodeID parent;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&parent, sizeof(parent));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				index_[v].spt_v[i] = hub;
				index_[v].spt_p[i] = parent;
				index_[v].spt_d[i] = hub_weight;
			}

			ifs.read((char*)&isize, sizeof(isize));
			bindex_[v].spt_v.resize(isize);
			bindex_[v].spt_d.resize(isize);
			for (NodeID i = 0; i < bindex_[v].size(); ++i) {
				NodeID hub;
				NodeID parent;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&parent, sizeof(parent));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));
				bindex_[v].spt_v[i] = hub;
				bindex_[v].spt_p[i] = parent;
				bindex_[v].spt_d[i] = hub_weight;
			}
		}
		ifs.close();*/
	}
	
	inline EdgeWeight query_p(NodeID s, NodeID t) {

		//EdgeWeight distance = INF_WEIGHT;
		//
		////const index_t_p &idx_s = index_p[s];
		////const index_t_p &idx_t = bindex_p[t];

		//NodeID *vs = index_p[s].spt_v;
		//NodeID *vt = bindex_p[t].spt_v;
		//EdgeWeight* ws = index_p[s].spt_d;
		//EdgeWeight* wt = bindex_p[t].spt_d;


		//_mm_prefetch(vs, _MM_HINT_T0);
		//_mm_prefetch(vt, _MM_HINT_T0);
		//_mm_prefetch(ws, _MM_HINT_T0); 
		//_mm_prefetch(wt, _MM_HINT_T0);

		//for (unsigned i = 0, j = 0; ; ) {
		//	if (*(vs + i) == *(vt + j)) {
		//		if (*(vs + i) == numOfVertices) break;  // Sentinel
		//		EdgeWeight td = *(ws + i) + *(wt + j);
		//		if (td < distance) distance = td;
		//		++i;
		//		++j;
		//	}
		//	else {
		//		i += *(vs + i) < *(vt + j) ? 1 : 0;
		//		j += *(vs + i) > *(vt + j) ? 1 : 0;
		//	}
		//}
		//return distance;


		EdgeWeight distance = INF_WEIGHT;

		const index_t_path_p &idx_s = index_p[s];
		const index_t_path_p &idx_t = bindex_p[t];


		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == v2) {
				if (v1 == numOfVertices) break;  // Sentinel
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) distance = td;
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}
		return distance;
	}


};

template<int kNumBitParallelRoots = 50>
class BPLabel {

public:
	index_t_bp<kNumBitParallelRoots>* index_bp;
	
	BPLabel() {
	}

	~BPLabel() {
		//Free();
	}

	EdgeWeight query_p(NodeID s, NodeID t) {

		EdgeWeight distance = INF_WEIGHT;

		NodeID *vs = index_bp[s].spt_v;
		NodeID *vt = index_bp[t].spt_v;
		EdgeWeight* ws = index_bp[s].spt_d;
		EdgeWeight* wt = index_bp[t].spt_d;

		_mm_prefetch(vs, _MM_HINT_T0);
		_mm_prefetch(vt, _MM_HINT_T0);
		_mm_prefetch(ws, _MM_HINT_T0);
		_mm_prefetch(wt, _MM_HINT_T0);

		for (int i = 0; i < kNumBitParallelRoots; ++i) {
			EdgeWeight td = index_bp[s].bpspt_d[i] + index_bp[t].bpspt_d[i];
			if (td - 2 <= distance) {
				td +=
					(index_bp[s].bpspt_s[i][0] & index_bp[t].bpspt_s[i][0]) ? -2 :
					((index_bp[s].bpspt_s[i][0] & index_bp[t].bpspt_s[i][1]) | (index_bp[s].bpspt_s[i][1] & index_bp[t].bpspt_s[i][0]))
					? -1 : 0;

				if (td < distance) distance = td;
			}
		}

		for (unsigned i = 0, j = 0; ; ) {
			if (*(vs + i) == *(vt + j)) {
				if (*(vs + i) == numOfVertices) break;  // Sentinel
				EdgeWeight td = *(ws + i) + *(wt + j);
				if (td < distance) distance = td;
				++i;
				++j;
			}
			else {
				i += *(vs + i) < *(vt + j) ? 1 : 0;
				j += *(vs + i) > *(vt + j) ? 1 : 0;
			}
		}
		return distance;
	}
	
	EdgeWeight query_p(NodeID s, NodeID t, bool& isBP) {

		EdgeWeight distance = INF_WEIGHT;

		const index_t_bp<kNumBitParallelRoots> &idx_s = index_bp[s];
		const index_t_bp<kNumBitParallelRoots> &idx_t = index_bp[t];

		_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
		_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
		_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

		isBP = false;

		for (int i = 0; i < kNumBitParallelRoots; ++i) {
			EdgeWeight td = index_bp[s].bpspt_d[i] + index_bp[t].bpspt_d[i];
			if (td - 2 <= distance) {
				td +=
					(index_bp[s].bpspt_s[i][0] & index_bp[t].bpspt_s[i][0]) ? -2 :
					((index_bp[s].bpspt_s[i][0] & index_bp[t].bpspt_s[i][1]) | (index_bp[s].bpspt_s[i][1] & index_bp[t].bpspt_s[i][0]))
					? -1 : 0;

				if (td < distance) {
					distance = td;
					isBP = true;
				}
			}
		}

		for (int i = 0, j = 0; ; ) {
			NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

			if (v1 == numOfVertices) break;  // Sentinel

			if (v1 == v2) {
				EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
				if (td < distance) {
					distance = td;
					isBP = false;
				}
				++i;
				++j;
			}
			else {
				i += v1 < v2 ? 1 : 0;
				j += v1 > v2 ? 1 : 0;
			}
		}
		return distance;
	}


	/*
	NodeID max_size() {
	NodeID maxsize = numeric_limits<NodeID>::min();
	for (int i = 0; i < V; ++i) maxsize = max(maxsize, index_[i].spt_v.size());
	return maxsize;
	}*/

	void print_stat() {
		cout << "Average Label Size: " << avg_size() << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}

	double avg_size() {
		double lab_count = 0;
		for (NodeID v = 0; v < numOfVertices; ++v) {
			NodeID isize;
			for (isize = 1; index_bp[v].spt_v[isize - 1] != numOfVertices; ++isize) continue;
			lab_count += isize;
		}
		lab_count = (double)lab_count / (double)numOfVertices - 1;
		return lab_count;
	}

	void Free() {
		for (int v = 0; v < numOfVertices; ++v) {
			free(index_bp[v].spt_v);
			free(index_bp[v].spt_d);
	
		}

		free(index_bp);
		index_bp = NULL;
	}
	
	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		int knumbit = kNumBitParallelRoots;
		ofs.write((const char*)&knumbit, sizeof(knumbit));

		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_bp<kNumBitParallelRoots> &idx = index_bp[v];

			for (int i = 0; i < kNumBitParallelRoots; ++i) {
				EdgeWeight d = idx.bpspt_d[i];
				uint64_t a = idx.bpspt_s[i][0];
				uint64_t b = idx.bpspt_s[i][1];
				ofs.write((const char*)&d, sizeof(d));
				ofs.write((const char*)&a, sizeof(a));
				ofs.write((const char*)&b, sizeof(b));
			}

			NodeID isize;
			for (isize = 1; idx.spt_v[isize - 1] != numOfVertices; ++isize) continue;  // Find the sentinel
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < isize; ++i) {
				ofs.write((const char*)&idx.spt_v[i], sizeof(idx.spt_v[i]));
				ofs.write((const char*)&idx.spt_d[i], sizeof(idx.spt_d[i]));
			}
		}

		ofs.close();
	}
	
	 
	void load_labels(const char* load_filename){
		
		index_bp = NULL;

		int knumbit;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;		
		ifs.read((char*)&knumbit, sizeof(isize));

		if (knumbit != kNumBitParallelRoots) {
			cout << knumbit << "!=" << kNumBitParallelRoots << endl;
			return;
		}
		

		index_bp = (index_t_bp<kNumBitParallelRoots>*)memalign(64, numOfVertices * sizeof(index_t_bp<kNumBitParallelRoots>));
		

		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_bp<kNumBitParallelRoots> &idx = index_bp[v];

			for (int i = 0; i < kNumBitParallelRoots; ++i) {

				//idx.bpspt_s[i] = (uint64_t*)memalign(64, 2 * sizeof(uint64_t));

				EdgeWeight d;
				uint64_t a, b;

				ifs.read((char*)&d, sizeof(EdgeWeight));
				ifs.read((char*)&a, sizeof(uint64_t));
				ifs.read((char*)&b, sizeof(uint64_t));
				idx.bpspt_d[i] = d;
				idx.bpspt_s[i][0] = a;
				idx.bpspt_s[i][1] = b;
			}

			ifs.read((char*)&isize, sizeof(isize));

			idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));


			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));

				idx.spt_v[i] = hub;
				idx.spt_d[i] = hub_weight;

			}
		}
		ifs.close();
	}


};

template<int kNumBitParallelRoots = 50>
class DBPLabel {

public:
	index_t_bp<kNumBitParallelRoots>* index_bp;
	index_t_bp<kNumBitParallelRoots>* bindex_bp;
	DBPLabel() {
	}

	~DBPLabel() {
	}

	/*EdgeWeight query_p(NodeID s, NodeID t) {

	EdgeWeight distance = INF_WEIGHT;

	NodeID *vs = index_p[s].spt_v;
	NodeID *vt = index_p[t].spt_v;
	EdgeWeight* ws = index_p[s].spt_d;
	EdgeWeight* wt = index_p[t].spt_d;

	_mm_prefetch(vs, _MM_HINT_T0);
	_mm_prefetch(vt, _MM_HINT_T0);
	_mm_prefetch(ws, _MM_HINT_T0);
	_mm_prefetch(wt, _MM_HINT_T0);

	for (unsigned i = 0, j = 0; ; ) {
	if (*(vs + i) == *(vt + j)) {
	if (*(vs + i) == numOfVertices) break;  // Sentinel
	EdgeWeight td = *(ws + i) + *(wt + j);
	if (td < distance) distance = td;
	++i;
	++j;
	}
	else {
	i += *(vs + i) < *(vt + j) ? 1 : 0;
	j += *(vs + i) > *(vt + j) ? 1 : 0;
	}
	}
	return distance;


	//EdgeWeight distance = INF_WEIGHT;

	//const index_t_p &idx_s = index_p[s];
	//const index_t_p &idx_t = index_p[t];

	//_mm_prefetch(&idx_s.spt_v[0], _MM_HINT_T0);
	//_mm_prefetch(&idx_t.spt_v[0], _MM_HINT_T0);
	//_mm_prefetch(&idx_s.spt_d[0], _MM_HINT_T0);
	//_mm_prefetch(&idx_t.spt_d[0], _MM_HINT_T0);

	//for (int i = 0, j = 0; ; ) {
	//	NodeID v1 = idx_s.spt_v[i], v2 = idx_t.spt_v[j];

	//	if (v1 == numOfVertices) break;  // Sentinel

	//	if (v1 == v2) {
	//		EdgeWeight td = idx_s.spt_d[i] + idx_t.spt_d[j];
	//		if (td < distance) distance = td;
	//		++i;
	//		++j;
	//	}
	//	else {
	//		i += v1 < v2 ? 1 : 0;
	//		j += v1 > v2 ? 1 : 0;
	//	}
	//}
	//return distance;
	}
	*/

	/*
	NodeID max_size() {
	NodeID maxsize = numeric_limits<NodeID>::min();
	for (int i = 0; i < V; ++i) maxsize = max(maxsize, index_[i].spt_v.size());
	return maxsize;
	}*/

	EdgeWeight query_p(NodeID s, NodeID t) {

		EdgeWeight distance = INF_WEIGHT;

		NodeID *vs = index_bp[s].spt_v;
		NodeID *vt = bindex_bp[t].spt_v;
		EdgeWeight* ws = index_bp[s].spt_d;
		EdgeWeight* wt = bindex_bp[t].spt_d;

		_mm_prefetch(vs, _MM_HINT_T0);
		_mm_prefetch(vt, _MM_HINT_T0);
		_mm_prefetch(ws, _MM_HINT_T0);
		_mm_prefetch(wt, _MM_HINT_T0);

		for (int i = 0; i < kNumBitParallelRoots; ++i) {
			EdgeWeight td = index_bp[s].bpspt_d[i] + bindex_bp[t].bpspt_d[i];
			if (td - 2 <= distance) {
				td +=
					(index_bp[s].bpspt_s[i][0] & bindex_bp[t].bpspt_s[i][0]) ? -2 :
					((index_bp[s].bpspt_s[i][0] & bindex_bp[t].bpspt_s[i][1]) | (index_bp[s].bpspt_s[i][1] & bindex_bp[t].bpspt_s[i][0]))
					? -1 : 0;

				if (td < distance) distance = td;
			}
		}

		for (unsigned i = 0, j = 0; ; ) {
			if (*(vs + i) == *(vt + j)) {
				if (*(vs + i) == numOfVertices) break;  // Sentinel
				EdgeWeight td = *(ws + i) + *(wt + j);
				if (td < distance) distance = td;
				++i;
				++j;
			}
			else {
				i += *(vs + i) < *(vt + j) ? 1 : 0;
				j += *(vs + i) > *(vt + j) ? 1 : 0;
			}
		}
		return distance;
	}

	EdgeWeight query_p(NodeID s, NodeID t, bool& isBP) {
		isBP = false;
		EdgeWeight distance = INF_WEIGHT;

		NodeID *vs = index_bp[s].spt_v;
		NodeID *vt = bindex_bp[t].spt_v;
		EdgeWeight* ws = index_bp[s].spt_d;
		EdgeWeight* wt = bindex_bp[t].spt_d;

		_mm_prefetch(vs, _MM_HINT_T0);
		_mm_prefetch(vt, _MM_HINT_T0);
		_mm_prefetch(ws, _MM_HINT_T0);
		_mm_prefetch(wt, _MM_HINT_T0);

		for (int i = 0; i < kNumBitParallelRoots; ++i) {
			EdgeWeight td = index_bp[s].bpspt_d[i] + bindex_bp[t].bpspt_d[i];
			if (td - 2 <= distance) {
				td +=
					(index_bp[s].bpspt_s[i][0] & bindex_bp[t].bpspt_s[i][0]) ? -2 :
					((index_bp[s].bpspt_s[i][0] & bindex_bp[t].bpspt_s[i][1]) | (index_bp[s].bpspt_s[i][1] & bindex_bp[t].bpspt_s[i][0]))
					? -1 : 0;

				if (td < distance) {
					distance = td;
					isBP = true;
				}
			}
		}

		for (unsigned i = 0, j = 0; ; ) {
			if (*(vs + i) == *(vt + j)) {
				if (*(vs + i) == numOfVertices) break;  // Sentinel
				EdgeWeight td = *(ws + i) + *(wt + j);
				if (td < distance) {
					distance = td;
					isBP = false;
				}
				++i;
				++j;
			}
			else {
				i += *(vs + i) < *(vt + j) ? 1 : 0;
				j += *(vs + i) > *(vt + j) ? 1 : 0;
			}
		}
		return distance;
	}



	void print_stat() {
		cout << "Average Label Size: " << avg_size() << endl;
		//cout << "Maximum Label Size: " << max_size() << endl;
	}

	double avg_size() {
		double lab_count = 0;
		for (NodeID v = 0; v < numOfVertices; ++v) {
			NodeID isize;
			for (isize = 1; index_bp[v].spt_v[isize - 1] != numOfVertices; ++isize) continue;
			lab_count += isize;
			for (isize = 1; bindex_bp[v].spt_v[isize - 1] != numOfVertices; ++isize) continue;
		}
		lab_count = (double)lab_count / (double)numOfVertices - 1 / (double)2;
		return lab_count;
	}

	void Free() {
		for (int v = 0; v < numOfVertices; ++v) {
			free(index_bp[v].spt_v);
			free(index_bp[v].spt_d);
			free(index_bp[v].bpspt_d);
			free(index_bp[v].bpspt_s);
			free(bindex_bp[v].spt_v);
			free(bindex_bp[v].spt_d);
			free(bindex_bp[v].bpspt_d);
			free(bindex_bp[v].bpspt_s);
		}

		free(index_bp);
		free(bindex_bp);
		index_bp = NULL;
		bindex_bp = NULL;
	}

	void save_labels(const char* save_filename) {
		ofstream ofs(save_filename, ios::binary | ios::out);

		ofs.write((const char*)&numOfVertices, sizeof(numOfVertices));
		int knumbit = kNumBitParallelRoots;
		ofs.write((const char*)&knumbit, sizeof(knumbit));

		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_bp<kNumBitParallelRoots> &idx = index_bp[v];
			index_t_bp<kNumBitParallelRoots> &r_idx = bindex_bp[v];

			for (int i = 0; i < kNumBitParallelRoots; ++i) {
				EdgeWeight d = idx.bpspt_d[i];
				uint64_t a = idx.bpspt_s[i][0];
				uint64_t b = idx.bpspt_s[i][1];
				ofs.write((const char*)&d, sizeof(d));
				ofs.write((const char*)&a, sizeof(a));
				ofs.write((const char*)&b, sizeof(b));
			}

			for (int i = 0; i < kNumBitParallelRoots; ++i) {
				EdgeWeight d = r_idx.bpspt_d[i];
				uint64_t a = r_idx.bpspt_s[i][0];
				uint64_t b = r_idx.bpspt_s[i][1];
				ofs.write((const char*)&d, sizeof(d));
				ofs.write((const char*)&a, sizeof(a));
				ofs.write((const char*)&b, sizeof(b));
			}

			NodeID isize;
			for (isize = 1; idx.spt_v[isize - 1] != numOfVertices; ++isize) continue;  // Find the sentinel
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < isize; ++i) {
				ofs.write((const char*)&idx.spt_v[i], sizeof(idx.spt_v[i]));
				ofs.write((const char*)&idx.spt_d[i], sizeof(idx.spt_d[i]));
			}
			for (isize = 1; r_idx.spt_v[isize - 1] != numOfVertices; ++isize) continue;  // Find the sentinel
			ofs.write((const char*)&isize, sizeof(isize));
			for (NodeID i = 0; i < isize; ++i) {
				ofs.write((const char*)&r_idx.spt_v[i], sizeof(r_idx.spt_v[i]));
				ofs.write((const char*)&r_idx.spt_d[i], sizeof(r_idx.spt_d[i]));
			}
		}

		ofs.close();
	}


	void load_labels(const char* load_filename) {

		index_bp = NULL;

		int knumbit;

		ifstream ifs(load_filename);
		NodeID isize = 0;
		ifs.read((char*)&isize, sizeof(isize));
		numOfVertices = isize;
		ifs.read((char*)&knumbit, sizeof(isize));

		if (knumbit != kNumBitParallelRoots) {
			cout << knumbit << "!=" << kNumBitParallelRoots << endl;
			return;
		}


		index_bp = (index_t_bp<kNumBitParallelRoots>*)memalign(64, numOfVertices * sizeof(index_t_bp<kNumBitParallelRoots>));
		bindex_bp = (index_t_bp<kNumBitParallelRoots>*)memalign(64, numOfVertices * sizeof(index_t_bp<kNumBitParallelRoots>));


		for (NodeID v = 0; v < numOfVertices; ++v) {
			index_t_bp<kNumBitParallelRoots> &idx = index_bp[v];
			index_t_bp<kNumBitParallelRoots> &r_idx = bindex_bp[v];

			for (int i = 0; i < kNumBitParallelRoots; ++i) {

				//idx.bpspt_s[i] = (uint64_t*)memalign(64, 2 * sizeof(uint64_t));

				EdgeWeight d;
				uint64_t a, b;

				ifs.read((char*)&d, sizeof(EdgeWeight));
				ifs.read((char*)&a, sizeof(uint64_t));
				ifs.read((char*)&b, sizeof(uint64_t));

				idx.bpspt_d[i] = d;
				idx.bpspt_s[i][0] = a;
				idx.bpspt_s[i][1] = b;
			}
			for (int i = 0; i < kNumBitParallelRoots; ++i) {

				//idx.bpspt_s[i] = (uint64_t*)memalign(64, 2 * sizeof(uint64_t));

				EdgeWeight d;
				uint64_t a, b;

				ifs.read((char*)&d, sizeof(EdgeWeight));
				ifs.read((char*)&a, sizeof(uint64_t));
				ifs.read((char*)&b, sizeof(uint64_t));

				r_idx.bpspt_d[i] = d;
				r_idx.bpspt_s[i][0] = a;
				r_idx.bpspt_s[i][1] = b;
			}

			ifs.read((char*)&isize, sizeof(isize));

			idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));


			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));

				idx.spt_v[i] = hub;
				idx.spt_d[i] = hub_weight;

			}


			ifs.read((char*)&isize, sizeof(isize));

			r_idx.spt_v = (NodeID*)memalign(64, isize * sizeof(NodeID));
			r_idx.spt_d = (EdgeWeight*)memalign(64, isize * sizeof(EdgeWeight));
			for (NodeID i = 0; i < isize; ++i) {
				NodeID hub;
				EdgeWeight hub_weight;
				ifs.read((char*)&hub, sizeof(hub));
				ifs.read((char*)&hub_weight, sizeof(hub_weight));

				r_idx.spt_v[i] = hub;
				r_idx.spt_d[i] = hub_weight;
			}
		}
		ifs.close();
	}


};

/***
 * class  FLabel inherented from Label(undirected weighted graph)
 * written by wanjingyi
 * **/
class FLabel : public Label{
	public:
		//variables

        vector<NodeID> inv; //fetch the original ID of a given order
        vector<NodeID> rank; // Fetch the ranking of a given vertex id.
		//construction and deconstruction
		FLabel(){
			inv.clear();
			rank.clear();
		}
		~FLabel(){
			Free();
		}
		/**
		 * function used to read query s-t frequencies from file
		 * written by wanjingyi
		 * return:return: total number of query times
		 * */

};

#endif