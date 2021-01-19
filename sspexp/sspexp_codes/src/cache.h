#pragma once
#ifndef CACHE_H
#define CACHE_H

#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <sstream>
#include "./graph.h"
#include "./paras.h"
#define numOfVertices SP_Constants::numOfVertices
#define numOfEdges SP_Constants::numOfEdges
#define INF_WEIGHT SP_Constants::INF_WEIGHT

/*
 *@description: the base cache stratege
 *@author: wanjingyi
 *@date: 2021-01-15
*/
class Cache{
    public:
        //variables
        NodeID numOfHfpoint;
        EdgeWeight** cache_p;
        bool* is_hfpoint;
        NodeID* cache_rank;
        //NodeID* cache_inv;
        //constrctions and deconstructions
        Cache(){
            numOfHfpoint=0;
            cache_p=NULL;
            cache_rank=NULL;
            is_hfpoint=NULL;
        }
        Cache(char* pointFreqFileName,int hfRate=5){
            is_hfpoint=NULL;
            is_hfpoint=new bool[numOfVertices];
            cache_rank=NULL;
            cache_rank=new NodeID[numOfVertices];
            load_cache_hfpoint(pointFreqFileName,hfRate);
            cache_p=NULL;
            cache_p=new EdgeWeight* [numOfHfpoint];
            for(int i=0;i<numOfHfpoint;++i) cache_p[i]=new EdgeWeight[numOfHfpoint];
            clear_cache();
        }

        ~Cache(){
            //delete pointer
            for(int i=0;i<numOfHfpoint;++i){
                delete [] cache_p[i];
                cache_p[i]=NULL;
            }
            delete [] cache_p;
            cache_p=NULL;
            delete [] is_hfpoint;
            is_hfpoint=NULL;
            delete [] cache_rank;
            cache_rank=NULL;
        }

        /*
         *@description: set all cache distances to INF_WEIGHT
         *@author: wanjingyi
         *@date: 2021-01-15
        */
        bool clear_cache(){
            for(int i=0;i<numOfHfpoint;++i){
                for(int j=0;j<numOfHfpoint;++j){
                    cache_p[i][j]=INF_WEIGHT;
                }
            }
            for(int i=0;i<numOfVertices;++i){
                is_hfpoint[i]=false;
                cache_rank[i]=INF_WEIGHT;
            }
            
        }

        void update_cache(char* load_filename){
            clear_cache();
            ifstream in(load_filename);//input HFPoint file to ifstream
			if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			NodeID id;int t,i=0;
			char line[24];
			//read each line representing HFpoint to vector 
			while (in.getline(line,sizeof(line))){
				stringstream hp(line);
				hp>>id>>t;		
				if(i>=numOfHfpoint)	break;
				is_hfpoint[id]=true;
                cache_rank[id]=i;
				i++;
			}
			if(i<numOfHfpoint){
				numOfHfpoint=i;
				cout<<"real numOfHfpoint = "<<numOfHfpoint<<endl;
			}
        }

    protected:
        /*
		 *@description: load hfpoint that needed to be cached
		 *@author: wanjingyi
		 *return: numOfHfpoint
		 *@date: 2021-01-16
		*/
		int load_cache_hfpoint(char* load_filename,int hfRate=5){//5%%
			numOfHfpoint= 0;//first line is the number of HFpoints
			numOfHfpoint= static_cast<int> ( (double)numOfVertices*hfRate/(double)1000);
			if(numOfHfpoint<=0) cout<<"error:numOfHfpoint<=0"<<endl;
			cout<<"initial numOfHfpoint = "<<numOfHfpoint<<endl;
			ifstream in(load_filename);//input HFPoint file to ifstream
			if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			NodeID id;int t,i=0;
			char line[24];
			//read each line representing HFpoint to vector 
			while (in.getline(line,sizeof(line))){
				stringstream hp(line);
				hp>>id>>t;		
				if(i>=numOfHfpoint)	break;
				is_hfpoint[id]=true;
                cache_rank[id]=i;
				i++;
			}
			if(i<numOfHfpoint){
				numOfHfpoint=i;
				cout<<"real numOfHfpoint = "<<numOfHfpoint<<endl;
			}
		}		

};


#endif //BETWEENNESS_H