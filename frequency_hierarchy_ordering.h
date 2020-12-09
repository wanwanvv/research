#pragma once
#ifndef FREQUENCY_HIERARCHY_ORDERING_H
#define FREQUENCY_HIERARCHY_ORDERING_H

#include<algorithm>
#include<unordered_set>
#include<time.h>
#include "graph.h"
#include "graph_search.h"
#include "labels.h"
#include "time_util.h" 
#include "heap.h"
#include "paras.h"
#include "ordering.h"
#include "construction.h"
#include<unordered_map>  
#include<cmath>
#include <queue>
#ifdef _WIN32
	#include<google/sparsehash/sparseconfig.h>
#endif
    #include<google/dense_hash_map>
	#include<google/dense_hash_set>



#define numOfVertices SP_Constants::numOfVertices
#define numOfEdges SP_Constants::numOfEdges
#define WEIGHTED_FLAG SP_Constants::WEIGHTED_FLAG  
#define DIRECTED_FLAG SP_Constants::DIRECTED_FLAG
#define cover_value_type long long
//DEBUG
#define DEBUG_FLAG 1
char debugFileName[255] = "../dataset/manhatan/SHP/SOrder"; 

//pair compare 
bool cmp(const pair<NodeID,int> a, const pair<NodeID,int> b) {
    return a.second>b.second;//自定义的比较函数
}

struct cmp_queue
    {template <typename T, typename U>
        bool operator()(T const &left, U const &right)
        {
        // 以 second 比较。输出结果为 Second 较大的在前 Second 相同时，先进入队列的元素在前。
            if (left.second < right.second)
                return true;
            return false;
        }
    };

class FOrdering{
    public:
        vector<NodeID> inv; //fetch the original ID of a given order
        vector<NodeID> rank; // Fetch the ranking of a given vertex id.
		vector< queryPair > query_pair_freq_rank; //2-dimonsion arrays to store the query pair times
		vector< pair<NodeID,int> > query_point_freq_rank;//store the query frequency of each point
		vector< vector<int> > query_pair_freq;//use to query
		vector< int> query_point_freq;//use to query

        FOrdering(){
            inv.clear();
            rank.clear();
			query_pair_freq.clear();
			query_point_freq.clear();
			query_pair_freq_rank.clear();
			query_point_freq_rank.clear();
        }
        ~FOrdering(){
            vector<NodeID>().swap(inv); 
            vector<NodeID>().swap(rank);
            vector< queryPair >().swap(query_pair_freq_rank);
            vector< pair<NodeID,int> >().swap(query_point_freq_rank);
            vector< vector<int> >().swap(query_pair_freq);
            vector< int>().swap(query_point_freq);
            inv.clear();
            rank.clear();
            query_pair_freq.clear();
            query_pair_freq_rank.clear();
            query_point_freq.clear();
            query_point_freq_rank.clear();
        }

        void save_rank(const char* order_file) {
            ofstream ofs(order_file);
            for (int i = 0; i < numOfVertices; ++i) {
                ofs << inv[i] << endl;
            }
            ofs.close();
	    }
        /*
         *@description: load query frequency from history data
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        void load_query_frequency(char* load_filename){ 
			 query_pair_freq.resize(numOfVertices); //initial
			 for(int i=0;i<numOfVertices;++i) query_pair_freq[i].resize(numOfVertices);
			 query_point_freq.resize(numOfVertices); //initial
			 ifstream in(load_filename);//input HFPoint file to ifstream
			 if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
			 NodeID s, t; int cnt=0,q=0;
			 char line[24];
			 //read each line representing HFpoint to vector 
			 while (in.getline(line,sizeof(line)))
			 {
				 stringstream hp(line);
				 hp>>s>>t>>cnt;
				query_pair_freq[s][t]=query_pair_freq[t][s]=q;
				query_point_freq[s]+=q;
				query_point_freq[t]+=q;
				cnt+=q;
			 }
			 in.close();
			 //store
			query_point_freq_rank.resize(numOfVertices);//ignore the i-i pair
			for(NodeID i=0;i<numOfVertices;++i) query_point_freq_rank[i]=make_pair(i,query_point_freq[i]);
			for(NodeID i=0;i<numOfVertices-1;++i){
				for(NodeID j=i+1;j<numOfVertices;++j){
					queryPair qp(i,j,query_pair_freq[i][j]);
					query_pair_freq_rank.push_back(qp);
				}
			}
            for(int i=0;i<numOfVertices;++i){
                inv[i]=query_point_freq_rank[i].first;
            }
			 //sorted by desending
			sort(query_point_freq_rank.begin(),query_point_freq_rank.end(), cmp);
			sort(query_pair_freq_rank.begin(), query_pair_freq_rank.end() );
			std::cout<<"Query_pair size: "<<query_pair_freq_rank.size()<<endl;
			std::cout<<"Total query times: "<<cnt<<endl;
		 }


};

class Hierarchy_fordering : public FOrdering{
    public:
        HFLabel labels; //undirected weighted graph
        int numOfHFPoint=0; //num of high frequency points

        //*****************constructions****************
        /*
         *@description:undirected weighted graph
         *@params:model:0-degree,1-betwenness
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        Hierarchy_fordering(WGraph& wgraph, char* query_freq_filename,int model){ //undirected weighted graph
            undirected_weighted_hflabel(wgraph,query_freq_filename,model);
        }

        /*
         *@description: undirected weighted graph for full path query
         *@author: wanjingyi
         *@date: 2020-11-20
        */
         Hierarchy_fordering(WGraph& wgraph, char* query_freq_filename,int model,bool p_flags){}
        

        //some util functions to be called by others
        void save_anaylysis_size(const char* write_filename){
			//write size analysis to file
			long long total_sum_size=0,hf_sum_size=0;
			double total_ave_size=0,hf_ave_size=0;
            string write_filename_prefix(write_filename);
			string asize_filename=write_filename_prefix.append("_analysis.size");
			ofstream ofs(asize_filename.c_str());
			if(!ofs.is_open()) {cerr<<"Cannot open "<<asize_filename<<endl;}
			for (NodeID v = 0; v < numOfVertices; ++v) 
			{
				NodeID isize = labels.index_[v].size()-1;
				total_sum_size+=isize;
				if(rank[v]<numOfHFPoint) hf_sum_size+=isize;
			}
            total_ave_size= (double) total_sum_size/(double) numOfVertices;
			hf_ave_size= (double) hf_sum_size/(double) numOfHFPoint;
			ofs<<"numOfVertices = "<<numOfVertices<<" total_sum_size = "<<total_sum_size<<" total_ave_size = "<<total_ave_size<<endl;
			ofs<<"numOfHFpoint = "<<numOfHFPoint<<" hf_sum_size = "<<hf_sum_size<<" hf_ave_size = "<<hf_ave_size<<endl;
			ofs.close();	

        }

    protected:
        /*
         *@description: calculate the first level(high frequency) points num(threshold)
         *@return: num-k
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        int calc_hl_k(){ 
            int k=0; double hfRatio=0.1;
            k=ceil(numOfVertices*hfRatio);
            std::cout<<"hf point num:"<<k<<endl;
            return k;
        }

        /*
         *@description: ordering low frequency point by degree
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        void lfpoint_Dordering(WGraph& wgraph){
            vector<bool> tmp(numOfVertices,0); 
            vector<pair<float, NodeID> > deg(numOfVertices);
            srand(100);
            for (size_t v = 0; v < numOfVertices; ++v) {
                if (DIRECTED_FLAG == true)
                    deg[v] = make_pair((wgraph.vertices[v + 1] - wgraph.vertices[v]) * (wgraph.r_vertices[v + 1] - wgraph.r_vertices[v]) + float(rand()) / RAND_MAX, v);
                else
                    deg[v] = make_pair((wgraph.vertices[v + 1] - wgraph.vertices[v]) + float(rand()) / RAND_MAX, v);
            }
            sort(deg.rbegin(), deg.rend());
            for (size_t v = 0; v < numOfHFPoint; ++v){
                tmp[inv[v]]=true;
                rank[inv[v]]=v;
            }
            size_t i=numOfHFPoint;
            for (size_t v = 0; v < numOfVertices; ++v) {
                if(!tmp[deg[v].second]) {
                    inv[i] = deg[v].second;
                    rank[deg[v].second]=i++;
                }
            }
            if(i!=numOfVertices) cerr<<"i!=numOfVertices"<<endl;
            //Relabel(wgraph);
        }

        /*
         *@description: ordering low frequency point by betwenness
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        void lfpoint_Bordering(WGraph& wgraph){
            
        }

        /*
         *@description: construct the hf_hierarchy labels
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        void hflabel_construction(WGraph& wgraph){
            Given_Ordering given_order(wgraph, inv, rank);
            PL_W pl_w(wgraph, given_order);
            labels.index_.assign(pl_w.labels.index_.begin(),pl_w.labels.index_.end());
        }

        /*
         *@description: undirected_weighted frequency hierarchy label construction based on degree
         *@author: wanjingyi
         *@date: 2020-11-20
        */
        void undirected_weighted_hflabel(WGraph& wgraph,char* query_freq_filename,int order_model){
            std::cout<<"*********************load_query_frequency begins*************************"<<endl;
            load_query_frequency(query_freq_filename);
            std::cout<<"*********************load_query_frequency finished*************************"<<endl;

            std::cout<<"*********************calc_hl_k begins*************************"<<endl;
            numOfHFPoint=calc_hl_k();
            std::cout<<"*********************calc_hl_k finished*************************"<<endl;

            std::cout<<"*********************LFPoint ordering begins*************************"<<endl;
            if(order_model==0) lfpoint_Dordering(wgraph);
            std::cout<<"*********************LFPoint ordering finished*************************"<<endl;

            std::cout<<"*********************label construction begins*************************"<<endl;
            hflabel_construction(wgraph);
            std::cout<<"*********************label construction finished*************************"<<endl;            


        }

};

/*
 *@description: the basis class of other synthesis ordering ways
 *@author: wanjingyi
 *@date: 2020-12-07
*/
class SOrdering{
    public:
        vector<NodeID> inv; // Fetch the original vertex id by a given ranking.
	    vector<NodeID> rank; // Fetch the ranking of a given vertex id.

        void Relabel(Graph& graph) { //unweighted graph
            for (NodeID v = 0; v < numOfVertices; ++v) rank[inv[v]] = v;
            // Array Representation
            vector<EdgeID> new_vertices(numOfVertices + 1);
            vector<NodeID> new_edges;
            new_edges.reserve(graph.edges.size());
            for (NodeID ranking = 0; ranking < numOfVertices; ++ranking) {
                NodeID originalVertex = inv[ranking];
                for (EdgeID eid = graph.vertices[originalVertex]; eid < graph.vertices[originalVertex + 1]; ++eid)
                    new_edges.push_back(rank[graph.edges[eid]]);
                new_vertices[ranking + 1] = new_edges.size();
            }
            graph.vertices.swap(new_vertices);
            graph.edges.swap(new_edges);

            if (DIRECTED_FLAG == true) {
                vector<EdgeID> r_new_vertices(numOfVertices + 1);
                vector<NodeID> r_new_edges;
                r_new_edges.reserve(graph.r_edges.size());
                for (NodeID ranking = 0; ranking < numOfVertices; ++ranking) {
                    NodeID originalVertex = inv[ranking];
                    for (EdgeID eid = graph.r_vertices[originalVertex]; eid < graph.r_vertices[originalVertex + 1]; ++eid)
                        r_new_edges.push_back(rank[graph.r_edges[eid]]);
                    r_new_vertices[ranking + 1] = r_new_edges.size();
                }
                graph.r_vertices.swap(r_new_vertices);
                graph.r_edges.swap(r_new_edges);
            }
        }

        void Relabel(WGraph& wgraph) {
		    for (NodeID v = 0; v < numOfVertices; ++v) rank[inv[v]] = v;
        	// Array Representation
            vector<EdgeID> new_vertices(numOfVertices + 1);
            vector<NodeEdgeWeightPair> new_edges;
            new_edges.reserve(wgraph.edges.size());
            for (NodeID ranking = 0; ranking < numOfVertices; ++ranking) {
                NodeID originalVertex = inv[ranking];
                for (EdgeID eid = wgraph.vertices[originalVertex]; eid < wgraph.vertices[originalVertex + 1]; ++eid)
                    new_edges.push_back(make_pair(rank[wgraph.edges[eid].first],wgraph.edges[eid].second));
                new_vertices[ranking + 1] = new_edges.size();
            }
            wgraph.vertices.swap(new_vertices);
            wgraph.edges.swap(new_edges);

            if (DIRECTED_FLAG == true) {
                vector<EdgeID> r_new_vertices(numOfVertices + 1);
                vector<NodeEdgeWeightPair> r_new_edges;
                r_new_edges.reserve(wgraph.r_edges.size());
                for (NodeID ranking = 0; ranking < numOfVertices; ++ranking) {
                    NodeID originalVertex = inv[ranking];
                    for (EdgeID eid = wgraph.r_vertices[originalVertex]; eid < wgraph.r_vertices[originalVertex + 1]; ++eid)
                        r_new_edges.push_back(make_pair(rank[wgraph.r_edges[eid].first],wgraph.r_edges[eid].second));
                    r_new_vertices[ranking + 1] = r_new_edges.size();
                }
                wgraph.r_vertices.swap(r_new_vertices);
                wgraph.r_edges.swap(r_new_edges);
            }
        } 

        void ReswapLabel(Graph& graph) {
            vector<vector<NodeID> > new_adj(numOfVertices);
            for (NodeID v = 0; v < numOfVertices; ++v) {
                for (NodeID i = 0; i < graph.adj[v].size(); ++i) {
                    new_adj[v].push_back(inv[graph.adj[rank[v]][i]]);
                }
            }
            graph.adj.swap(new_adj);
        }

        void save_rank(const char* order_file) {
            ofstream ofs(order_file);
            for (int i = 0; i < numOfVertices; ++i) {
                ofs << inv[i] << endl;
            }
            ofs.close();
        }

        SOrdering(){
            inv.clear();
		    rank.clear();
        }

        ~SOrdering(){
            inv.clear();
		    rank.clear();
        }
};

/*
 *@description: ordering class by heuristic selection iteratively
 *@author: wanjingyi
 *@date: 2020-12-07
*/
class Synthesis_Ordering : public SOrdering{
    public:
        
        Processing::calcCoefficient<double> _calcCoef;
        HFLabel labels;
        NodeID last_available;
        NodeID numOfHFpoint;
        vector<NodeID> HFPoint_inv;//point ordered by frequency
        vector<NodeID> HFPoint_rank;//fetch the freq order by nodeId
        vector<bool> HFPointIndex;//whether the point is high freq point 
        vector<unsigned int> HFPoint_freq;//get the query times by NodeId
        vector<NodeID>  Degree;//get the node degree by id index 
        vector<NodeID> Degree_inv;//get the degree by id index
        vector<NodeID> Degree_rank;//fetch the degree order by id index
        vector<pair<EdgeWeight,NodeID> > Depth;//store the distance
       // vector<NodeID> Depth_inv;//get the current distance from choosen node by id index
        vector<NodeID> Depth_rank;//fetch the distance order by id index
        vector<pair<unsigned int,NodeID> > Coverage;//store the size of descendant difference of each node
        //vector<NodeID> Coverage_inv;//get the current size of descendant difference by id index
        vector<NodeID> Coverage_rank;//fetch the descendantDifference order by id index
        //typedef 
        typedef double orderWeightType;//weight used to weigh the importance of the node
        typedef priority_queue<pair<NodeID,orderWeightType>,vector<pair<NodeID,orderWeightType> >,cmp_queue> max_queue;//big top heap
        typedef	vector<NodeID> tree;//store the parent nodes
        long long total_sum_size=0,hf_sum_size=0;//total size variables
	    double total_ave_size=0,hf_ave_size=0;//average size variables
        ofstream debug_out;//output debug information

        //*****************construciton fuctions********************
        Synthesis_Ordering(){
            numOfHFpoint=0;
            HFPoint_inv.clear();
            HFPoint_rank.clear();
            HFPointIndex.clear();
            HFPoint_freq.clear();
            Degree.clear();
            Degree_inv.clear();
            Degree_rank.clear();
            Depth.clear();
            //Depth_inv.clear();
            Depth_rank.clear();
            Coverage.clear();
            //Coverage_inv.clear();
            Coverage_rank.clear();
            inv.clear();
            rank.clear();
        }
        ~Synthesis_Ordering(){
            numOfHFpoint=0;
            HFPoint_inv.clear();
            HFPoint_rank.clear();
            HFPointIndex.clear();
            HFPoint_freq.clear();
            Degree.clear();
            Degree_inv.clear();
            Degree_rank.clear();
            Depth.clear();
            //Depth_inv.clear();
            Depth_rank.clear();
            Coverage.clear();
            //Coverage_inv.clear();
            Coverage_rank.clear();
            inv.clear();
            rank.clear();
        }
        Synthesis_Ordering(CHFGraph& chfgraph,Processing::calcCoefficient<double> calcCoef,char* load_filename,int hfRate,bool isDrected) //weighted and directed graph
        {
        }

        Synthesis_Ordering(CHFGraph& chfgraph,Processing::calcCoefficient<double> calcCoef,char* hfpointFIleName,int hfRate)//weighted and undirected graph
        {
            //initialize all params vector
            initIterms();
            //DEBUG
            if(DEBUG_FLAG){
                debug_out.open(debugFileName,ios::out);
                if(!debug_out.is_open()) {cerr<<"Cannot open "<<debugFileName<<endl;}
            }
            _calcCoef=calcCoef;//coeffient struct read from command line
            double _labeling_time = GetCurrentTimeSec();
            if(_calcCoef.is_freq_mult) load_HFpoint(hfpointFIleName,hfRate);//frequency
            if(_calcCoef.is_deg_mult) getDegree(chfgraph);//degree
            undirected_weighted_sigpoint_selection(chfgraph);
             _labeling_time = GetCurrentTimeSec() - _labeling_time;
            cout << "Indexing time:" << _labeling_time *1e6 <<  " microseconds" << endl;
        }

        //****************************public functions***********************
        void save_analysisSize_to_file(const char* write_filename){
            string write_filename_prefix(write_filename);//all writefile name common prefix
            //analysis size
            string asize_filename=write_filename_prefix.append(".asize");
            total_ave_size= (double) total_sum_size/(double) numOfVertices;
			hf_ave_size= (double) hf_sum_size/(double) numOfHFpoint;
            cout<<"numOfVertices = "<<numOfVertices<<" total_sum_size = "<<total_sum_size<<" total_ave_size = "<<total_ave_size<<endl;
			cout<<"numOfHFpoint = "<<numOfHFpoint<<" hf_sum_size = "<<hf_sum_size<<" hf_ave_size = "<<hf_ave_size<<endl;
            ofstream ofs(asize_filename.c_str());
            if(!ofs.is_open()) {cerr<<"Cannot open "<<asize_filename<<endl;}
			ofs<<"numOfVertices = "<<numOfVertices<<" total_sum_size = "<<total_sum_size<<" total_ave_size = "<<total_ave_size<<endl;
			ofs<<"numOfHFpoint = "<<numOfHFpoint<<" hf_sum_size = "<<hf_sum_size<<" hf_ave_size = "<<hf_ave_size<<endl;
			ofs.close();	
            //debug information
            // string write_filename_prefix1(write_filename);
            // string debug_filename=write_filename_prefix1.append(".debug");
            // ofstream ofs1(debug_filename.c_str());
            // if(!ofs1.is_open()) {cerr<<"Cannot open "<<debug_filename<<endl;}
        }
        

    protected:
        //******************class tool functions*************
        /*
         *@description: used to init all order params 0
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        void initIterms(){
            Degree.resize(numOfVertices,0);
            Degree_inv.resize(numOfVertices);
            Degree_rank.resize(numOfVertices,numOfVertices);
            HFPoint_freq.resize(numOfVertices,0);
            HFPoint_inv.resize(numOfVertices);
            HFPoint_rank.resize(numOfVertices,numOfVertices);
            Depth.resize(numOfVertices);
            //Depth_inv.resize(numOfVertices);
            Depth_rank.resize(numOfVertices,numOfVertices);
            Coverage.resize(numOfVertices);
            //Coverage_inv.resize(numOfVertices);
            Coverage_rank.resize(numOfVertices,numOfVertices);
            inv.reserve(numOfVertices);
            rank.resize(numOfVertices,numOfVertices);
        }
        
        /*
         *@description: used to init the start order status
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        void updatePQueue(benchmark::heap<2, orderWeightType, NodeID> & wqueue,vector<bool> usd){
            std::cout << "Initialize elimination weights..." << endl;
            for(NodeID v=0;v<numOfVertices;++v){
                if(usd[v]) continue;
                orderWeightType orderWeight=calculateOrderWeight(v);
                wqueue.update(v,orderWeight);
            }
        }

        /*
         *@description: used to iteratively select the maxmum to dijkstra search and 
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        void undirected_weighted_sigpoint_selection(CHFGraph& chfgraph){
            double _labeling_time = GetCurrentTimeSec();
            //*****************variables*******************
            benchmark::heap<2, EdgeWeight, NodeID> pqueue(numOfVertices);//priority_queue used to get the minimum distance node
            benchmark::heap<2, orderWeightType, NodeID> wqueue(numOfVertices);//priority_queue used to get the maxmal weight node
            NodeID choosen;orderWeightType choosen_w;
            int choose_cnt=0;
            tree parent_tree;
            vector<bool> usd(numOfVertices,false);//flag whether has been as source
            vector<NodeID> parent_tree(numOfVertices,numOfVertices);//numOfVertices means root node with no parent
            vector<NodeID> root_hop(numOfVertices,0);//store the hop from root
            vector<NodeID> coverage(numOfVertices,0);//store the shortest distances coverage
            vector<EdgeWeight> depth(numOfVertices,INF_WEIGHT);//store the depth of bfs SP-Tree  of each node
            vector<NodeID> descendants;//store the vertices by visited order
            descendants.reserve(numOfVertices);
            //each dij visited node(queue top node)
            vector<bool> vis(numOfVertices,false); 
            queue<NodeID> visited_que;//FIFO
            vector<EdgeWeight> distances(numOfVertices, INF_WEIGHT); //store the distances from source
            benchmark::heap<2, EdgeWeight, NodeID> pqueue(numOfVertices);//priority_queue used to dijkstra
            vector<EdgeWeight> dst_r(numOfVertices+1,INF_WEIGHT);//pruned algorithm: store the source node's label distances to other nodes
            vector<pair<vector<NodeID>, vector<EdgeWeight> > >
			tmp_idx(numOfVertices, make_pair(vector<NodeID>(1, numOfVertices),
			vector<EdgeWeight>(1, INF_WEIGHT)));//store the labels and distances
            vector<NodeID> source_coverage(numOfVertices,0); //the total coverage of each source node
            int source_cnt=0;
            if(inv.size()!=0) std::cout<<"Initially inv.size()!=0 !"<<endl;
            updatePQueue(wqueue,usd);//initialize status order
            while (!wqueue.empty())
            {
                wqueue.extract_min(choosen,choosen_w);//get the top order node
                //update the order
                inv.push_back(choosen);
                rank[choosen]=inv.size()-1;
                if(usd[choosen]) std::cout<<"Node has been usd......"<<endl;
                source_coverage[choosen]=source_dij_bfs(choosen,parent_tree,descendants,root_hop,pqueue,distances,vis,dst_r,usd,tmp_idx,chfgraph,depth);
                calcCover(descendants,parent_tree,coverage,root_hop);
                updateWeightParams(descendants,depth,coverage);
                if(DEBUG_FLAG){//DEBUG
                    debug_out<<"******************* "<<source_cnt<<" "<<choosen<<" *******************"<<endl;
                    debug_out<<"coverage :";
                    for(size_t i=0;i<numOfVertices;++i) debug_out<<" ("<<Coverage[i].second<<","<<Coverage[i].first<<")"<<endl;
                    debug_out<<endl;
                    debug_out<<"depth : ";
                    for(size_t i=0;i<numOfVertices;++i) debug_out<<" ("<<Depth[i].second<<","<<Depth[i].first<<")"<<endl;
                    debug_out<<endl;
                }
                updatePQueue(wqueue,usd);
                clearTmpList(descendants,coverage,parent_tree,root_hop,depth);
                source_cnt++;
                if(source_cnt%1000==0) std::cout<<"cnt : "<<source_cnt<<endl;
            }
            std::cout<<"Dijkstra take times - source_cnt: "<<source_cnt<<endl;
            //*********************store the lables************************
            std::cout<<"labels construction finished, start to store......."<<endl;
            for(NodeID v=0;v<numOfVertices;++v){
                NodeID k=tmp_idx[v].first.size();
                total_sum_size+=k-1;
                if(HFPointIndex[v]) hf_sum_size+=k-1;
                labels.index_[v].spt_v.reserve(k);
                labels.index_[v].spt_d.reserve(k);
                for(NodeID i=0;i<numOfVertices;++i) labels.index_[v].spt_v[i]=rank[tmp_idx[v].first[i]];
                for(NodeID i=0;i<numOfVertices;++i) labels.index_[v].spt_d[i]=rank[tmp_idx[v].second[i]];
                tmp_idx[v].first.clear();
			    tmp_idx[v].second.clear();
                tmp_idx[v].first.shrink_to_fit();
			    tmp_idx[v].second.shrink_to_fit();
            }
            cout << "Indexing time:" << _labeling_time *1e6 <<  " microseconds" << endl;
            if(DEBUG_FLAG){
                debug_out<<"source_coverage :";
                for(size_t i=0;i<source_coverage.size();++i) debug_out<<" "<<source_coverage[i];
                debug_out<<endl;
                debug_out.close();
            }
            std::cout<<"labels store finished..........."<<endl;
        }

        /*
         *@description: used to update all items after each dijkstra turn
         *@author: wanjingyi
         *@date: 2020-12-09
        */
        void updateWeightParams(vector<NodeID>& descendants,vector<EdgeWeight>& depth,vector<NodeID>& coverage){
            NodeID i,j;
            //************coverage rank******************
            for(i=0;i<descendants.size();++i){
                NodeID v=descendants[i];
                Coverage.push_back(make_pair(coverage[v],v));
            }
            sort(Coverage.rbegin(),Coverage.rend());
            for(i=0;i<descendants.size();++i) Coverage_rank[Coverage[i].second]=i;
            //remaing
            //************coverage rank******************
            for(i=0;i<numOfVertices;++i) Depth.push_back(make_pair(depth[i],i));
            sort(Depth.rbegin(),Depth.rend());
            for(j=0;j<numOfVertices;++j) Depth_rank[Depth[j].second]=j;
        }

        /*
         *@description: used to calculate the node coverage shortest path and its depth
         *@author: wanjingyi
         *@date: 2020-12-09
        */
        void calcCover(vector<NodeID>& descendants, tree& parent_tree, vector<NodeID>& coverage, vector<NodeID>& root_hop){
            for(size_t i=descendants.size()-1;i>=0;--i){
                NodeID v=descendants[i];
                coverage[v]++;
                if(parent_tree[v]!=numOfVertices){
                    coverage[parent_tree[v]]+=coverage[v];
                    if(root_hop[parent_tree[v]]<root_hop[v]) root_hop[parent_tree[v]]=root_hop[v];
                }
            }
        }

        void clearTmpList(vector<NodeID>& descendants, vector<NodeID>& coverage, tree& parent_tree, vector<NodeID>& root_hop,vector<EdgeWeight>& depth){
            for(size_t i=0;i<descendants.size();++i){
                NodeID v=descendants[i];
                coverage[v]=0;
                if(parent_tree[v]!=numOfVertices) coverage[parent_tree[v]]=0;
                parent_tree[v]=numOfVertices;
                root_hop[v]=0;
            }
            descendants.clear();
            //*****reset to |numOfVertices|******
            for(NodeID i=0;i<numOfVertices;++i){
                depth[i]=INF_WEIGHT;
                Depth_rank[i]=numOfVertices;
                Coverage_rank[i]=numOfVertices;
            }
            Depth.clear();
            Coverage.clear();
            return;
        }

        /*
         *@description: used to implement dijkstra starting from the choosen node
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        NodeID source_dij_bfs(NodeID source,tree& parent_tree,vector<NodeID>& descendants,vector<NodeID>& root_hop,
       benchmark::heap<2, EdgeWeight, NodeID>& pqueue,vector<EdgeWeight>& distances,vector<bool>& vis,vector<EdgeWeight>& dst_r,
       vector<bool>& usd, vector<pair<vector<NodeID>, vector<EdgeWeight> > >& tmp_idx,CHFGraph& chfgraph,vector<EdgeWeight>& depth){
            NodeID visited_arcs=0;
            const pair<vector<NodeID>,vector<EdgeWeight>>& tmp_idx_r=tmp_idx[source];//source's labels to other vertices
            for(size_t i=0;i<tmp_idx_r.first.size();++i) dst_r[tmp_idx_r.first[i]]=tmp_idx_r.second[i];
            pqueue.update(source,0);
            distances[source]=0;
            vis[source]=true;
            parent_tree[source]=numOfVertices;
            root_hop[source]=0;
            //**********************dijkstra begins*******************
            while(!pqueue.empty()){
                NodeID v;EdgeWeight v_d;
                pqueue.extract_min(v,v_d);
                pair<vector<NodeID>,vector<EdgeWeight> > tmp_idx_v=tmp_idx[v];
                vis[v]=true;
                //visited_que.push(v);
                if(usd[v]) continue;
                for(size_t i=0;i<tmp_idx_v.first.size();++i){//pruned algprithm
                    NodeID w=tmp_idx_v.first[i];
                    EdgeWeight td=tmp_idx_v.second[i]+dst_r[w];
                    if(td<v_d){
                        depth[v]=td;
                        goto pruned;
                    }
                }
                //Traverse
                tmp_idx_v.first.back()=source;
                tmp_idx_v.second.back()=v_d;
                tmp_idx_v.first.push_back(numOfVertices);
                tmp_idx_v.second.push_back(INF_WEIGHT);
                descendants.push_back(v);
                visited_arcs++;

                for(EdgeID eid=chfgraph.vertices[v];eid<chfgraph.vertices[v+1];++eid){
                    NodeID w=chfgraph.edges[eid].first;
                    EdgeWeight w_d=chfgraph.edges[eid].second+v_d;
                    if(!vis[w]){
                        if(distances[w]>w_d){
                            pqueue.update(w,w_d);
                            distances[w]=w_d;
                            parent_tree[w]=v;
                            root_hop[w]=root_hop[v]+1;
                        }
                    }
                }
                pruned:
                    {}
            }
            //**********************dijkstra end*******************
            
            //**********************clear tmp variables****************
            // vis[source]=false;
            // while(!visited_que.empty()){
            //     NodeID vis_v=visited_que.front();
            //     visited_que.pop();
            //     vis[vis_v]=false;
            //     distances[vis_v]=INF_WEIGHT;
            //     pqueue.clear(vis_v);
            // }
            for(NodeID i=0;i<numOfVertices;++i){
                //update distances
                if(depth[i]==INF_WEIGHT||depth[i]<distances[i]) depth[i]=distances[i];
                //clear the tmp list
                vis[i]=false;
                distances[i]=INF_WEIGHT;
                pqueue.clear(i);
            }
            pqueue.clear_n();
            for(size_t i=0;i<tmp_idx_r.first.size();++i) dst_r[tmp_idx_r.first[i]]=INF_WEIGHT; 
            usd[source]=true;//usd mark
            return visited_arcs;
        }

        /*
         *@description: used to claculate order weight of each node
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        orderWeightType calculateOrderWeight(const NodeID node){
            //orderWeightType result=(orderWeightType)_calcCoef.freq_mult*HFPoint_freq[node]+(orderWeightType)Degree[node]*_calcCoef.deg_mult;
            orderWeightType result=(orderWeightType)_calcCoef.freq_mult*HFPoint_rank[node]+(orderWeightType)Degree_rank[node]*_calcCoef.deg_mult+(orderWeightType)_calcCoef.cov_mult*Coverage_rank[node]+(orderWeightType)_calcCoef.dep_mult*Depth_rank[node];
            return result;
        }

        /*
         *@description: used to read high frequency query point and its query times from file
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        void load_HFpoint(char* load_filename,int hfRate=50){ //hfRate/1000
            std::cout<<"***********load hfpoint begins************"<<endl;
            numOfHFpoint = 0;//first line is the number of HFpoints
            numOfHFpoint = static_cast<int> ( (double)numOfVertices*hfRate/(double)1000);
            if(numOfHFpoint<=0) cout<<"error:numOfHFpoint<=0"<<endl;
            cout<<"numOfHFpoint  = "<<numOfHFpoint <<endl;
            ifstream in(load_filename);//input HFPoint file to ifstream
            if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
            HFPoint_inv.resize(numOfVertices);
            HFPoint_rank.resize(numOfVertices);
            HFPointIndex.resize(numOfVertices,0);
            vector<pair<NodeID,NodeID> > queryFreq(numOfVertices);//(freq,id)
            NodeID id,freq;
            //read each line representing HFpoint to vector 
            for(NodeID i=0;i<numOfVertices;++i){
                in>>id>>freq;
                HFPoint_freq[id]=freq;
                queryFreq.push_back(make_pair(freq,id));
            }
            //sort the node by query times
            sort(queryFreq.rbegin(),queryFreq.rend());//descending order
            //initialize query freq information
            for(NodeID i=0;i<numOfVertices;++i){
                NodeID v=queryFreq[i].second;
                HFPoint_rank[v]=i;
                HFPoint_inv[i]=v;
                if(i<numOfHFpoint) HFPointIndex[v]=true;
            }
            std::cout<<"***********load hfpoint finished************"<<endl;
		}

        /*
         *@description: used to get the degree of each node and relative node order
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        void getDegree(CHFGraph& chfgraph){
            chfgraph.calcDeg();
            for(size_t i=0;i<=numOfVertices;++i){
                    NodeID v=chfgraph.deg[i].second;
                    NodeID d=chfgraph.deg[i].first;
                    Degree[v]=d;
                    Degree_rank[v]=i;
                    Degree_inv[i]=v;
            }
        }
};

#endif //FREQUENCY_HIERARCHY_ORDERING