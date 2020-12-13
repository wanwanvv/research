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
#define countType unsigned int
#define MAX_VALUE 0x7FFFFFFF
//DEBUG
#define DEBUG_FLAG 1
char debugFileName[255] = "../dataset/manhatan/SHP/SOrder"; 

//pair compare 
bool cmp1(const pair<NodeID,int> a, const pair<NodeID,int> b) {
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
			sort(query_point_freq_rank.begin(),query_point_freq_rank.end(), cmp1);
			sort(query_pair_freq_rank.begin(), query_pair_freq_rank.end() );
			std::cout<<"Query_pair size: "<<query_pair_freq_rank.size()<<endl;
			std::cout<<"Total query times: "<<cnt<<endl;
		 }


};

class Hierarchy_fordering : public FOrdering{
    public:
        HFLabel labels; //undirected weighted graph
        int numOfHFpoint=0; //num of high frequency points

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

        /*
         *@description: output analysis label size
         *@author: wanjingyi
         *@date: 2020-12-13
        */
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
				if(rank[v]<numOfHFpoint) hf_sum_size+=isize;
			}
            total_ave_size= (double) total_sum_size/(double) numOfVertices;
			hf_ave_size= (double) hf_sum_size/(double) numOfHFpoint;
			ofs<<"numOfVertices = "<<numOfVertices<<" total_sum_size = "<<total_sum_size<<" total_ave_size = "<<total_ave_size<<endl;
			ofs<<"numOfHFpoint = "<<numOfHFpoint<<" hf_sum_size = "<<hf_sum_size<<" hf_ave_size = "<<hf_ave_size<<endl;
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
            for (size_t v = 0; v < numOfHFpoint; ++v){
                tmp[inv[v]]=true;
                rank[inv[v]]=v;
            }
            size_t i=numOfHFpoint;
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
            numOfHFpoint=calc_hl_k();
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
        NodeID numOfHFpoint;//High frequency point num
        vector<bool> HFPointIndex;//whether the point is high freq point 
        HFLabel labels;
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
    typedef double orderWeightType;//weight used to weigh the importance of the node
    typedef	vector<NodeID> tree;//store the parent nodes
    public:    
        Processing::calcCoefficient<double> _calcCoef;
        NodeID last_available;
        // vector<NodeID> HFPoint_inv;//point ordered by frequency
        // vector<NodeID> HFPoint_rank;//fetch the freq order by nodeId
        vector<unsigned int> Freq;//get the query time of each node
        // vector<unsigned int> HFPoint_freq;//get the query times by NodeId
        vector<NodeID>  Degree;//get the node degree by id index 
        // vector<NodeID> Degree_inv;//get the degree by id index
        // vector<NodeID> Degree_rank;//fetch the degree order by id index
        vector<EdgeWeight > Depth;//store the distance
       // vector<NodeID> Depth_inv;//get the current distance from choosen node by id index
        // vector<NodeID> Depth_rank;//fetch the distance order by id index
        // vector<pair<unsigned int,NodeID> > Coverage;//store the size of descendant difference of each node
        vector<unsigned int> Coverage;//store the size of descendant difference of each node
        //vector<NodeID> Coverage_inv;//get the current size of descendant difference by id index
        // vector<NodeID> Coverage_rank;//fetch the descendantDifference order by id index
        //typedef 
        long long total_sum_size=0,hf_sum_size=0;//total size variables
	    double total_ave_size=0,hf_ave_size=0;//average size variables
        ofstream debug_out;//output debug information
        unsigned int  max_freq=0;//max query time used to normalization
        unsigned int max_degree=0;//max degree of all nodes
        unsigned int max_coverage=0;//max coverage of all nodes
        unsigned int max_depth=0;//max depth of all nodes
        unsigned int  min_freq=MAX_VALUE;//min query time used to normalization
        unsigned int min_degree=MAX_VALUE;//min degree of all nodes
        unsigned int min_coverage=MAX_VALUE;//min coverage of all nodes
        unsigned int min_depth=MAX_VALUE;//min depth of all nodes
        unsigned int  interval_freq=0;//max query time -min query time
        unsigned int interval_degree=0;//max_degree -min_degree
        unsigned int interval_coverage=0;//max_coverage-min_coverage
        unsigned int interval_depth=0;//max_depth-min_depth

        //*****************construciton fuctions********************
        Synthesis_Ordering(){
            numOfHFpoint=0;
            Freq.clear();
            Degree.clear();
            Depth.clear();
            Coverage.clear();
            inv.clear();
            rank.clear();
        }
        ~Synthesis_Ordering(){
            numOfHFpoint=0;
            Freq.clear();
            HFPointIndex.clear();
            Degree.clear();
            Depth.clear();
            Coverage.clear();
            inv.clear();
            rank.clear();
        }
        Synthesis_Ordering(CHFGraph& chfgraph,Processing::calcCoefficient<double> calcCoef,char* load_filename,int hfRate,bool isDrected) //weighted and directed graph
        {
        }

        Synthesis_Ordering(CHFGraph& chfgraph,Processing::calcCoefficient<double> calcCoef,char* queryFreqFileName,int hfRate)//weighted and undirected graph
        {
            std::cout<<"Synthesis_Ordering undirected weighted graph...."<<endl;
            //initialize all params vector
            initIterms();
            std::cout<<"Init all items finished!"<<endl;
            //DEBUG
            if(DEBUG_FLAG){
                debug_out.open(debugFileName,ios::out);
                if(!debug_out.is_open()) {cerr<<"Cannot open "<<debugFileName<<endl;}
            }
            _calcCoef=calcCoef;//coeffient struct read from command line
            //std::cout<<"_calcCoef mult : "<<_calcCoef.deg_mult<<" "<<_calcCoef.freq_mult<<" "<<_calcCoef.cov_mult<<" "<<_calcCoef.dep_mult<<" "<<endl;
            double _labeling_time = GetCurrentTimeSec();
            if(_calcCoef.is_freq_mult) load_HFpoint(queryFreqFileName,hfRate);//frequency
            if(_calcCoef.is_deg_mult) getDegree(chfgraph);//degree
            undirected_weighted_sigpoint_selection(chfgraph);
             _labeling_time = GetCurrentTimeSec() - _labeling_time;
            cout << "Indexing time:" << _labeling_time *1e6 <<  " microseconds" << endl;
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
            Freq.resize(numOfVertices,0);
            Depth.resize(numOfVertices,INF_WEIGHT);
            Coverage.resize(numOfVertices,0);
            inv.reserve(numOfVertices);
            rank.resize(numOfVertices,numOfVertices);
        }
        
        /*
         *@description: used to init the start order status
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        // void updatePQueue(benchmark::heap<2, orderWeightType, NodeID> & wqueue,vector<bool> usd){
        //     for(NodeID v=0;v<numOfVertices;++v){
        //         if(usd[v]) continue;
        //         orderWeightType orderWeight=calculateOrderWeight(v);
        //         wqueue.update(v,orderWeight);
        //     }
        // }

        /*
         *@description: used to iteratively select the maxmum to dijkstra search and 
         *@author: wanjingyi
         *@date: 2020-12-08
        */
        void undirected_weighted_sigpoint_selection(CHFGraph& chfgraph){
            double _labeling_time = GetCurrentTimeSec();
            //*****************variables*******************
            benchmark::heap<2, EdgeWeight, NodeID> pqueue(numOfVertices);//priority_queue used to get the minimum distance node
            //benchmark::heap<2, orderWeightType, NodeID> wqueue(numOfVertices);//priority_queue used to get the maxmal weight node
            NodeID choosen;orderWeightType choosen_w;
            int choose_cnt=0;
            tree parent_tree(numOfVertices,numOfVertices);//numOfVertices means root node with no parent
            vector<bool> usd(numOfVertices,false);//flag whether has been as source
            vector<NodeID> root_hop(numOfVertices,0);//store the hop from root
            vector<NodeID> coverage(numOfVertices,0);//store the shortest distances coverage
            vector<EdgeWeight> depth(numOfVertices,INF_WEIGHT);//store the depth of bfs SP-Tree  of each node
            vector<NodeID> descendants;//store the vertices by visited order
            descendants.reserve(numOfVertices);
            //each dij visited node(queue top node)
            vector<bool> vis(numOfVertices,false); 
            queue<NodeID> visited_que;//FIFO
            vector<EdgeWeight> distances(numOfVertices, INF_WEIGHT); //store the distances from source
            vector<EdgeWeight> dst_r(numOfVertices+1,INF_WEIGHT);//pruned algorithm: store the source node's label distances to other nodes
            vector<pair<vector<NodeID>, vector<EdgeWeight> > >
			tmp_idx(numOfVertices, make_pair(vector<NodeID>(1, numOfVertices),
			vector<EdgeWeight>(1, INF_WEIGHT)));//store the labels and distances
            vector<NodeID> source_coverage(numOfVertices,0); //the total coverage of each source node
            int source_cnt=0;
            if(inv.size()!=0) std::cout<<"Initially inv.size()!=0 !"<<endl;
            
            while (source_cnt<numOfVertices)
            {
                std::cout<<"******************"<<source_cnt<<"*********************"<<endl;
                getNextConstructNode(choosen,choosen_w,usd);
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
                    for(NodeID i=0;i<numOfVertices;++i) debug_out<<" ("<<i<<","<<Coverage[i]<<")"<<endl;
                    debug_out<<endl;
                    debug_out<<"depth : ";
                    for(NodeID i=0;i<numOfVertices;++i) debug_out<<" ("<<i<<","<<Depth[i]<<")"<<endl;
                    debug_out<<endl;
                }
                clearTmpList(descendants,coverage,parent_tree,root_hop,depth);
                source_cnt++;
                if(source_cnt%1000==0) std::cout<<"cnt : "<<source_cnt<<endl;
            }
            std::cout<<"Dijkstra take times - source_cnt: "<<source_cnt<<endl;
            cout << "Indexing time:" << _labeling_time *1e6 <<  " microseconds" << endl;
            double ave_labeling_time=_labeling_time/(double) numOfVertices;
            cout<<"average indexing time:"<<ave_labeling_time*1e6 <<  " microseconds" << endl;
            //*********************store the lables************************
            std::cout<<"labels construction finished, start to store......."<<endl;
            labels.index_.resize(numOfVertices);
            for(NodeID v=0;v<numOfVertices;++v){
                NodeID k=tmp_idx[v].first.size();
                labels.index_[v].spt_v.reserve(k);
                labels.index_[v].spt_d.reserve(k);
                for(NodeID i=0;i<numOfVertices;++i) labels.index_[v].spt_v[i]=rank[tmp_idx[v].first[i]];
                for(NodeID i=0;i<numOfVertices;++i) labels.index_[v].spt_d[i]=rank[tmp_idx[v].second[i]];
                tmp_idx[v].first.clear();
			    tmp_idx[v].second.clear();
                tmp_idx[v].first.shrink_to_fit();
			    tmp_idx[v].second.shrink_to_fit();
            }
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
            NodeID i;
            max_coverage=0;min_coverage=MAX_VALUE;interval_coverage=0;
            max_depth=0;min_depth=MAX_VALUE;interval_depth=0;
            for(i=0;i<descendants.size();++i){
                NodeID v=descendants[i];
                Coverage[v]=coverage[v];
            }
            for(i=0;i<numOfVertices;++i){
                if(depth[i]!=INF_WEIGHT&&depth[i]>max_depth) max_depth=depth[i];
                if(depth[i]!=INF_WEIGHT&&depth[i]<min_depth) min_depth=depth[i];
                if(Coverage[i]>max_coverage) max_coverage=Coverage[i];
                if(Coverage[i]<min_coverage) min_coverage=Coverage[i];
            }
            interval_coverage=max_coverage-min_coverage;
            if(max_depth==MAX_VALUE) return;
            for(i=0;i<numOfVertices;++i){
                if(Depth[i]==INF_WEIGHT) Depth[i]=max_depth;
            }
            interval_depth=max_depth-min_depth;
            return;
        }

        /*
         *@description: used to calculate the node coverage shortest path and its depth
         *@author: wanjingyi
         *@date: 2020-12-09
        */
        void calcCover(vector<NodeID>& descendants, tree& parent_tree, vector<NodeID>& coverage, vector<NodeID>& root_hop){
            for(NodeID i=descendants.size()-1;i>=0;--i){
                NodeID v=descendants[i];
                coverage[v]++;
                if(parent_tree[v]!=numOfVertices){
                    coverage[parent_tree[v]]+=coverage[v];
                    if(root_hop[parent_tree[v]]<root_hop[v]) root_hop[parent_tree[v]]=root_hop[v];
                }
            }
        }
    
        /*
         *@description: used to clear all tmp variables
         *@author: wanjingyi
         *@date: 2020-12-11
        */
        void clearTmpList(vector<NodeID>& descendants, vector<NodeID>& coverage, tree& parent_tree, vector<NodeID>& root_hop,vector<EdgeWeight>& depth){
            for(size_t i=0;i<descendants.size();++i){
                NodeID v=descendants[i];
                coverage[v]=0;
                if(parent_tree[v]!=numOfVertices) coverage[parent_tree[v]]=0;
                parent_tree[v]=numOfVertices;
                root_hop[v]=0;
            }
            descendants.clear();
            //*****reset params******
            for(NodeID i=0;i<numOfVertices;++i){
                depth[i]=INF_WEIGHT;
                Depth[i]=INF_WEIGHT;
                Coverage[i]=0;
            }
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
            orderWeightType result=0;
            if(_calcCoef.freq_mult&&interval_freq!=0) result+=(orderWeightType)_calcCoef.freq_mult*((orderWeightType)(Freq[node]-min_freq)/(orderWeightType)interval_freq);
            if(_calcCoef.deg_mult&&interval_degree!=0)  result+=(orderWeightType)_calcCoef.deg_mult*((orderWeightType)(Degree[node]-min_degree)/(orderWeightType)interval_degree);
            if(_calcCoef.cov_mult&&interval_coverage!=0) result+=(orderWeightType)_calcCoef.cov_mult*((orderWeightType)(Coverage[node]-min_coverage)/(orderWeightType)interval_coverage);
            if(_calcCoef.dep_mult&&interval_depth!=0) result+=(orderWeightType)_calcCoef.dep_mult*((orderWeightType)(Depth[node]-min_depth)/(orderWeightType)interval_depth);
            return result;
        }

        /*
         *@description: used to get the fisrt order node to take dijkstra
         *@author: wanjingyi
         *@date: 2020-12-11
        */
        void getNextConstructNode(NodeID& choose,orderWeightType& choose_w,vector<bool> usd){
            NodeID max_v=numOfVertices;
            orderWeightType max_w=0;
            for(NodeID v=0;v<numOfVertices;++v){
                if(usd[v]) continue;
                orderWeightType result=calculateOrderWeight(v);
                //std::cout<<result<<" "; //to be deleted
                if(result>max_w||max_v==numOfVertices){
                    std::cout<<"max_v ="<<v<<" max_w = "<<max_w<<endl;//to be deleted
                    max_v=v;
                    max_w=result;
                }
            }
            if(max_v==numOfVertices) std::cout<<"error: max_v==numOfVertices!"<<endl;//to be deleted
            choose=max_v;
            choose_w=max_w;
            return;
            //std::cout<<endl;//to be deleted
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
            HFPointIndex.resize(numOfVertices,0);
            vector<pair<NodeID,NodeID> > queryFreq;//(freq,id)
            NodeID id,freq;
            //read each line representing HFpoint to vector 
            for(NodeID i=0;i<numOfVertices;++i){
                in>>id>>freq;
                Freq[id]=freq;
                queryFreq.push_back(make_pair(freq,id));
            }
            //sort the node by query times
            //sort(queryFreq.rbegin(),queryFreq.rend());//descending order
            //initialize query freq information
            for(NodeID i=0;i<numOfVertices;++i){
                NodeID v=queryFreq[i].second;
                if(i<numOfHFpoint) HFPointIndex[v]=true;
            }
            max_freq=queryFreq[0].first;
            min_freq=queryFreq[queryFreq.size()-1].first;
            interval_freq=max_freq-min_freq;
            // if(interval_freq==0){
            //     std::cout<<"all frequencies equals!"<<endl;
            //     _calcCoef.freq_mult=false;
            // }
            std::cout<<"max_query_time = "<<max_freq<<endl;
            std::cout<<"min_query_time = "<<min_freq<<endl;
            std::cout<<"interval_freq = "<<interval_freq<<endl;
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
                    // //compute max_degree
                    // if(d>max_degree) max_degree=d;
                    // //compute the min degree
                    // if(d<min_degree) min_degree=d;
            }
            max_degree=chfgraph.deg[0].first;
            min_degree=chfgraph.deg[chfgraph.deg.size()-1].first;
            interval_degree=max_degree-min_degree;
            std::cout<<"interval_degree = "<<interval_degree<<endl;
        }
};

/*
 *@description: class used to order nodes with linear itemrs*coefficeient = weight sum
 *@author: wanjingyi
 *@date: 2020-12-13
*/
template<typename weightType>
class Linear_Ordering :public SOrdering{
    public:
        Processing::calcCoefficient<double> _calcCoef;
        HFLabel labels;
        vector<weightType> _freq;//get the query time of each node by index
        vector<weightType> _betwenness;//get the betwenness by node index
        vector<weightType> _coverage;//get the coverage by node index
        vector<weightType> _depth;//get the depth by node index
        vector<weightType> _degree;//get the degree by node index
        vector<NodeID> _freq_inv;//fetch the index by rank
        vector<NodeID> _freq_rank;//fetch the rank  by index
        vector<NodeID> _betwenness_inv;//fetch the index by rank
        vector<NodeID> _betwenness_rank;//fetch the rank  by index
        vector<NodeID> _coverage_inv;//fetch the index by rank
        vector<NodeID> _coverage_rank;//fetch the rank  by index
        vector<NodeID> _depth_inv;//fetch the index by rank
        vector<NodeID> _depth_rank;//fetch the rank  by index
        vector<NodeID> _degree_inv;//fetch the index by rank
        vector<NodeID> _degree_rank;//fetch the rank  by index
        long long total_sum_size=0,hf_sum_size=0;//total size variables
	    double total_ave_size=0,hf_ave_size=0;//average size variables
        ofstream debug_out;//output debug information
        unsigned int  max_freq=0;//max query time used to normalization
        unsigned int max_degree=0;//max degree of all nodes
        unsigned int max_coverage=0;//max coverage of all nodes
        unsigned int max_depth=0;//max depth of all nodes
        unsigned int  min_freq=MAX_VALUE;//min query time used to normalization
        unsigned int min_degree=MAX_VALUE;//min degree of all nodes
        unsigned int min_coverage=MAX_VALUE;//min coverage of all nodes
        unsigned int min_depth=MAX_VALUE;//min depth of all nodes
        unsigned int  interval_freq=0;//max query time -min query time
        unsigned int interval_degree=0;//max_degree -min_degree
        unsigned int interval_coverage=0;//max_coverage-min_coverage
        unsigned int interval_depth=0;//max_depth-min_depth
        vector<bool> isConstruct;//indicates whether the node has been constructed
        //***************************construction functions********************

        Linear_Ordering(){

        }
        ~Linear_Ordering(){

        }

        /*
        *@description: construction used for undirected and weighted graph
        *@author: wanjingyi
        *@date: 2020-12-13
        */
        Linear_Ordering(WGraph& wgraph,Processing::calcCoefficient<double> calcCoef,char* queryFreqFileName,char* betwennessFileName,char* coverageFileName,int hfRate){
            std::cout<<"Linear_Ordering undirected weighted graph...."<<endl;
            initIterms();//init all values
            _calcCoef=calcCoef;//get the coefficient from command line
            if(_calcCoef.is_freq_mult) load_HFpoint(queryFreqFileName,hfRate);//frequency
            if(_calcCoef.is_deg_mult) getDegree(wgraph);//degree
            if(_calcCoef.is_bet_mult) getBetwennessOrderFromFile(betwennessFileName);//betwenness
            if(_calcCoef.is_cov_mult) getCoverageOrderFromFile(coverageFileName);//coverage
            calcWeightByOrder();//cumpute the order

        }

    protected:
        /*
         *@description: calculate the node weight by rank
         *@author: wanjingyi
         *@date: 2020-12-13
        */
        void calcWeightByOrder(){
            std::cout<<"*****************************calcWeightByOrder begins!**************************"<<endl;
            vector<pair<weightType,NodeID> > orderWeight;
            orderWeight.reserve(numOfVertices);
            for(NodeID v=0;v<numOfVertices;++v){
                weightType result=0;
                result=(weightType)_freq_rank[v]*_calcCoef.freq_mult+(weightType)_degree_rank[v]*_calcCoef.deg_mult+
                (weightType)_betwenness_rank[v]*_calcCoef.bet_mult+(weightType)_coverage_rank[v]*_calcCoef.cov_mult+
                (weightType)_depth_rank[v]*_calcCoef.dep_mult;
                orderWeight.push_back(make_pair(result,v) );
            }
            //sort
            sort(orderWeight.begin(),orderWeight.end());
            for(NodeID i=0;i<numOfVertices;++i){
                NodeID v=orderWeight[i].second;
                inv[i]=v;
                rank[v]=i;
                std::cout<<i<<"-("<<v<<","<<orderWeight[i].first<<") "<<endl;//to be deleted
            }
           std::cout<<endl;//to be deleted
            std::cout<<"*****************************calcWeightByOrder finished!**************************"<<endl;
        }

        /*
        *@description: used to init all iterms' values
        *@author: wanjingyi
        *@date: 2020-12-13
        */
        bool initIterms(){
            //inv rank
            inv.resize(numOfVertices);
            rank.resize(numOfVertices);
            //isCnstruct
            isConstruct.resize(numOfVertices,false);
            //query frequency
            _freq.resize(numOfVertices,0);
            _freq_rank.resize(numOfVertices,0);
            _freq_inv.resize(numOfVertices);
            numOfHFpoint=0;
            HFPointIndex.resize(numOfVertices,0);
            //degree
            _degree.resize(numOfVertices,0);
            _degree_rank.resize(numOfVertices,0);
            _degree_inv.resize(numOfVertices);
            //betwenness
            _betwenness.resize(numOfVertices,0);
            _betwenness_inv.resize(numOfVertices);
            _betwenness_rank.resize(numOfVertices,0);
            //coverage
            _coverage.resize(numOfVertices,0);
            _coverage_inv.resize(numOfVertices);
            _coverage_rank.resize(numOfVertices,0);
            //depth
            _depth_rank.resize(numOfVertices,0);
            _depth.resize(numOfVertices,0);
            _depth_inv.resize(numOfVertices);
        }

        /*
         *@description: used to read betwenness order from file
         *@author: wanjingyi
         *@date: 2020-12-13
        */
        void getBetwennessOrderFromFile(char* load_filename){
            std::cout<<"**********************getBetwennessOrderFromFile begins!**********************"<<endl;
            ifstream in(load_filename);//input betwenness file to ifstream
            if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
            NodeID id;
            for(NodeID i=0;i<numOfVertices;++i){
                in>>id;
                _betwenness_rank[id]=i;
                _betwenness_inv[i]=id;
            }
            in.close();
            std::cout<<"**********************getBetwennessOrderFromFile finished!**********************"<<endl;
        }

        void getBetwennessFromFile(){

        }

        /*
         *@description: used to read coverage order from file
         *@author: wanjingyi
         *@date: 2020-12-13
        */
        void getCoverageOrderFromFile(char* load_filename){
            std::cout<<"***********************getCoverageOrderFromFile begins!*************************"<<endl;
            ifstream in(load_filename);//input coverage file to ifstream
            if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
            NodeID id;
            for(NodeID i=0;i<numOfVertices;++i){
                in>>id;
                _coverage_rank[id]=i;
                _coverage_inv[i]=id;
            }
            in.close();
            std::cout<<"***********************getCoverageOrderFromFile finished!!*************************"<<endl;
        }

        void getCoverageFromFile(){

        }

        /*
         *@description: used to get the degree of each node
         *@author: wanjingyi
         *@date: 2020-12-13
        */
        void getDegree(WGraph& wgraph){
            std::cout<<"*************getDegree begins!*************"<<endl;
            int cnt[6]={0};//num of vertices of degree 0,1,2,3,4,>4
            vector<pair<float,NodeID>> deg;//store the degree
			deg.resize(numOfVertices);
            srand(100);
			for(NodeID v=0;v<numOfVertices;++v){
				unsigned int degree=wgraph.vertices[v+1]-wgraph.vertices[v];
				deg[v]=make_pair((float)degree+float(rand()) / RAND_MAX,v);
                _degree[v]=degree;
				if(degree<=4) cnt[degree]++;
				else cnt[5]++;
			}
			sort(deg.rbegin(),deg.rend());
			std::cout<<"vertices num of degree: 0-"<<cnt[0]<<" 1-"<<cnt[1]<<" 2-"<<cnt[2]<<" 3-"<<cnt[3]<<" 4-"<<cnt[4]<<" >4-"<<cnt[5]<<endl;

            for(NodeID i=0;i<=numOfVertices;++i){
                    NodeID v=deg[i].second;
                    NodeID d=deg[i].first;
                    _degree_inv[v]=i;
                    _degree_rank[i]=v;
            }
            max_degree=deg[0].first;
            min_degree=deg[deg.size()-1].first;
            interval_degree=max_degree-min_degree;
            std::cout<<"interval_degree = "<<interval_degree<<endl;
            std::cout<<"*************getDegree finished!*************"<<endl;
        }

        /*
         *@description: used to load frequency and hfpoint
         *@author: wanjingyi
         *@date: 2020-12-13
        */
        void load_HFpoint(char* load_filename,int hfRate=50){ //hfRate/1000
            std::cout<<"***********load hfpoint begins************"<<endl;
            numOfHFpoint = 0;//first line is the number of HFpoints
            numOfHFpoint = static_cast<int> ( (double)numOfVertices*hfRate/(double)1000);
            if(numOfHFpoint<=0) cout<<"error:numOfHFpoint<=0"<<endl;
            cout<<"numOfHFpoint  = "<<numOfHFpoint <<endl;
            ifstream in(load_filename);//input HFPoint file to ifstream
            if(!in.is_open()) {cerr<<"Cannot open "<<load_filename<<endl;}
            vector<pair<NodeID,NodeID> > queryFreq;//(freq,id)
            NodeID id,freq;
            //read each line representing HFpoint to vector 
            for(NodeID i=0;i<numOfVertices;++i){
                in>>id>>freq;
                _freq[id]=(weightType)freq;
                queryFreq.push_back(make_pair(freq,id));
            }
            //sort the node by query times
            //sort(queryFreq.rbegin(),queryFreq.rend());//descending order
            //initialize query freq information
            for(NodeID i=0;i<numOfVertices;++i){
                NodeID v=queryFreq[i].second;
                _freq_inv[i]=v;
                _freq_rank[v]=i;
                if(i<numOfHFpoint) HFPointIndex[v]=true;
            }
            max_freq=queryFreq[0].first;
            min_freq=queryFreq[queryFreq.size()-1].first;
            interval_freq=max_freq-min_freq;
            std::cout<<"max_query_time = "<<max_freq<<endl;
            std::cout<<"min_query_time = "<<min_freq<<endl;
            std::cout<<"interval_freq = "<<interval_freq<<endl;
            std::cout<<"***********load hfpoint finished************"<<endl;
        }

};



#endif //FREQUENCY_HIERARCHY_ORDERING