#ifndef _COMMAND_QUERY_PROCESSING
#define _COMMAND_QUERY_PROCESSING


#include "../src/graph.h"
#include "../src/ordering.h"
#include "../src/construction.h"
#include "../src/coverage_ordering.h"
#include "../src/coverage_ordering_path.h"
#include "../src/coverage_ordering_bp.h"
#include "../src/coverage_ordering_compress.h"
#include "../src/labels.h"
#include "../src/time_util.h"
#include "../command.h"
#include<cstring>

using namespace time_util; 

namespace command{
    class QueryProcessing: public Command{
    public:
        void exit_with_help(){
            printf("Usage:\n");
			printf("\tsspexp_run -q -d [directedGraphFlag] -s [specialFlag] [-o [orderFileName]] -l [inputLabelFileName] -n [queryNum] \n");
            printf("-------------------------------------------------------------------\n");
            printf("Parameter explanation:\n");
            printf("\t[directedGraphFlag]: 0 or 1, for undirected and directed graphs, default is 0\n");
            printf("\t[weightedGraphFlag] = 0 or 1, for unweighted and weighted graphs, default is 0\n");
            printf("\t[specialFlag] = 0: default label\n \t\t\t1: path label\n \t\t\t2: bp label\n \t\t\t3: HLC label\n \t\t\t4: HLCM label\n");
            printf("\t[queryNum]: the number of randomly generated non-duplicated reachable queries. Extra half of queryNum will be generated for warm-up\n");
            printf("\t[orderFileName]: only required for path query processing\n");

            printf("-------------------------------------------------------------------\n");
            printf("Examples:\n");
            printf("Query processing for 1000000 random queries for directed path label alabel_path.label\n");
			printf("\tsspexp_run -q -d 1 -s 1 -l alabel_path.label -n 1000000 \n");
            printf("Query processing for 1000000 random queries for directed HLC label alabel_hlc.label with alabel_hlc.order\n");
            printf("\tsspexp_run -q -d 1 -s 1 -o alabel_hlc.order -l alabel_hlc.label -n 1000000 \n");
            printf("-------------------------------------------------------------------\n");
            exit(1);
        }
        int main(int argc, char *argv[])
        {
            char graphFileName[255] = "";
            char labelFileName[255] = "";
            char orderFileName[255] = "";
            int t_directed_flag = 0;
            int t_weighted_flag = 0;
            int t_special_flag = 0;
            int t_ordering_flag = 0;
			int numQuery = 1000000;
			int warmup = numQuery / 2;
            double beta = 1;
			/***Hfpoint fil by wanjingyi*****/
			bool isHF=false; //flag to determine whether use high frequency
			char hfpointFIleName[255]="";
			 char labelSizeFileName[255] = ""; //filename used to save label size modified by wanjingyi
			 int hfRate=5; //high frequency point rate 1/1000
			 int updateModel=0; //update cases:0-default noUpdate,1-update low,2-update high
			 int queryModel=0; //queryModel 0-default:random
			 char qdisFileName[256]=""; //query distance file
			 bool isOutputDis=false;//write query distance to file or not
			 char ulabelFileName[256]=""; //update lables list file
			 bool isOutputUpdateLabel=false;//write update lables list to file
			 vector<EdgeWeight> distance_p; //store the query distance result by original labels
			 vector<EdgeWeight> distance_hf; //store the query distance result by update labels


            if(argc < 14) // modified by wanjingyi
                exit_with_help();
            
            for(int i = 2; i < argc; i++){
                if(argv[i][0] != '-') break;
                if(++i >= argc)
                    exit_with_help();
                switch (argv[i-1][1]){
                    case 'd':
                        t_directed_flag = atoi(argv[i]);
                        break;
                    case 'w':
                        t_weighted_flag = atoi(argv[i]);
                        break;
                    case 's':
                        t_special_flag = atoi(argv[i]);
                        break;
                    case 'l':
                        strcpy(labelFileName, argv[i]);
						cout<<" labelFileName="<<labelFileName<<endl;
                        break;
                    case 'o':
                        strcpy(orderFileName, argv[i]);
                        if(argc < 14)
                            exit_with_help();
					    cout<<" orderFileName="<<orderFileName<<endl;
                        break;
					case 'i':
                        strcpy(labelSizeFileName,argv[i]);
						cout<<"labelSizeFileName = "<<labelSizeFileName<<endl;
                        break;
					case 'h':
                        strcpy(hfpointFIleName,argv[i]); ///modified by wanjingyi
						isHF=true;
						cout<<"hfpointFIleName = "<<hfpointFIleName<<endl;
					case 'v':
                        strcpy(qdisFileName,argv[i]); ///modified by wanjingyi
						isOutputDis=true;
						cout<<"qdisFileName = "<<qdisFileName<<endl;
                        break;
					case 'u':
                        strcpy(ulabelFileName,argv[i]); ///modified by wanjingyi
						isOutputUpdateLabel=true;
						cout<<"ulabelFileName = "<<ulabelFileName<<endl;
                        break;
					case 'n':
                        numQuery = atoi(argv[i]);
						warmup = numQuery / 2;
                        break;
					case 'r':
                        hfRate = atoi(argv[i]);
                        break;
					case 'c':
                        updateModel = atoi(argv[i]);
                        break;
					case 't':
                        queryModel = atoi(argv[i]);
                        break;
                    default:
                        exit_with_help();
                }
            }
            
            if (t_directed_flag == 1)
                DIRECTED_FLAG = true;
            if (t_weighted_flag == 1)
                WEIGHTED_FLAG = true;
            
            if(t_directed_flag != 1 && t_directed_flag != 0)
                exit_with_help();
            if(t_weighted_flag != 1 && t_weighted_flag != 0)
                exit_with_help();
            if(t_special_flag <0 || t_special_flag > 4)
                exit_with_help();
            if(numQuery <0 )
                exit_with_help();
			
            vector<pair<int, int> > queries(numQuery + warmup);
            srand(31101982);
			
			// Query processing
            if(t_special_flag == 0){ // default labels
                if (DIRECTED_FLAG == true){
					DLabel lab;	
					lab.load_labels(labelFileName);
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p(s,t);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
                 }else{	
					HFLabel lab;
					cout<<"*********************load_labels begins!*****************"<<endl;	
					lab.load_labels(labelFileName);
					cout<<"*********************load_labels successffully!*****************"<<endl;

					/****************test written by wanjingyi*****************/
					cout<<"**********************load_order begins!****************"<<endl;
					lab.load_order(orderFileName); //read points' order from file
					cout<<"**********************load_order successffully!****************"<<endl;	

					cout<<"**********************load_HFpoint begins!****************"<<endl;
					lab.load_HFpoint(hfpointFIleName,hfRate); //read HFpoint from file
					cout<<"**********************load_HFpoint successffully!****************"<<endl;				

					cout<<"********************load_label_size begins!****************"<<endl;
					lab.load_label_size(labelSizeFileName); //read label size from file
					cout<<"********************load_label_size successfully!****************"<<endl;

					// cout<<"*******************update_labels begins!*******************"<<endl;
					// cout<<"updateModel = "<<updateModel<<endl;
					// if(updateModel !=0)
					// {
					// 	if(updateModel==1) lab.update_low_labels();
					// 	else lab.update_high_labels();
					// }
					// cout<<"*******************update_labels successfully!*******************"<<endl;
					/****************test written by wanjingyi*****************/
					//initialize distance vector
					distance_p.resize(numQuery,0);
					distance_hf.resize(numQuery,0);
					cout<<"*****************query construction  begins*********************"<<endl;
					for(int i = 0; i < numQuery + warmup; ++i){
						int s1,t1,s_rand,t_rand;
						switch (queryModel)
						{
						case 0: //random for no update
							s1= rand()%numOfVertices;
							t1= rand()%numOfVertices;
							break;
						case 1: //s-t:h-l
							s_rand=rand()%lab.numOfHFpoint;
							s1= lab.HFOripoint[s_rand];
							t_rand=rand()%(lab.LFOripoint.size());
							t1= lab.LFOripoint[t_rand];
							break;
						case 2: //(s-t):(l:h)
							s_rand=rand()%(lab.LFOripoint.size());
							s1= lab.LFOripoint[s_rand];
							t_rand=rand()%lab.numOfHFpoint;
							t1= lab.HFOripoint[t_rand];
							break;
						case 3: //4-(s-t):(h-h)
							s_rand=rand()%lab.numOfHFpoint;
							s1= lab.HFOripoint[s_rand];
							t_rand=rand()%lab.numOfHFpoint;
							t1= lab.HFOripoint[t_rand];
							break;
						case 4: //5-(s-t):(l-l)
							s_rand=rand()%(lab.LFOripoint.size());
							s1= lab.LFOripoint[s_rand];
							t_rand=rand()%(lab.LFOripoint.size());
							t1= lab.LFOripoint[t_rand];
							break;
						default:
							break;
						}
						double distance = lab.query_p(s1,t1); //modified by wanjingyi
						if(s1 == t1){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s1, t1);
					}
					cout<<"*****************query construction  finished!*********************"<<endl;
					cout<<"*****************query_p original labels*********************"<<endl;
					double start_time,total_time,qtime;
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);//modified by wanjingyi
					}				
					start_time = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						distance_p[i-warmup]=lab.query_p(s,t); //modified by wanjingyi
					}
					total_time=GetCurrentTimeSec()-start_time;
					qtime = total_time / (double)numQuery;
					cout << "Total query time:" << total_time * 1e6 <<  " microseconds" << endl;
					cout << "Avarage query time:" << qtime * 1e6 <<  " microseconds" << endl;
					cout<<"*****************query_p original labels*********************"<<endl;

					if(updateModel !=0)
					{
						cout<<"*******************update_labels begins!*******************"<<endl;
						cout<<"updateModel = "<<updateModel<<endl;
						if(updateModel==1){
							lab.update_low_labels();
							if(isOutputUpdateLabel) lab.write_update_labels(ulabelFileName);
							cout<<"*******************update_labels successfully!*******************"<<endl;
							for(int i = 0; i < warmup; ++i){
								int s = queries[i].first;
								int t = queries[i].second;	
								lab.query_lf(s,t);//modified by wanjingyi
							}
							cout<<"*****************query update labels begins!*********************"<<endl;
							start_time = GetCurrentTimeSec();
							for(int i = warmup; i < warmup + numQuery; ++i){
								int s = queries[i].first;
								int t = queries[i].second;	
								//cout<<"query i="<<i-warmup;
								distance_hf[i-warmup]=lab.query_lf(s,t); //modified by wanjingyi
							}
							total_time=GetCurrentTimeSec()-start_time;
						}else if(updateModel==2){
							lab.update_high_labels();
							if(isOutputUpdateLabel) lab.write_update_labels(ulabelFileName);
							cout<<"*******************update_labels successfully!*******************"<<endl;
							for(int i = 0; i < warmup; ++i){
							int s = queries[i].first;
							int t = queries[i].second;	
							lab.query_hf(s,t);//modified by wanjingyi
							}		
							cout<<"*****************query update labels begins!*********************"<<endl;
							start_time = GetCurrentTimeSec();
							for(int i = warmup; i < warmup + numQuery; ++i){
								int s = queries[i].first;
								int t = queries[i].second;	
								distance_hf[i-warmup]=lab.query_hf(s,t); //modified by wanjingyi
							}
							total_time=GetCurrentTimeSec()-start_time;
						}

						qtime = total_time / (double)numQuery;
						cout << "Total query time:" << total_time * 1e6 <<  " microseconds" << endl;
						cout << "Avarage query time:" << qtime * 1e6 <<  " microseconds" << endl;
						cout<<"*****************query update labels finished!*********************"<<endl;
					}
					
					if(isOutputDis){
						cout<<"*************save query distance to file begins********"<<endl;		 
						lab.save_query_distance(queries,qdisFileName,distance_p,distance_hf,numQuery,warmup,queryModel,updateModel);
						cout<<"*************save query distance to file successfully********"<<endl;
					}


				 }
            } else if(t_special_flag == 1){ // path labels
                if (DIRECTED_FLAG == true){
					DPLabel lab;	
					lab.load_labels(labelFileName);
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p(s,t);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					
					ifstream order_ifs(orderFileName);
					vector<NodeID> rank(numOfVertices);
					vector<NodeID> inv(numOfVertices);
					for(int i = 0; i < numOfVertices; ++i){
						NodeID tv;
						order_ifs >> tv;
						rank[tv] = i;
						inv[i] = tv;
					}
					
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_path(s,t,  rank, inv);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_path(s,t, rank, inv);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
				}else{
					PLabel lab;	
					lab.load_labels(labelFileName);
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p(s,t);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					
					ifstream order_ifs(orderFileName);
					vector<NodeID> rank(numOfVertices);
					vector<NodeID> inv(numOfVertices);
					for(int i = 0; i < numOfVertices; ++i){
						NodeID tv;
						order_ifs >> tv;
						rank[tv] = i;
						inv[i] = tv;
					}
					
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_path(s,t,  rank, inv);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_path(s,t, rank, inv);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
				}
            } else if(t_special_flag == 2){ // BP labels
                if (DIRECTED_FLAG == true){	
					DBPLabel<50> lab;	
					lab.load_labels(labelFileName);
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p(s,t);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
				}else{
					BPLabel<50> lab;	
					lab.load_labels(labelFileName);
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p(s,t);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s,t);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
				}
            } else if(t_special_flag == 3){ // HLC labels
                if (DIRECTED_FLAG == true){		
					CLabel lab;	
					lab.load_labels_d(labelFileName);
                    
                    long ts = 0;
                    vector<NodeID> dis_vec(numOfVertices, INF_WEIGHT);
                    vector<long> ts_vec(numOfVertices, -1);
                    vector<NodeID> que(numOfVertices);
                    vector<EdgeWeight> que_d(numOfVertices);
                    
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p_d(s, t, ts++, dis_vec, ts_vec, que, que_d);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p_d(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p_d(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
				}else{
					CLabel lab;	
					lab.load_labels(labelFileName);
                    
                    long ts = 0;
                    vector<NodeID> dis_vec(numOfVertices, INF_WEIGHT);
                    vector<long> ts_vec(numOfVertices, -1);
                    vector<NodeID> que(numOfVertices);
                    vector<EdgeWeight> que_d(numOfVertices);
                    
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p(s, t, ts++, dis_vec, ts_vec, que, que_d);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;
                }
            } else if(t_special_flag == 4){ // HLCM labels
                if (DIRECTED_FLAG == true){
					CLabel lab;	
					lab.load_two_level_labels_d(labelFileName);
                    
                    long ts = 0;
                    vector<NodeID> dis_vec(numOfVertices, INF_WEIGHT);
                    vector<long> ts_vec(numOfVertices, -1);
                    vector<NodeID> que(numOfVertices);
                    vector<EdgeWeight> que_d(numOfVertices);
                    
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p_two_level_d(s, t, ts++, dis_vec, ts_vec, que, que_d);
						if(s == t){
							i--;
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p_two_level_d(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p_two_level_d(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;					
				}else{       
					CLabel lab;	
					lab.load_two_level_labels(labelFileName);
                    
                    
                    long ts = 0;
                    vector<NodeID> dis_vec(numOfVertices, INF_WEIGHT);
                    vector<long> ts_vec(numOfVertices, -1);
                    vector<NodeID> que(numOfVertices);
                    vector<EdgeWeight> que_d(numOfVertices);
                    
					for(int i = 0; i < numQuery + warmup; ++i){
						int s = rand()%numOfVertices;
						int t = rand()%numOfVertices;
						double distance = lab.query_p_two_level(s, t, ts++, dis_vec, ts_vec, que, que_d);
						if(s == t){
							i--; 
							continue;
						}		
						
						if(distance >= INF_WEIGHT){
							i--;
							continue;
						}		
						queries[i]= make_pair(s, t);
					}
					for(int i = 0; i < warmup; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p_two_level(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					
					double qtime = GetCurrentTimeSec();
					for(int i = warmup; i < warmup + numQuery; ++i){
						int s = queries[i].first;
						int t = queries[i].second;	
						lab.query_p_two_level(s, t, ts++, dis_vec, ts_vec, que, que_d);
					}
					qtime = (GetCurrentTimeSec() - qtime) / (double)numQuery;
					cout << "Reponse time:" << qtime * 1e6 <<  " microseconds" << endl;					
								
                }
            }
        }
    };
}
#endif 
