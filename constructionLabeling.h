#ifndef _COMMAND_CONSTRUCTION_LABELING
#define _COMMAND_CONSTRUCTION_LABELING

#include <cstring>
#include "../command.h"
#include "../src/time_util.h"
#include "../src/paras.h"
#include "../src/frequency_hierarchy_ordering.h"

namespace command{
    class ConstructLabel:public Command{
        public:
            void exit_with_help(){
                printf("Usage:\n");
                printf("\tsspexp_run -z -d [directedGraphFlag] -w [weightedGraphFlag] -s [specialFlag] -m [indexingSchemes] [-o [rderingSchemes]] -g [graphFileName] \n -e [exportLabelFileName] -q [query_freq_file] [-p labels_file] [-i [label_size_file]] [-f size_analysis_file] \n [-p labels_file] [-i [label_size_file]] [-f size_analysis_file]\n");
                printf("-------------------------------------------------------------------\n");
            }

            int main(int argc, char* argv[]){
                char graphFileName[255] = "";
                char labelFileName[255] = "";
                int t_directed_flag = 0;
                int t_weighted_flag = 0;
                int t_special_flag = 0;
                int t_ordering_flag = 0; //0-degree, 1-betwenness, 2-node centrality
                int t_indexing_flag=0; //indexing model opetions:
                char queryFreqFileName[255] = ""; //queryTime file name(format:s t queryTime) with no order
                char labelSizeFileName[255] = ""; 
                char labelListFileName[255] = ""; 
                char asizeFileName[256]=""; //analysis size file
                char hfpointFIleName[255]="";//hfpoint file name (format:NodeID queryTimes) by descending order
                bool isOutputAnalysis=false;//whether write analysis size to file
                bool isOutputLabelSize=false;
                bool isOutputLabelList=false;
                bool isLoadQueryFreq=false;
                bool isLoadHFPoint=false;
                int hfRate=50;//default is 50/1000
                int k_deg=0,k_freq=0,k_cov=0,k_dep=0;//coefficients(int)
                bool is_deg=false, is_freq=false, is_cov=false, is_dep=false;

                if(argc<16) exit_with_help();

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
                        case 'm':
                            t_indexing_flag= atoi(argv[i]);
                            break;
                        case 'o':
                            t_ordering_flag = atoi(argv[i]);
                            break;
                        case 'g':
                            strcpy(graphFileName,argv[i]);
                            std::cout<<" graphFileName="<<graphFileName<<endl;
                            break;
                        case 'e':
                            strcpy(labelFileName, argv[i]);
                            std::cout<<"labelFileName="<<labelFileName<<endl;
                            break;
                        case 'q':
                            strcpy(queryFreqFileName, argv[i]);
                            if(*queryFreqFileName!='\0'){
                                isLoadQueryFreq=true;
                                std::cout<<"queryFreqFileName="<<queryFreqFileName<<endl;
                            }else cerr<<"queryFreqFileName cann't be null!"<<endl;          
                            break;
                        case 'p':
                            strcpy(labelListFileName,argv[i]);
                            isOutputLabelList=true;
                            std::cout<<"labelListFileName="<<labelListFileName<<endl; 
                            break;
                        case 'i':
                            strcpy(labelSizeFileName,argv[i]);
                            isOutputLabelSize=true;
                            std::cout<<"labelSizeFileName="<<labelSizeFileName<<endl;
                            break;
                        case 'f':
                            strcpy(asizeFileName,argv[i]); ///modified by wanjingyi
                            isOutputAnalysis=true;
                            std::cout<<"asizeFileName = "<<asizeFileName<<endl;
                        case 'h':
                            strcpy(hfpointFIleName,argv[i]); ///modified by wanjingyi
                            if(hfpointFIleName=="") cerr<<"hfpointFIleName cannot be null!"<<endl;
                            else{
                                isLoadHFPoint=true;
                                std::cout<<"hfpointFIleName = "<<hfpointFIleName<<endl;
                            }    
                        case 'r':
                            hfRate = atoi(argv[i]);
                            break;
                        case 'a':
                            k_deg = atoi(argv[i]);
                            if(k_deg!=0) is_deg=true;
                            break;
                        case 'b':
                            k_freq = atoi(argv[i]);
                            if(k_freq!=0) is_freq=true;
                            break;
                        case 'c':
                            k_cov = atoi(argv[i]);
                            if(k_cov!=0) is_cov=true;
                            break;
                        case 'j':
                            k_dep = atoi(argv[i]);
                            if(k_dep!=0) is_dep=true;
                            break;
                        default:
                            exit_with_help();
                    }
                }


            if (t_directed_flag == 1)
                DIRECTED_FLAG = true;
            if (t_weighted_flag == 1)
                WEIGHTED_FLAG = true;

            WGraph wgraph;
            CHFGraph chfgraph;
            Graph graph;
            if(t_indexing_flag==0){
                if(WEIGHTED_FLAG == true) wgraph.load_graph(graphFileName);
                else graph.load_graph(graphFileName);
            }
            else if(t_indexing_flag==1){
                chfgraph.load_graph(graphFileName);
            }
            std::cout << numOfVertices << " nodes and " << numOfEdges << " arcs " << endl;
            
            if (numOfVertices == 0 || numOfEdges == 0){
                std::cout << "Corruptted graph file" << endl;
                return 0;
            }

            //indexing
            if(t_special_flag==0){
                std::cout<<"t_special_flag==0 default label"<<endl;
                if(DIRECTED_FLAG==true){
                    std::cout<<"DIRECTED_FLAG==true"<<endl;
                    
                }else{
                    if(WEIGHTED_FLAG==true){
                        std::cout<<"WEIGHTED_FLAG==true"<<endl;
                        if(t_indexing_flag==0){//first indexing way designed by wanjingyi
                            std::cout<<"*************0-constructing labels by the first indexing way.***************"<<endl;
                            if(t_ordering_flag==0){
                                std::cout<<"t_ordering_flag=0:LFPoint order is based on degree."<<endl;
                                double _labeling_time = GetCurrentTimeSec();
                                Hierarchy_fordering hierarchy_forder(wgraph,queryFreqFileName,t_ordering_flag);
                                std::cout << "Indexing time:" << _labeling_time *1e6 <<  " microseconds" << endl;
                                double ave_labeling_time=_labeling_time/(double) numOfVertices;
                                std::cout<<" average indexing time:"<<ave_labeling_time*1e6 <<  " microseconds" << endl;

                                string orderFileName(labelFileName);//name by labelFileName not need to input
                                orderFileName.append(".order");
                                hierarchy_forder.save_rank(orderFileName.c_str());

                                string labelFile(labelFileName);
                                labelFile.append(".label");
                                hierarchy_forder.labels.save_labels(labelFile.c_str());
                                std::cout<<"save_labels successfully!"<<endl;

                                if(isOutputLabelSize){
                                    hierarchy_forder.labels.save_label_size(labelSizeFileName, hierarchy_forder.inv);
                                    std::cout<<"save_label_size succesfully!"<<endl;
                                }

                                if(isOutputLabelList){
                                    hierarchy_forder.labels.write_labels(labelListFileName,hierarchy_forder.inv);
                                    std::cout<<"write_labels succesfully!"<<endl;
                                }
                                if(isOutputAnalysis){
                                    std::cout<<"*************write_analysis_size begins!**************"<<endl;
                                    hierarchy_forder.save_anaylysis_size(asizeFileName);
                                    std::cout<<"*************write_analysis_size finished!*************"<<endl;
                                }
                                return 0;

                            }else if(t_ordering_flag==1){
                                std::cout<<"t_ordering_flag=1:LFPoint order is based on betweeness."<<endl;
                            }
                        }
                        else if(t_indexing_flag==1){ //the second way of heuristic order
                            std::cout<<"**************1-heuristic selection push algorithm************"<<endl;
                            //compute the coefficients  of order
                            Processing::calcCoefficient<double> calcCoef((double)(k_deg/10),(double)(k_freq/10),(double)(k_cov/10),(double)(k_dep/10),is_deg, is_freq, is_cov, is_dep);
                            Synthesis_Ordering synthesis_ordering(chfgraph, calcCoef,hfpointFIleName,hfRate);
                            //save rank to txt file
                            cout<<"******************save_rank begins!****************"<<endl;
                            string orderFileName(labelFileName);
                            orderFileName.append(".order");
                            synthesis_ordering.save_rank(orderFileName.c_str());
                            cout<<"******************save_rank finished!****************"<<endl;
                            cout<<"***************save_labels begins!(binary and txt)*******************"<<endl;
                            //save labels to binary and txt file
                            string labelFile(labelFileName);
                            labelFile.append(".label");
                            synthesis_ordering.labels.save_labels(labelFile.c_str());
                            if(isOutputLabelList) synthesis_ordering.labels.write_labels(labelListFileName,synthesis_ordering.inv);
                            cout<<"***************save_labels successfully!*******************"<<endl;
                            //save label size
                            cout<<"***************save_label_size begins!*******************"<<endl;
                            if(isOutputLabelSize) synthesis_ordering.labels.save_label_size(labelSizeFileName,synthesis_ordering.inv);
                            cout<<"***************save_label_size successfully!*******************"<<endl;
                            //save debug and analysis information
                            if(isOutputAnalysis){
                                cout<<"*************write_analysis_size begins!**************"<<endl;
                                synthesis_ordering.save_analysisSize_to_file(asizeFileName);
                                cout<<"*************write_analysis_size finished!*************"<<endl;
					        }
                        }

                    }else{
                        std::cout<<"WEIGHTED_FLAG==false"<<endl;
                    }
                }
            }else if(t_special_flag==1){
                std::cout<<"t_special_flag==1 full path label"<<endl;
            }else if(t_special_flag==2){
                std::cout<<"t_special_flag==1 bp label"<<endl;
            }else if(t_special_flag==3){
                std::cout<<"t_special_flag==1 HLC label"<<endl;
            }else if(t_special_flag==4){
                std::cout<<"t_special_flag==1 HLCM label"<<endl;
            }
            }
            
    };
}

#endif