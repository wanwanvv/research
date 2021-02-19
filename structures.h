#ifndef _STRUCTURE_H
#define _STRUCTURE_H

#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <map>
#include <memory>
#include <algorithm>

using namespace std;
typedef vector<int>::iterator iterator_type;

class node{
public:
    int _curNum{-1};
    //int _repr{-1};
    int _height{-1};
    vector<node*> _childs;
    node(){}
    node(int id):_curNum(id){}
};

class tree{
public:
    vector<node> _nodes;
    node* _root{NULL};
    int _numOfNodes{0};
    //****constructions******
    tree(int n){
        _numOfNodes=n;
        _nodes.resize(n);
        for(size_t i=0;i<_numOfNodes;++i){
            node node_t(i);
            _nodes[i]=node_t;
        }
    }

    void initialize(int n){
        _numOfNodes=n;
        _nodes.resize(n);
        for(size_t i=0;i<_numOfNodes;++i){
            node node_t(i);
            _nodes[i]=node_t;
        }
    }

    void build_test_tree(){
        _root=&_nodes[8];
        _root->_height=0;
        _root->_childs.push_back(&_nodes[1]);
        _root->_childs.push_back(&_nodes[2]);
        _root->_childs.push_back(&_nodes[4]);
        _nodes[1]._height=1;
        _nodes[2]._height=1;
        _nodes[4]._height=1;
        _nodes[1]._childs.push_back(&_nodes[0]);
        _nodes[1]._childs.push_back(&_nodes[3]);
        _nodes[1]._childs.push_back(&_nodes[5]);
        _nodes[0]._height=2;
        _nodes[3]._height=2;
        _nodes[5]._height=2;
        _nodes[0]._childs.push_back(&_nodes[6]);
        _nodes[0]._childs.push_back(&_nodes[7]);
        _nodes[6]._height=3;
        _nodes[7]._height=3;
    }
};

class Naive_rmq{
public:
    //2D array of precomputed
    //_arr[b][a] is the index of the minimum value in the range[a,a+b+1]
    vector<vector<int> > _arr;
    int _n;
    iterator_type _begin;
    iterator_type _end;
    Naive_rmq(iterator_type b,iterator_type e):_n(e-b),_begin(b),_end(e)
    {
        _arr.resize(_n);
        //fill(_arr.begin(),_arr.end(),vector<int>());
        fill_in();
    }

    int query(int u_index,int v_index){
        return _arr[v_index-u_index][u_index];
    }
protected:
    /**
    *Dynamic program to compute the answers to every possible query on the input
    **/
    void fill_in()
    {
        //The first level_arr[0] contains the answers to RMQ queries on interval of length 1,
        //which must just be the first element in the interval
        vector<int>& _arr_0=_arr[0];
        _arr_0.resize(_n);
        for(size_t i=0;i<_n;++i) _arr_0[i]=i;
        for(auto it=_arr.begin();it+1<_arr.end();++it){
            //Zips neighboring indexes of this level with a functor that chooses the index
            // producing the lesser value from each pair of indexes
            transform(it->begin(),it->end()-1,it->begin()+1,back_inserter(*(it+1)),
                      [this](const size_t& x,const size_t& y){
                        return _begin[x]<_begin[y]?x:y;
                      });
        }
    }
};

class Sparse_rmq{
public:
    //variables
    iterator_type _begin;
    iterator_type _end;
    //logn is the depth we need to precompute answers down to
    int _logn{0};
    //2D array of precompute answers,_arr[a][b] is the minimum value in the range [a,a+b]
    vector<vector<int> > _arr;
    int _n;
    //constructions
    Sparse_rmq(iterator_type b,iterator_type e):_n(e-b),_begin(b),_end(e){
        _logn=max(1,int(floor(log2(_n))));
        _arr.resize(_logn+1);
        fill_in();
    }

    int query(int u_index,int v_index) const
    {
        ++v_index;
        const int depth=max(0,int(floor(log2(v_index-u_index))));
        const int p_u=_arr[depth][u_index];
        const int p_v=_arr[depth][v_index-(1<<depth)];
        return _begin[p_u]<_begin[p_v]?p_u:p_v;
    }
protected:
    void fill_in()
    {
        //Each interval of zero length retuning the value at i
        vector<int>& _arr_0=_arr[0];
        _arr_0.resize(_n);
        for(size_t i=0;i<_n;++i) _arr_0[i]=i;
        //The depth goes up to lognn-2^d
        for(int d=0;d<_logn;++d){
            //Interval is length 2^d,the up limit id (n-2^d)
            const int width=1<<d;
            //From the next a array by zipping pairs of elements in the dth array
            //that are width apart,taking the lesser one
            transform(_arr[d].begin(),_arr[d].end(),
                      _arr[d].begin()+width,back_inserter(_arr[d+1]),
                      [this](const int& x, const int& y){
                        return _begin[x]<_begin[y]?x:y;
                      });
        }
    }
};

class LCA{
public:
    //lca variables
    node* _tree_root{NULL};
    vector<int> _euler;
    vector<int> _level;
    vector<int> _index;//get the index by node num
    int _numOfVertices{0};
    int _n{0};
    //rmq variables
    int _logn;
    int _block_size;
    /**
    * Arrays of length 2n/lg(n) where the first array contains the minimum
    * element in the ith block of the input, and the second contains the
    * position of that element (as an offset from the beginning of the
    * original input, not from the beginning of the block).
    */
    vector<int> _super_array_vals;
    vector<int> _super_array_ids;
    typedef std::vector<int> block_identifier;//represent a normalized block
    /**
    *Tha map is responsible for each naive_rmq
    *and fro sure we only build once for each of sub block
    **/
    map<block_identifier,unique_ptr<Naive_rmq> > _sub_block_rmqs;
    /**
    *An array mapping sub blocks(by their indexes) to the naive rmq
    **/
    vector<Naive_rmq*> _sub_block_rmq_array;
    //The sparse RMQ implementation over _super_array_vals
    unique_ptr<Sparse_rmq> _super_rmq;
    //******constructiond*****

    LCA(node* t,int n){
        //initialize variables
        _tree_root=t;
        _numOfVertices=n;
        _n=2*_numOfVertices-1;
        _euler.reserve(_n);
        _level.reserve(_n);
        _index.resize(_numOfVertices,-1);
        //constrcut euler list
        dfs_preprocess(_tree_root);
        //block size
        _n=_level.size();
        _logn=max(1,int(floor(log2(_n))));
        _block_size=max(1,_logn/2);
        cout<<"_logn="<<_logn<<" _block_size="<<_block_size<<endl;//to be deleted
        //ouput
        cout<<"_euler:"<<endl;//to be deleted
        for(size_t i=0;i<_euler.size();++i) cout<<_euler[i]<<" ";//to be deleted
        cout<<endl;//to be deleted
        cout<<"_level:"<<endl;//to be deleted
        for(size_t i=0;i<_level.size();++i) cout<<_level[i]<<" ";//to be deleted
        cout<<endl;//to be deleted
        construct_rmq();
    }

    int lca_query(int u,int v){
        int u_index=_index[u];
        int v_index=_index[v];
        if(v_index<u_index){//swap
            int tmp=u_index;
            u_index=v_index;
            v_index=tmp;
        }
        return _euler[rmq_query(u_index,v_index)];
    }

protected:
    void test_sparse_rmq(){
        cout<<"LCA sparse rmq test:"<<endl;
        Sparse_rmq sparse_rmq(_level.begin(),_level.end());
        for(int i=0;i<_numOfVertices;++i)
        {
            for(int j=i;j<_numOfVertices;++j)
            {
                cout<<i<<"-"<<j<<":";
                int i_index=_index[i];
                int j_index=_index[j];
                if(j_index<i_index){
                    int tmp=i_index;
                    i_index=j_index;
                    j_index=tmp;
                }
                cout<<_euler[sparse_rmq.query(i_index,j_index)]<<endl;
            }
        }
    }

    void test_naive_rmq(){
        Naive_rmq naive_rmq(_level.begin(),_level.end());
        cout<<"LCA naive rmq test:"<<endl;
        for(int i=0;i<_numOfVertices;++i)
        {
            for(int j=i;j<_numOfVertices;++j)
            {
                int i_index=_index[i];
                int j_index=_index[j];
                if(j_index<i_index){
                    int tmp=i_index;
                    i_index=j_index;
                    j_index=tmp;
                }
                cout<<i<<"-"<<j<<":";
                cout<<_euler[naive_rmq.query(i_index,j_index)]<<endl;
            }
        }
    }

    int rmq_query(int u_index,int v_index)
    {

        int u_block_idx=int(u_index/_block_size);
        int u_offset=int(u_index%_block_size);
        int v_block_idx=int(v_index/_block_size);
        int v_offset=int(v_index%_block_size);
        //cout<<"u_index="<<u_index<<" v_index="<<v_index<<" u_block_idx="<<u_block_idx<<" u_offset="<<u_offset<<" v_block_idx="<<v_block_idx<<" v_offset="<<v_offset<<endl;//to be deleted
        Naive_rmq& u_naive=*_sub_block_rmq_array[u_block_idx];
        Naive_rmq& v_naive=*_sub_block_rmq_array[v_block_idx];
        int block_diff=v_block_idx-u_block_idx;
        if(block_diff==0){
            //cout<<"in the same block:"<<endl;//to be deleted
            //u and v are in the same block.One naive_rmq search suffices
            return(u_block_idx*_block_size)+u_naive.query(u_offset,v_offset);
        }else{
            const iterator_type u_block_end=min(_level.end(),_level.begin()+((u_block_idx+1)*_block_size));
            int u_min_idx=(u_block_idx*_block_size)+u_naive.query(u_offset,u_block_end-(_level.begin()+(_block_size*u_block_idx))-1);
            int v_min_idx=(v_block_idx*_block_size)+v_naive.query(0,v_offset);
            //cout<<"u_min_idx="<<u_min_idx<<" v_min_idx="<<v_min_idx<<endl;//to be deleted
            if(block_diff==1){
                //cout<<"in the adjacent block:"<<endl;//to be deleted
                //u and v are in adjacent block,not query super array
                //it doesn't handle zero-length intervals property
                return _level[u_min_idx]<_level[v_min_idx]?u_min_idx:v_min_idx;
            }else{
                //cout<<"in the interval block:"<<endl;//to be deleted
                //Full algorithm, using the sparse RMQ implementation
                //on the super array between u and v's blocks
                int super_idx=_super_rmq->query(u_block_idx+1,v_block_idx-1);
                const int& u_min_val=_level[u_min_idx];
                const int& v_min_val=_level[v_min_idx];
                if(u_min_val<v_min_val){
                    return u_min_val<_super_array_vals[super_idx]?u_min_idx:_super_array_ids[super_idx];
                }else{
                    return v_min_val<_super_array_vals[super_idx]?v_min_idx:_super_array_ids[super_idx];
                }
            }
        }
    }

    void dfs_preprocess(node* curr_node){
        int id=curr_node->_curNum;
        _euler.push_back(id);
        _level.push_back(curr_node->_height);
        if(_index[id]==-1) _index[id]=_euler.size()-1;
        for(size_t i=0;i<curr_node->_childs.size();++i)
        {
            dfs_preprocess(curr_node->_childs[i]);
            _euler.push_back(id);
            _level.push_back(curr_node->_height);
        }
    }

    void construct_rmq(){
        //initialize
        int block_length=ceil((float)_n/(float)_block_size);
        //cout<<" block_size="<<_block_size<<" block_length="<<block_length<<endl;//to be deleted
        _super_array_vals.reserve(block_length);
        _super_array_ids.reserve(block_length);
        //For each block,find the min element and normalize it and compute its naive rmq
        for(iterator_type block_begin=_level.begin();block_begin<_level.end();block_begin+=_block_size)
        {
            const iterator_type block_end=min(block_begin+_block_size,_level.end());
            //Find the min element of the block by brute force
            iterator_type block_min=min_element(block_begin,block_end);
            _super_array_vals.push_back(*block_min);
            _super_array_ids.push_back(block_min-_level.begin());
            //Compute the normalized block
            int init=*block_begin;
            vector<int> normalized_block(block_end-block_begin);
            transform(block_begin,block_end,normalized_block.begin(),[init](const int& val){return val-init;});
            //Find the normalized block int the map of RMQ structures,if not found, construct one
            unique_ptr<Naive_rmq>& naive_ptr=_sub_block_rmqs[normalized_block];
            if(!naive_ptr){
                naive_ptr.reset(new Naive_rmq(normalized_block.begin(),normalized_block.end()));
            }
            //Revord a pointer to the RMQ structure at this sub_block's index
            _sub_block_rmq_array.push_back(naive_ptr.get());
        }
        //cout<<"real block_length="<<_super_array_vals.size()<<" b-e="<<_super_array_vals.end()-_super_array_vals.begin()<<endl;//to be deleted
        //Construct the RMQ structure over the super array
        _super_rmq.reset(new Sparse_rmq(_super_array_vals.begin(),_super_array_vals.end()));
    }
};

#endif // _STRUCTURE_H
