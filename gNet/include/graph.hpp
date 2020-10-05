// 
//     Graph header of gNet_CPP.
//
//     GraphNode class stores tensor of node, ops and previous node pointers.
// 	   With GraphNode class, BackPropagation algorithm can work which is 
//     different from gNet_py implementation. Yet, main purpose is same.

//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 05 / 09 / 2020
//                 Move make_grap_node from ops.hpp to graph.hpp
// 


# pragma once
#include "tensor_ops.hpp"


class GraphNode
{
public:
	// if node has ops, store here
	tensor_ops::ops_base* node_ops;
	// basic constructor
	GraphNode() {};
	// tensor of node 
	gNet::Tensor* tensor_of_node;
	// previouse node pointers
	GraphNode* pre_node_1;
	GraphNode* pre_node_2;

	// registering tensor on the node
	void register_tensor(gNet::Tensor* t);
	// registering ops on the node and its result as tensor of node
	void register_ops(tensor_ops::ops_base*  ops);
	// apply backpropagation algorithm towards related nodes
	void backward(bool first_calling = true);
};

namespace graph
{
	// making graph node in templated form.
	// this is the base templated function.
	template<typename T>
	GraphNode* make_graph_node(T param)
	{
	}

	// if candidate is gNet::Tensor, create node and register
	// the tensor.
	template <> inline
	GraphNode* make_graph_node<gNet::Tensor*>(gNet::Tensor* candidate)
	{		
		GraphNode* node = new GraphNode();
		node->register_tensor(candidate);
		return node;
	}


	// if candidate input is already GraphNode, just pass it.
	template <> inline
	GraphNode* make_graph_node<GraphNode*>(GraphNode* candidate)
	{		
		return candidate;
	}
}