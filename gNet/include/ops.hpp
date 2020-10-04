// 
//     ops header of gNet_CPP.
//
//     This header can help to calculate ops and register to the new node.
//     It is top layer of tensor_ops header whihc is core ops of gNet_CPP.
// 

//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 04 / 09 / 2020
//                 Creating file.
// 


#pragma once
#include "graph.hpp"

namespace ops
{
	// making graph node in templated form.
	// this is the base templated function.
	template<typename T>
	GraphNode* make_graph_node(T param)
	{
	}

	// if candidate is gNet::Tensor, create node and register
	// the tensor.
	template <>
	GraphNode* make_graph_node<gNet::Tensor*>(gNet::Tensor* candidate)
	{		
		GraphNode* node = new GraphNode();
		node->register_tensor(candidate);
		return node;
	}


	// if candidate input is already GraphNode, just pass it.
	template <>
	GraphNode* make_graph_node<GraphNode*>(GraphNode* candidate)
	{		
		return candidate;
	}

	// power ops of gNet which check the input is Tensor or GraphNode.
	// If input is GraphNode, it will not create for input and just calculate
	// result node and its value. If it is Tensor, it will create a node for 
	// it then calculate result node and its value.
	template <typename T>
	GraphNode* power(T t1, float power)
	{
		GraphNode* pt1 = ops::make_graph_node(t1);
		tensor_ops::ops_base* ops = new tensor_ops::power(pt1, power);
		GraphNode* newNode = new GraphNode();
		newNode->register_ops(ops);
		return newNode;
	}

	// add ops of gNet which check the input is Tensor or GraphNode.
	// If inputs are GraphNode, it will not create for input and just calculate
	// result node and its value. If they are Tensor, it will create a node for 
	// them then calculate result node and its value.
	template <typename T1, typename T2>
	GraphNode* add(T1 t1, T2 t2)
	{
		GraphNode* pt1 = ops::make_graph_node(t1);
		GraphNode* pt2 = ops::make_graph_node(t2);
		tensor_ops::ops_base* ops = new tensor_ops::add(pt1, pt2);
		GraphNode* newNode = new GraphNode();
		newNode->register_ops(ops);
		return newNode;
	}

	// matmul ops of gNet which check the input is Tensor or GraphNode.
	// If inputs are GraphNode, it will not create for input and just calculate
	// result node and its value. If they are Tensor, it will create a node for 
	// them then calculate result node and its value.
	template <typename T1, typename T2>
	GraphNode* matmul(T1 t1, T2 t2)
	{
		GraphNode* pt1 = ops::make_graph_node(t1);
		GraphNode* pt2 = ops::make_graph_node(t2);
		tensor_ops::ops_base* ops = new tensor_ops::matmul(pt1, pt2);
		GraphNode* newNode = new GraphNode();
		newNode->register_ops(ops);
		return newNode;
	}

	// mul ops of gNet which check the input is Tensor or GraphNode.
	// If inputs are GraphNode, it will not create for input and just calculate
	// result node and its value. If they are Tensor, it will create a node for 
	// them then calculate result node and its value.
	template <typename T1, typename T2>
	GraphNode* mul(T1 t1, T2 t2)
	{
		GraphNode* pt1 = ops::make_graph_node(t1);
		GraphNode* pt2 = ops::make_graph_node(t2);
		tensor_ops::ops_base* ops = new tensor_ops::mul(pt1, pt2);
		GraphNode* newNode = new GraphNode();
		newNode->register_ops(ops);
		return newNode;
	}

}
