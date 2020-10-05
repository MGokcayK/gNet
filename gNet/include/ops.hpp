// 
//     ops header of gNet_CPP.
//
//     This header can help to calculate ops and register to the new node.
//     It is top layer of tensor_ops header whihc is core ops of gNet_CPP.
// 

//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 05 / 09 / 2020
//                 Move make_grap_node from ops.hpp to graph.hpp
// 


#pragma once
#include "graph.hpp"

namespace ops
{
	

	// power ops of gNet which check the input is Tensor or GraphNode.
	// If input is GraphNode, it will not create for input and just calculate
	// result node and its value. If it is Tensor, it will create a node for 
	// it then calculate result node and its value.
	template <typename T>
	GraphNode* power(T t1, float power)
	{
		GraphNode* pt1 = graph::make_graph_node(t1);
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
		GraphNode* pt1 = graph::make_graph_node(t1);
		GraphNode* pt2 = graph::make_graph_node(t2);
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
		GraphNode* pt1 = graph::make_graph_node(t1);
		GraphNode* pt2 = graph::make_graph_node(t2);
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
		GraphNode* pt1 = graph::make_graph_node(t1);
		GraphNode* pt2 = graph::make_graph_node(t2);
		tensor_ops::ops_base* ops = new tensor_ops::mul(pt1, pt2);
		GraphNode* newNode = new GraphNode();
		newNode->register_ops(ops);
		return newNode;
	}

}
