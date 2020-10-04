#include "../include/graph.hpp"

void GraphNode::register_tensor(gNet::Tensor* t)
{
	this->tensor_of_node = t;
};

// register tensor ops to graph
void GraphNode::register_ops(tensor_ops::ops_base* ops)
{
	this->node_ops = ops;
	this->tensor_of_node = &ops->output;
	this->pre_node_1 = ops->pGN1;
	this->pre_node_2 = ops->pGN2;
}

void GraphNode::backward(bool first_calling)
{
	if (first_calling)
	{
		////initialize reverse AD
		this->tensor_of_node->grad = this->tensor_of_node->value.constant(0) +
									 this->tensor_of_node->value.constant(1.);
	}
	// if this node has operation
	if (this->node_ops)
	{
		// get the node's operation
		tensor_ops::ops_base* ops = this->node_ops;
		// if node's operation's depends on gradient function 1, calculate it
		// then connect to previous node 1 and call its backward method
		if (ops->output.depends_on.grad_fn_1)
		{
			// calculate node's backward operation 1, because it has grad_fn_1
			auto ops_backward1 = ops->backward1(this->tensor_of_node->grad);
			//cout << ops->output.depends_on.ops_name1 << endl;
			// update grad of previous node 1.
			this->pre_node_1->tensor_of_node->grad += ops_backward1;
			// call its backward method
			this->pre_node_1->backward(false);
		}
		// if node's operation's depends on gradient function 2, calculate it
		// then connect to previous node 2 and call its backward methdo
		if (ops->output.depends_on.grad_fn_2)
		{
			// calculate node's backward operation 2, because it has grad_fn_2
			auto ops_backward2 = ops->backward2(this->tensor_of_node->grad);
			//cout << ops->output.depends_on.ops_name2 << endl;
			// update grad of previous node 2.
			this->pre_node_2->tensor_of_node->grad += ops_backward2;
			// call its backward method
			this->pre_node_2->backward(false);
		}
	}

}