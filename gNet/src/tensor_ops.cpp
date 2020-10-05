#include "../include/tensor_ops.hpp"
#include "../include/graph.hpp"

#ifdef NDEBUG
#define assertG(condition, message) 0
#else
#define assertG(condition, message)\
   (!(condition)) ?\
	  (std::cerr << "Assertion failed: (" << #condition << "), "\
	  << "function " << __FUNCTION__\
	  << ", file " << __FILE__\
	  << ", line " << __LINE__ << "."\
	  << std::endl << message << std::endl, abort(), 0) : 1
#endif


// custom broadcasting function
// inputs :
// - temporary tensor1 of ops
// - temporary tensor2 of ops
// - pointer of GraphNode1
// - pointer of GraphNode2
// - shape variable of ops
void g_broadcasting(Eigen::Tensor<float, 7, Eigen::RowMajor> &t1,
					Eigen::Tensor<float, 7, Eigen::RowMajor> &t2,
					GraphNode* pGN1,
					GraphNode* pGN2,
					std::array<int, 7> &shape,
					std::string ops_name
					)
{
	// make sure that inside dimensions of tensor should be equal else broadcast
	if (pGN1->tensor_of_node->store_shape != pGN2->tensor_of_node->store_shape)
	{
		// map GraphNode1's tensor's store_shape to eigen::vector
		Eigen::Vector<int, Eigen::Dynamic> shape1 = Eigen::Map<Eigen::Vector<int, Eigen::Dynamic>> (pGN1->tensor_of_node->store_shape.data(),
																									pGN1->tensor_of_node->store_shape.size());
		// map GraphNode2's tensor's store_shape to eigen::vector
		Eigen::Vector<int, Eigen::Dynamic> shape2 = Eigen::Map<Eigen::Vector<int, Eigen::Dynamic>> (pGN2->tensor_of_node->store_shape.data(),
																									pGN2->tensor_of_node->store_shape.size());

		// check one of the tensor's NumDim equal 1, broadcast
		// if first tensor's NumDim equal one broadcast w.r.t second tensor dimension
		if (pGN1->tensor_of_node->NumDim == 1)
		{
			// finding if tensor1's element size is same one of tensor2's dimensions element size
			auto itr = std::find(shape2.begin(), 
								 shape2.end(), 
								 shape1[0] );
			// if tensor1's element size is not same as one of tensor2's dimensions element size 
			// or tensor1's element size is not equal 1, give assert
			assertG(((itr != shape1.end()) || (shape1[0] == 1)),
				"\n Dimension of Tensor2 (" << shape1[0] << 
				") should be equal 1 or one of Tensor1's shape for 1D Tensor in mul ops!");
			// if tensor1's element size is equal one, broadcast as same shape as tensor2 and return directly from g_broadcasting function.		
			if (shape1[0] == 1)
			{	
				// broadcast temporary 
				Eigen::Tensor<float, 7, Eigen::RowMajor> temp = t1.broadcast(pGN2->tensor_of_node->ops_shape);
				// reshape t1 to allocate memory
				t1.reshape(pGN2->tensor_of_node->ops_shape);
				// assing temporary to t1
				t1 = temp;
				// assign new shape
				shape = pGN2->tensor_of_node->store_shape;
				return;
			}
			// if tensor1's element size is not one and tensor1's element size is same as one of tensor2's dimensions element size 
			// broadcast tp equal dimension size and rest of them will be 1, then it will broadcasting as regularly which is not 
			// have NumDim == 1 condition.
			else
			{
				// create and initialize broradcasting shape
				std::array<Eigen::Index, 7> broadcast_shape;
				broadcast_shape.fill(1);
				// find index of same element size dimension
				int ind = itr - shape2.begin();
				// set broadcast shape
				broadcast_shape[ind] = shape2[ind];		
				// set related part of broadcast shape to regular shape to 
				// handle regular broadcasting which is not have NumDim == 1 condition
				for (int i =0; i<ind+1 ; i++) 
				{
					shape1[i] = broadcast_shape[i];
				}
				// reshape t1 to get ready regular broadcast
				t1 = t1.reshape(broadcast_shape);
			}	
		}
		// if second tensor's NumDim equal one broadcast w.r.t first tensor dimension
		else if (pGN2->tensor_of_node->NumDim == 1)
		{
			// finding if tensor2's element size is same one of tensor1's dimensions element size
			auto itr = std::find(shape1.begin(), 
								 shape1.end(), 
								 shape2[0] );
			// if tensor2's element size is not same as one of tensor1's dimensions element size 
			// or tensor2's element size is not equal 1, give assert
			assertG(((itr != shape1.end()) || (shape2[0] == 1)),
				"\n Dimension of Tensor2 (" << shape2[0] << 
				") should be equal 1 or one of Tensor1's shape for 1D Tensor in mul ops!");
			// if tensor2's element size is equal one, broadcast as same shape as tensor1 and return directly from g_broadcasting function.										
			if (shape2[0] == 1)
			{	
				std::cout << "BEFORE BROADCAST d2 s1" << std::endl;
				// broadcast temporary 
				Eigen::Tensor<float, 7, Eigen::RowMajor> temp = t2.broadcast(pGN1->tensor_of_node->ops_shape);
				// reshape t2 to allocate memory
				t2.reshape(pGN1->tensor_of_node->ops_shape);
				// assing temporary to t2
				t2 = temp;
				// assign new shape
				shape = pGN1->tensor_of_node->store_shape;
				return;
			}
			// if tensor2's element size is not one and tensor2's element size is same as one of tensor1's dimensions element size 
			// broadcast tp equal dimension size and rest of them will be 1, then it will broadcasting as regularly which is not 
			// have NumDim == 1 condition.
			else
			{
				// create and initialize broradcasting shape
				std::array<Eigen::Index, 7> broadcast_shape;
				broadcast_shape.fill(1);
				// find index of same element size dimension
				int ind = itr - shape1.begin();
				// set broadcast shape
				broadcast_shape[ind] = shape1[ind];		
				// set related part of broadcast shape to regular shape to 
				// handle regular broadcasting which is not have NumDim == 1 condition
				for (int i =0; i<ind+1 ; i++) 
				{
					shape2[i] = broadcast_shape[i];
				}
				// reshape t2 to get ready regular broadcast
				t2 = t2.reshape(broadcast_shape);
			}			
		}

		// regular broadcasting 

		// initialize resulting shape, sign shape and broadcast shape variables
		Eigen::VectorXi result_shape, sign_shape;
		std::array<Eigen::Index, 7> brdcst;
		// find difference between two tensor shape
		result_shape = shape1 - shape2;
		sign_shape = result_shape;
		// replate 0s to 1s
		std::replace(sign_shape.begin(), sign_shape.end(), 0, 1);
		// find which tensor will be broadcasting by finding sign
		// if sign is positive, it means GraphNode2's tensor will be broadcasted
		// if sign is negative, it means GraphNode1's tensor will be broadcasted
		auto sign = std::accumulate(std::begin(sign_shape), std::end(sign_shape), 1, std::multiplies<int>());
		// initalize addition vector
		Eigen::VectorXi add_vec(7);
		add_vec.setConstant(1);
		// broadcasting
		int ind = -1;
		if (sign > 0)
		{
			// find unequal dimensions
			auto equality = shape1.cwiseEqual(shape2);
			for (int i = 0; i < equality.size(); i++)
			{
				if (equality[i] == 0) 
				{
					ind = i;
				}
			}
			// check different dimension will be 1 or not. If it is 1, broadcast, else not
			assertG((shape2[ind] == 1), " It cannot be broadcasting in " + ops_name + " ops!");
			// find result shape
			result_shape = result_shape + add_vec;
			// copy result shape to broadcasting shape which is array of Eigen::Index
			std::copy_n(result_shape.begin(), result_shape.size(), brdcst.begin());
			// assign temporary broadcasting result
			Eigen::Tensor<float, 7, Eigen::RowMajor> temp = t2.broadcast(brdcst);
			// reshape GraphNode2's tensor shape w.r.t GraphNode1's tensor shape
			t2.reshape(pGN1->tensor_of_node->ops_shape);
			// assign store shape by copying related tensor's store shape
			std::copy_n(pGN1->tensor_of_node->store_shape.begin(),
				pGN1->tensor_of_node->store_shape.size(), shape.begin());
			// assign broadcasting result
			t2 = temp;
		}
		else
		{
			// find unequal dimensions
			auto equality = shape2.cwiseEqual(shape1);
			for (int i = 0; i < equality.size(); i++)
			{
				if (equality[i] == 0)
				{
					ind = i;
				}
			}
			// check different dimension will be 1 or not. If it is 1, broadcast, else not
			assertG((shape1[ind] == 1),  " It cannot be broadcasting in " + ops_name + " ops!");
			// find result shape
			result_shape *= -1; // make result shape positive (shape2 - shape1)
			result_shape = result_shape + add_vec;
			// copy result shape to broadcasting shape which is array of Eigen::Index
			std::copy_n(result_shape.begin(), result_shape.size(), brdcst.begin());
			// assign temporary broadcasting result
			Eigen::Tensor<float, 7, Eigen::RowMajor> temp = t1.broadcast(brdcst);
			// reshape GraphNode1's tensor shape w.r.t GraphNode2's tensor shape
			t1.reshape(pGN2->tensor_of_node->ops_shape);
			// assign store shape by copying related tensor's store shape
			std::copy_n(pGN2->tensor_of_node->store_shape.begin(),
				pGN2->tensor_of_node->store_shape.size(), shape.begin());
			// assign broadcasting result
			t1 = temp;
		}
	}
	else
	{
		// when there is no broadcasting, take shape directly one of tensor
		shape = pGN1->tensor_of_node->store_shape;
	}
}





// BASE OPS
tensor_ops::ops_base::ops_base()
{}

void tensor_ops::ops_base::forward()
{}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::ops_base::backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	return this->grad;
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::ops_base::backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	return this->grad;
}





// POWER OPS
tensor_ops::power::power(GraphNode* t, float power)
{
	// assign pointer of GraphNode1
	this->pGN1 = t;
	// initial value of ops
	this->value = this->pGN1->tensor_of_node->value;
	// assign gradient calculation of ops
	this->have_grad = this->pGN1->tensor_of_node->have_grad;
	// assign initial shape of ops
	this->shape = this->pGN1->tensor_of_node->store_shape;
	// power
	this->pwr = power;
	// calculate forward
	this->forward();
}

void tensor_ops::power::forward()
{
	// calculate new value
	this->value = this->value.pow(this->pwr);

	// assign dependencies
	if (this->have_grad)
	{
		this->depends_on.ops_name1 = "_pow";
		this->depends_on.grad_fn_1 = true;
	}

	// assign output
	this->output = gNet::Tensor(this->value.data(), this->shape, this->depends_on, this->have_grad);
}


Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::power::backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// calculate grad to GraphNode1
	if (this->pwr == 0)
	{
		this->grad = grad.constant(0.);
	}
	else if (this->pwr < 0)
	{
		this->grad = this->pwr  * (float)1. / (this->pGN1->tensor_of_node->value.pow((float)(this->pwr - 1))) * grad;
	}
	else
	{
		this->grad = this->pwr * (this->pGN1->tensor_of_node->value.pow((float)(this->pwr - 1))) * grad;
	}
	return this->grad;
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::power::backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// ops has no GraphNode2; thus, no gradient calculation.
	return this->grad;
};





// ADD OPS
tensor_ops::add::add(GraphNode* t1, GraphNode* t2)
{
	// assign pointer of GraphNode1
	this->pGN1 = t1;
	// assign pointer of GraphNode2
	this->pGN2 = t2;
	// Calculate forward
	this->forward();
}

void tensor_ops::add::forward()
{
	// make sure that dimension of tensors are equal
	assertG((this->pGN1->tensor_of_node->NumDim == this->pGN2->tensor_of_node->NumDim),
		"\n Dimension of Tensor1 (" << this->pGN1->tensor_of_node->NumDim <<
		") and Tensor2 (" << this->pGN2->tensor_of_node->NumDim <<
		") should be equal in add ops!");
	// temporary tensor 1 (t1) and tensor 2 (t2) to make ops
	Eigen::Tensor<float, 7, Eigen::RowMajor> t1 = this->pGN1->tensor_of_node->value;
	Eigen::Tensor<float, 7, Eigen::RowMajor> t2 = this->pGN2->tensor_of_node->value;
	// set operation name for asserting in broadcasting
	std::string ops_name = "Add";
	// broadcasting control and apply if it is needed
	g_broadcasting(t1, t2, this->pGN1, this->pGN2, this->shape, ops_name);
	// calculate value of ops
	Eigen::Tensor<float, 7, Eigen::RowMajor> value = t1 + t2;
	// assign whether result will have grad or not
	this->have_grad = (this->pGN1->tensor_of_node->have_grad || this->pGN2->tensor_of_node->have_grad);
	// assign depends_on
	if (this->pGN1->tensor_of_node->have_grad)
	{
		this->depends_on.ops_name1 = "_add1";
		this->depends_on.grad_fn_1 = true;
	}

	if (this->pGN2->tensor_of_node->have_grad)
	{
		this->depends_on.ops_name2 = "_add2";
		this->depends_on.grad_fn_2 = true;
	}
	// assing output tensor
	this->output = gNet::Tensor(value.data(), this->shape, this->depends_on, this->have_grad);
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::add::backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// if broadcasting applied, find the related dimension
	std::array<int, 1> ind_of_one;
	for (unsigned int i = 0; i < this->pGN1->tensor_of_node->store_shape.size(); i++)
	{
		if (this->pGN1->tensor_of_node->store_shape[i] == 1)
		{
			ind_of_one[0] = i;
			// after finding dimension, sum across that dimension
			Eigen::Tensor<float, 6, Eigen::RowMajor> temp(grad.sum(ind_of_one));
			// map to temp data then return it
			this->grad = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(temp.data(),
				this->pGN1->tensor_of_node->ops_shape);
			return this->grad;
		}
	}
	// if broadcating not applied, it means that grad shape is same as output shape
	this->grad = grad;
	return this->grad;
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::add::backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// if broadcasting applied, find the related dimension
	std::array<int, 1> ind_of_one;
	for (unsigned int i =0 ; i < this->pGN2->tensor_of_node->store_shape.size(); i++)
	{
		if (this->pGN2->tensor_of_node->store_shape[i] == 1)
		{
			ind_of_one[0] = i;
			// after finding dimension, sum across that dimension
			Eigen::Tensor<float, 6, Eigen::RowMajor> temp(grad.sum(ind_of_one));
			// map to temp data then return it
			this->grad = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(temp.data(),
				this->pGN2->tensor_of_node->ops_shape);
			return this->grad;
		}
	}
	// if broadcating not applied, it means that grad shape is same as output shape
	this->grad = grad;
	return this->grad;
}





// MATMUL OPS
tensor_ops::matmul::matmul(GraphNode* t1, GraphNode* t2)
{
	// assign pointer of GraphNode1
	this->pGN1 = t1;
	// assign pointer of GraphNode2
	this->pGN2 = t2;
	// calculate forward
	this->forward();
}

void tensor_ops::matmul::forward()
{
	// make sure that number of dimension of tensors is 2
	assertG(((this->pGN1->tensor_of_node->NumDim == 2) == (this->pGN2->tensor_of_node->NumDim == 2)),
		"\n Dimension of Tensor1 (" << this->pGN1->tensor_of_node->NumDim <<
		") and Tensor2 (" << this->pGN2->tensor_of_node->NumDim <<
		") should be equal 2 in matmul ops!" );

	// make sure that inside dimensions of tensor should be equal
	assertG(((this->pGN1->tensor_of_node->store_shape[1]) == (this->pGN2->tensor_of_node->store_shape[0])),
		"\n 2nd Dimension of Tensor1 ("<< this->pGN1->tensor_of_node->store_shape[1] <<
		") and 1st Dimension Tensor2 ("<< this->pGN2->tensor_of_node->store_shape[0] <<
		") should be equal in matmul ops!");

	// map raw data to Eigen::Matrix of Tensor1 (t1) and Tensor2 (t2)
	Eigen::MatrixXf t1, t2;
	t1 = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		(this->pGN1->tensor_of_node->value.data(),
		 this->pGN1->tensor_of_node->store_shape[0],
		 this->pGN1->tensor_of_node->store_shape[1]);

	t2 = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		(this->pGN2->tensor_of_node->value.data(),
		 this->pGN2->tensor_of_node->store_shape[0],
		 this->pGN2->tensor_of_node->store_shape[1]);

	// create temporary Eigen::Matrix to store result
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v;
	v = t1 * t2;

	// Fill shape of output properly
	this->shape.fill(0);
	this->shape[0] = (int)v.rows();
	this->shape[1] = (int)v.cols();

	// TensorMap from Eigen::Matrix to Eigen::Tensor
	Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> val(v.data(), v.size()) ;


	// Assign graphnode has gradient or not
	this->have_grad = (this->pGN1->tensor_of_node->have_grad || this->pGN2->tensor_of_node->have_grad);

	// Assign dependency and ops name
	if (this->pGN1->tensor_of_node->have_grad)
	{
		this->depends_on.ops_name1 = "_matmul1";
		this->depends_on.grad_fn_1 = true;
	}

	if (this->pGN2->tensor_of_node->have_grad)
	{
		this->depends_on.ops_name2 = "_matmul2";
		this->depends_on.grad_fn_2 = true;
	}

	// Assign output as tensor
	this->output = gNet::Tensor(val.data(), this->shape, this->depends_on, this->have_grad);
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::matmul::backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// Create temporary matrix for grad and tensor
	Eigen::MatrixXf t, g;
	t = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		(this->pGN2->tensor_of_node->value.data(),
		 this->pGN2->tensor_of_node->store_shape[0],
	     this->pGN2->tensor_of_node->store_shape[1]);

	g = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		(grad.data(),
		 this->shape[0],
		 this->shape[1]);

	// Create temporary value matrix
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v;
	v = g * t.transpose();

	// Calculate grad
	std::array<Eigen::Index, 7> temp_ops_shape;
	temp_ops_shape.fill(1);
	temp_ops_shape[0] = (int)v.rows();
	temp_ops_shape[1] = (int)v.cols();
	Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> val(v.data(), v.size());
	this->grad = val.reshape(temp_ops_shape);

	// Return
	return this->grad;
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::matmul::backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// Create temporary matrix for grad and tensor
	Eigen::MatrixXf t, g;
	t = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		(this->pGN1->tensor_of_node->value.data(),
			this->pGN1->tensor_of_node->store_shape[0],
			this->pGN1->tensor_of_node->store_shape[1]);

	g = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		(grad.data(),
		 this->shape[0],
	 	 this->shape[1]);

	// Create temporary value matrix
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v;
	v = t.transpose() * g;

	// Calculate grad
	std::array<Eigen::Index, 7> temp_ops_shape;
	temp_ops_shape.fill(1);
	temp_ops_shape[0] = (int)v.rows();
	temp_ops_shape[1] = (int)v.cols();
	Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> val(v.data(), v.size());
	this->grad = val.reshape(temp_ops_shape);

	// Return
	return this->grad;
}






// MUL OPS
tensor_ops::mul::mul(GraphNode* t1, GraphNode* t2)
{
	// assign pointer of GraphNode1
	this->pGN1 = t1;
	// assign pointer of GraphNode2
	this->pGN2 = t2;
	// Calculate forward
	this->forward();
}

void tensor_ops::mul::forward()
{
	// make sure that dimension of tensors are equal or one of them will be 1
	assertG(((this->pGN1->tensor_of_node->NumDim == this->pGN2->tensor_of_node->NumDim ) ||
		(this->pGN1->tensor_of_node->NumDim == 1 || this->pGN2->tensor_of_node->NumDim == 1)),
		"\n Dimension of Tensor1 (" << this->pGN1->tensor_of_node->NumDim <<
		") and Tensor2 (" << this->pGN2->tensor_of_node->NumDim <<
		") should be equal in mul ops!");

	Eigen::Tensor<float, 7, Eigen::RowMajor> t1 = this->pGN1->tensor_of_node->value;
	Eigen::Tensor<float, 7, Eigen::RowMajor> t2 = this->pGN2->tensor_of_node->value;

	
	// if one of both tensor's NumDim not equal one, broadcasting as multidimensional
	std::string ops_name = "Mul";

	g_broadcasting(t1, t2, this->pGN1, this->pGN2, this->shape, ops_name);
	
	this->value = t1 * t2; // this is only `*` because of being Eigen::Tensor. If t1 and t2 are Eigen::Matrix, operation should be t1.array() * t2.array()

	this->have_grad = (this->pGN1->tensor_of_node->have_grad || this->pGN2->tensor_of_node->have_grad);

	if (this->pGN1->tensor_of_node->have_grad)
	{
		this->depends_on.ops_name1 = "_mul1";
		this->depends_on.grad_fn_1 = true;
	}

	if (this->pGN2->tensor_of_node->have_grad)
	{
		this->depends_on.ops_name2 = "_mul2";
		this->depends_on.grad_fn_2 = true;
	}

	this->output = gNet::Tensor(this->value.data(), this->shape, this->depends_on, this->have_grad);
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::mul::backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// create and initialize grad_shape
	std::array<int, 7> grad_shape;
	grad_shape.fill(0);
	// calculate grad dimeension
	int grad_dim = 7 - (int)std::count(this->shape.begin(), this->shape.end(), 0);
	// fill grad_shape
	for (int i=0; i<grad_dim; i++)
	{
		grad_shape[i] = grad.dimension(i);
	}
	// create temporary tensor of grad to calculate bp
	gNet::Tensor tGrad(grad.data(), grad_shape);
	// create its node to handle broadcasting
	GraphNode* pGrad = graph::make_graph_node(&tGrad);
	// create temporary tensor of tensor2 to calculate bp
	Eigen::Tensor<float, 7, Eigen::RowMajor> t2 = this->pGN2->tensor_of_node->value;
	// broadcasting
	std::string ops_name = "Mul_BP1";	
	g_broadcasting(pGrad->tensor_of_node->value , t2, pGrad, this->pGN2, grad_shape, ops_name);
	// calcualte added dimension
	int ndims_added = pGrad->tensor_of_node->NumDim - this->pGN1->tensor_of_node->NumDim;
	// calcualte grad 
	this->grad = pGrad->tensor_of_node->value * t2; 
	// if there is added dimension handle it by summing dimension 0
	for (int n=0; n<ndims_added; n++)
	{	
		// summation index
		std::array<int, 1> ind_of_one = {0};
		// summation
		Eigen::Tensor<float, 6, Eigen::RowMajor> temp(this->grad.sum(ind_of_one));
		// mapping shape
		std::array<Eigen::Index, 7> dim_shape;
		dim_shape.fill(1);
		for (int i=0; i<6; i++) dim_shape[i] = temp.dimension(i);
		// mapping
		this->grad = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(temp.data(),dim_shape) ;
	}
	// if broadcasting applied, find the related dimension
	std::array<int, 1> ind_of_one;
	for (unsigned int i = 0; i < this->pGN1->tensor_of_node->store_shape.size(); i++)
	{
		if (this->pGN1->tensor_of_node->store_shape[i] == 1)
		{
			ind_of_one[0] = i;
			// after finding dimension, sum across that dimension
			Eigen::Tensor<float, 6, Eigen::RowMajor> temp(this->grad.sum(ind_of_one));
			// map to temp data then return it
			this->grad = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(temp.data(),
				this->pGN1->tensor_of_node->ops_shape);
			return this->grad;
		}
	}
	return this->grad;
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::mul::backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	// create and initialize grad_shape
	std::array<int, 7> grad_shape;
	grad_shape.fill(0);
	// calculate grad dimeension
	int grad_dim = 7 - (int)std::count(this->shape.begin(), this->shape.end(), 0);
	// fill grad_shape
	for (int i=0; i<grad_dim; i++)
	{
		grad_shape[i] = grad.dimension(i);
	}
	// create temporary tensor to calculate bp
	gNet::Tensor tGrad(grad.data(), grad_shape);
	// create its node to handle broadcasting
	GraphNode* pGrad = graph::make_graph_node(&tGrad);
	// create temporary tensor of tensor1 to calculate bp
	Eigen::Tensor<float, 7, Eigen::RowMajor> t2 = this->pGN1->tensor_of_node->value;
	// broadcasting
	std::string ops_name = "Mul_BP2";
	g_broadcasting(pGrad->tensor_of_node->value , t2, pGrad, this->pGN1, grad_shape, ops_name);
	// calcualte added dimension
	int ndims_added = this->output.NumDim - this->pGN2->tensor_of_node->NumDim;
	// calcualte grad 
	this->grad = pGrad->tensor_of_node->value  * t2; 
	// if there is added dimension handle it by summing dimension 0
	for (int n=0; n<ndims_added; n++)
	{	
		// summation index
		std::array<int, 1> ind_of_one = {0};
		// summation
		Eigen::Tensor<float, 6, Eigen::RowMajor> temp(this->grad.sum(ind_of_one));
		// mapping shape
		std::array<Eigen::Index, 7> dim_shape;
		dim_shape.fill(1);
		for (int i=0; i<6; i++) dim_shape[i] = temp.dimension(i);
		// mapping
		this->grad = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(temp.data(),dim_shape);
	}
	// if broadcasting applied, find the related dimension
	std::array<int, 1> ind_of_one;
	for (unsigned int i = 0; i < this->pGN2->tensor_of_node->store_shape.size(); i++)
	{
		if (this->pGN2->tensor_of_node->store_shape[i] == 1)
		{
			ind_of_one[0] = i;
			// after finding dimension, sum across that dimension
			Eigen::Tensor<float, 6, Eigen::RowMajor> temp(this->grad.sum(ind_of_one));
			// map to temp data then return it
			this->grad = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(temp.data(),
				this->pGN2->tensor_of_node->ops_shape);
			return this->grad;
		}
	}
	return this->grad;
}
