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
					std::array<int, 7> &shape
					)
{
	// make sure that inside dimensions of tensor should be equal else broadcast
	if (pGN1->tensor_of_node->store_shape != pGN2->tensor_of_node->store_shape)
	{
		// map GraphNode1's tensor's store_shape to eigen::vector
		Eigen::Map<Eigen::Vector<int, Eigen::Dynamic>> shape1(pGN1->tensor_of_node->store_shape.data(),
															  pGN1->tensor_of_node->store_shape.size());
		// map GraphNode2's tensor's store_shape to eigen::vector
		Eigen::Map<Eigen::Vector<int, Eigen::Dynamic>> shape2(pGN2->tensor_of_node->store_shape.data(),
															  pGN2->tensor_of_node->store_shape.size());
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
			auto equality = shape2.cwiseEqual(shape1);
			for (int i = 0; i < equality.size(); i++)
			{
				if (equality[i] == 0) 
				{
					ind = i;
				}
			}
			// check different dimension will be 1 or not. If it is 1, broadcast, else not
			assertG((shape2[ind] == 1), " It cannot be broadcasting in Add ops!");
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
			assertG((shape1[ind] == 1), " It cannot be broadcasting in Add ops!");
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

	Eigen::Tensor<float, 7, Eigen::RowMajor> t1 = this->pGN1->tensor_of_node->value;
	Eigen::Tensor<float, 7, Eigen::RowMajor> t2 = this->pGN2->tensor_of_node->value;

	g_broadcasting(t1, t2, this->pGN1, this->pGN2, this->shape);
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
	Eigen::MatrixXf v;
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

	// Free memory.
	v.resize(0,0);
	t1.resize(0,0);
	t2.resize(0,0);

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

	// Free memory
	v.resize(0, 0);
	g.resize(0, 0);
	t.resize(0, 0);

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

	// Free memory
	v.resize(0, 0);
	g.resize(0, 0);
	t.resize(0, 0);

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
	// make sure that dimension of tensors are equal
	assertG((this->pGN1->tensor_of_node->NumDim == this->pGN2->tensor_of_node->NumDim ),
		"\n Dimension of Tensor1 (" << this->pGN1->tensor_of_node->NumDim <<
		") and Tensor2 (" << this->pGN2->tensor_of_node->NumDim <<
		") should be equal in mul ops!");

	// make sure that inside dimensions of tensor should be equal
	assertG((this->pGN1->tensor_of_node->store_shape == this->pGN2->tensor_of_node->store_shape ),
		"\n Shape of Tensor1 and Shape of Tensor2 should be equal in mul ops!");

	this->value = this->pGN1->tensor_of_node->value * this->pGN2->tensor_of_node->value;

	this->have_grad = (this->pGN1->tensor_of_node->have_grad || this->pGN2->tensor_of_node->have_grad);

	this->shape = this->pGN1->tensor_of_node->store_shape;

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
	this->grad = grad * this->pGN2->tensor_of_node->value;
	return this->grad;
}

Eigen::Tensor<float, 7, Eigen::RowMajor> tensor_ops::mul::backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad)
{
	this->grad = grad * this->pGN1->tensor_of_node->value;
	return this->grad;
}
