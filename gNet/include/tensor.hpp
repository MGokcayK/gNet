// 
//     tensor header of gNet_CPP.
//
//     The core of gNet_CPP which creates tensor of gNet.
//     This implementation based on gNet_py; yet, there is 
//     some difference because of linear algebra library.
//     In gNet_CPP, Eigen is lin.alg. library. In gNet_py,
//     Numpy is lin.alg. library. 
//
//    Because of their differences, implementations will be
// 	  slightly different; yet, main approach and results will
//    same. 
//


//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 04 / 09 / 2020
//                 Creating file.
// 

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include "unsupported\Eigen\CXX11\Tensor"
#include "dependency.hpp"

namespace gNet
{
	class Tensor
	{
	public:
		// value and grad of tensor as Eigen::Tensors 7D whihc is maximum dims of gNet for now.
		Eigen::Tensor<float, 7, Eigen::RowMajor> value;
		Eigen::Tensor<float, 7, Eigen::RowMajor> grad;
		// Store and operation shape of tensor. The reaseon behind of seperation of shape is,
		// calculation of NumDims by counting unused dimensions which are 0. 
		std::array<int, 7> store_shape = { 0,0,0,0,0,0,0 };
		// During reshape operations on tensor, ops_shape should be used.
		std::array<Eigen::Index, 7> ops_shape;
		// Number of Dimensions of tensor.
		int NumDim;
		// Set whether tensor has gradient or not.
		bool have_grad;
		// Dependency of tensor which helps to BackProp.
		Dependency depends_on;

		// Basic constructor.
		Tensor() {};

		// Dynamic shaped raw data tensor constructor without depends_on.
		Tensor(
			float* data,
			std::vector<int> dims,
			bool have_grad = false
		)
		{
			// copy dimensions to shape
			std::copy_n(dims.begin(), dims.size(), this->store_shape.begin());
			// make sure that if dims has 0 element (like store_shape) replace to 1
			// for calculate size_tensor properly.
			std::replace(dims.begin(), dims.end(), 0, 1);
			// assign tensor has grad calculation or not
			this->have_grad = have_grad;
			// set number of dimension
			this->NumDim = 7 - (int)std::count(this->store_shape.begin(), this->store_shape.end(), 0);
			// set ops shape
			std::copy_n(this->store_shape.begin(), this->store_shape.size(), this->ops_shape.begin());
			std::fill(this->ops_shape.rbegin(), this->ops_shape.rbegin() + (7 - this->NumDim), 1);
			// map raw data to tensor
			this->value = Eigen::TensorMap<Eigen::Tensor<float, 7 , Eigen::RowMajor>> (data, this->ops_shape);
			// assign grad of tensor
			this->grad = this->value.constant(0);

		};

		// Static shaped raw data tensor constructor without depends_on.
		// The difference between static and dynamic shaped it, if data comes from gNet_py, 
		// just dynamic shaped (vector) data will mapped on the tensor. If tensor data created
		// in gNet_CPP, it has already static shape. Thus; no need copy shape from dynamic shape
		// to static shape.
		// Also, Eigen uses static shape (array). Thus, static shape is needed.
		Tensor(
			float* data,
			std::array<int, 7> dims,
			bool have_grad = false
		)
		{
			// assign dimensions to shape bec input is already array
			this->store_shape = dims;
			// make sure that if dims has 0 element (like store_shape) replace to 1
			// for calculate size_tensor properly.
			std::replace(dims.begin(), dims.end(), 0, 1);
			// assign tensor has grad calculation or not
			this->have_grad = have_grad;
			// set number of dimension
			this->NumDim = 7 - (int)std::count(this->store_shape.begin(), this->store_shape.end(), 0);
			// set ops shape
			std::copy_n(this->store_shape.begin(), this->store_shape.size(), this->ops_shape.begin());
			std::fill(this->ops_shape.rbegin(), this->ops_shape.rbegin() + (7 - this->NumDim), 1);
			// map raw data to tensor
			this->value = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(data, this->ops_shape);
			// assign grad of tensor
			this->grad = this->value.constant(0);

		};

		// Dynamic shaped raw data tensor constructor with depends_on.
		Tensor(
			float* data,
			std::vector<int> dims,
			Dependency depends_on,
			bool have_grad = false
		)
		{
			// copy dimensions to shape
			std::copy_n(dims.begin(), dims.size(), this->store_shape.begin());
			// make sure that if dims has 0 element (like store_shape) replace to 1
			// for calculate size_tensor properly.
			std::replace(dims.begin(), dims.end(), 0, 1);
			// assign tensor has grad calculation or not
			this->have_grad = have_grad;
			// assign whether tensor's dependency or not
			this->depends_on = depends_on;
			// set number of dimension
			this->NumDim = 7 - (int)std::count(this->store_shape.begin(), this->store_shape.end(), 0);
			// set ops shape
			std::copy_n(this->store_shape.begin(), this->store_shape.size(), this->ops_shape.begin());
			std::fill(this->ops_shape.rbegin(), this->ops_shape.rbegin() + (7 - this->NumDim), 1);
			// map raw data to tensor
			this->value = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(data, this->ops_shape);
			// assign grad of tensor
			this->grad = this->value.constant(0);

		};

		// Static shaped raw data tensor constructor with depends_on.
		Tensor(
			float* data,
			std::array<int, 7> dims,
			Dependency depends_on,
			bool have_grad = false
		)
		{
			// assign dimensions to shape bec input is already array
			this->store_shape = dims;
			// make sure that if dims has 0 element (like store_shape) replace to 1
			// for calculate size_tensor properly.
			std::replace(dims.begin(), dims.end(), 0, 1);
			// assign tensor has grad calculation or not
			this->have_grad = have_grad;
			// assign whether tensor's dependency or not
			this->depends_on = depends_on;
			// set number of dimension
			this->NumDim = 7 - (int)std::count(this->store_shape.begin(), this->store_shape.end(), 0);
			// set ops shape
			std::copy_n(this->store_shape.begin(), this->store_shape.size(), this->ops_shape.begin());
			std::fill(this->ops_shape.rbegin(), this->ops_shape.rbegin() + (7 - this->NumDim), 1);
			// map raw data to tensor
			this->value = Eigen::TensorMap<Eigen::Tensor<float, 7, Eigen::RowMajor>>(data, this->ops_shape);
			// assign grad of tensor
			this->grad = this->value.constant(0);
		};



	};

}