// 
//     tensor_ops header of gNet_CPP.
//     
//     This file is core tensor operations' header file. 
//	   Base tensor operations done here.
// 


//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 04 / 09 / 2020
//                 Creating file.
// 


#pragma once
#include "tensor.hpp"

class GraphNode;

namespace tensor_ops
{
	class ops_base
	{
	private:
		Eigen::Tensor<float, 7, Eigen::RowMajor> value;
		Eigen::Tensor<float, 7, Eigen::RowMajor> grad;
		std::array<int, 7> shape;
		bool have_grad;

	public:
		GraphNode* pGN1;
		GraphNode* pGN2;
		Dependency depends_on;
		gNet::Tensor output;

		ops_base();

		virtual void forward();

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad);

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad);
	};


	class power : public ops_base
	{

	private:
		Eigen::Tensor<float, 7, Eigen::RowMajor> value;
		Eigen::Tensor<float, 7, Eigen::RowMajor> grad;
		std::array<int, 7> shape;
		float pwr;

		bool have_grad;

	public:
		power(GraphNode* t,
			float power
		);

		virtual void forward() override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;



	};


	class add : public ops_base
	{

	private:
		Eigen::Tensor<float, 7, Eigen::RowMajor> value;
		Eigen::Tensor<float, 7, Eigen::RowMajor> grad;
		std::array<int, 7> shape;
		bool have_grad;

	public:
		add(GraphNode* t1,
			GraphNode* t2);

		virtual void forward() override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

	};



	class matmul : public ops_base
	{

	private:
		Eigen::Tensor<float, 1, Eigen::RowMajor> value;
		Eigen::Tensor<float, 7, Eigen::RowMajor> grad;
		std::array<int, 7> shape;

		bool have_grad;

	public:
		matmul(GraphNode* t1,
			GraphNode* t2);

		virtual void forward() override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

	};



	class mul : public ops_base
	{

	private:
		Eigen::Tensor<float, 7, Eigen::RowMajor> value;
		Eigen::Tensor<float, 7, Eigen::RowMajor> grad;
		std::array<int, 7> shape;

		bool have_grad;

	public:
		mul(GraphNode* t1,
			GraphNode* t2);

		virtual void forward() override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward1(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

		virtual Eigen::Tensor<float, 7, Eigen::RowMajor> backward2(Eigen::Tensor<float, 7, Eigen::RowMajor> grad) override;

	};

}