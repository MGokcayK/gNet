// 
//     tester source file of gNet_CPP.
//     
//     This file helps testing gNet_CPP ops 
//	   by comparing gNet_py results.
// 


//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 04 / 09 / 2020
//                 Creating file.
// 


#include "../include/ops.hpp"

extern "C" {                 // Make sure we have C-declarations in C++ programs

    void cpp_ops_add_test  (float* data1, bool grad1, int ndim1, int* np_shape1,  
                            float* data2, bool grad2, int ndim2, int* np_shape2, 
                            float* py_result, int ndim_out, int* np_shape_py_res)
    {
        std::vector<int> dyn_shape1, dyn_shape2, dyn_shape_out;
        for (int i = 0; i < ndim1; i++) dyn_shape1.push_back(np_shape1[i]);
        for (int i = 0; i < ndim2; i++) dyn_shape2.push_back(np_shape2[i]);
        size_t py_result_size = 1; 
        for (int i = 0; i < ndim_out; i++) 
        {
            py_result_size = py_result_size * np_shape_py_res[i];
        }

        gNet::Tensor t1(data1, dyn_shape1, grad1);
        gNet::Tensor t2(data2, dyn_shape2, grad2);
        Eigen::Map<Eigen::VectorXf> py_result_vector(py_result, py_result_size);

        auto result = ops::add(&t1, &t2);

        result->backward();

        Eigen::Map<Eigen::VectorXf> cpp_result_vector(result->tensor_of_node->value.data(), result->tensor_of_node->value.size());

        std::cout << "Results are equal : " << py_result_vector.isApprox(cpp_result_vector) << "\n";
    }

    void cpp_ops_add_grad_test1(float* data1, bool grad1, int ndim1, int* np_shape1,  
                                float* data2, bool grad2, int ndim2, int* np_shape2, 
                                float* py_result, int ndim_out, int* np_shape_py_res)
    {
        std::vector<int> dyn_shape1, dyn_shape2, dyn_shape_out;
        for (int i = 0; i < ndim1; i++) dyn_shape1.push_back(np_shape1[i]);
        for (int i = 0; i < ndim2; i++) dyn_shape2.push_back(np_shape2[i]);
        size_t py_result_size = 1; 
        for (int i = 0; i < ndim_out; i++) 
        {
            py_result_size = py_result_size * np_shape_py_res[i];
        }

        gNet::Tensor t1(data1, dyn_shape1, grad1);
        gNet::Tensor t2(data2, dyn_shape2, grad2);
        Eigen::Map<Eigen::VectorXf> py_result_vector(py_result, py_result_size);

        auto result = ops::add(&t1, &t2);

        result->backward();

        Eigen::Map<Eigen::VectorXf> cpp_result_vector(t1.grad.data(), t1.grad.size());

        std::cout << "Results are equal : " << py_result_vector.isApprox(cpp_result_vector) << "\n";
    }

    void cpp_ops_add_grad_test2(float* data1, bool grad1, int ndim1, int* np_shape1,  
                                float* data2, bool grad2, int ndim2, int* np_shape2, 
                                float* py_result, int ndim_out, int* np_shape_py_res)
    {
        std::vector<int> dyn_shape1, dyn_shape2, dyn_shape_out;
        for (int i = 0; i < ndim1; i++) dyn_shape1.push_back(np_shape1[i]);
        for (int i = 0; i < ndim2; i++) dyn_shape2.push_back(np_shape2[i]);
        size_t py_result_size = 1; 
        for (int i = 0; i < ndim_out; i++) 
        {
            py_result_size = py_result_size * np_shape_py_res[i];
        }

        gNet::Tensor t1(data1, dyn_shape1, grad1);
        gNet::Tensor t2(data2, dyn_shape2, grad2);
        Eigen::Map<Eigen::VectorXf> py_result_vector(py_result, py_result_size);

        auto result = ops::add(&t1, &t2);

        result->backward();

        Eigen::Map<Eigen::VectorXf> cpp_result_vector(t2.grad.data(), t2.grad.size());

        std::cout << "Results are equal : " << py_result_vector.isApprox(cpp_result_vector) << "\n";
    }


}