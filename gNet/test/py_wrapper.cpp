// 
//     tester source file of gNet_CPP.
//     
//     This file helps testing gNet_CPP wrapper.
//     Which shows whethre data flowing is correctly or not.
// 

//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 04 / 09 / 2020
//                 Creating file.
// 

#include "../include/tensor.hpp"

extern "C"
{
    void py_wrapper_test(float* data, int ndim, int* np_shape)
    {
        std::vector<int> dyn_shape; 
        for (int i=0; i<ndim; i++)
        {
            dyn_shape.push_back(np_shape[i]);
        }

        gNet::Tensor t_test(data, dyn_shape);

        std::cout << "Hello, from py_wrapper.cpp !\n";
        std::cout << "DIM   :" << t_test.NumDim << std::endl;
        std::cout << "Shape :" << t_test.value.dimensions() << std::endl; 
        std::cout << "Eigen::Tensor :" << std::endl << t_test.value << std::endl;
    }


}