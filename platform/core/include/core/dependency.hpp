// 
//     Dependency header of gNet_CPP.

//     Author : @MGokcayK 
//     Create : 04 / 09 / 2020
//     Update : 04 / 09 / 2020
//                 Creating file.
// 

#pragma once
#include <string>

using namespace std;

struct Dependency
{
	// store if ops has grad function calculation
	bool grad_fn_1 = false;
	bool grad_fn_2 = false;
	// store ops name to debug easily
	string ops_name1;
	string ops_name2;
};