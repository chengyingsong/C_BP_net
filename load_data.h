#ifndef DATA_H
#define DATA_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "Matrix.h"
using namespace std;
int ReverseInt(int i);
void read_Mnist_Label(string filename, Matrix &labels,int len);
void read_Mnist_Images(string filename, Matrix &images,int len);
unordered_map<string, Matrix> load_data();
#endif