#pragma once
#ifndef MODEL_H
#define MODEL_H
#include"Matrix.h"
#include<vector>
using namespace std;


void ReLU(Matrix& a);
double sigmoid(double x);
void tanh(Matrix& a);
Matrix tanh_derivative(const Matrix& a);
Matrix ReLu_derivative(Matrix& a);

void Softmax(Matrix& a);

vector<Matrix> forward(Matrix& sample_x);
void backprop(vector<Matrix> H, double y);

void load_model(const char* filename);
void save_model(const char* filename);

#endif
