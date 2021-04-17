#ifndef MATRIX_H
#define MATRIX_H
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <assert.h>
using namespace std;

class Matrix
{
public:
    /* data */
    vector<vector<double>> data;
    int shape0;
    int shape1;
    Matrix(int shape0, int shape1);
    Matrix();
    //有转置方法，点乘，矩阵乘
    Matrix transpose();
    Matrix dot(Matrix b);
    void init(double std);
    void assign(vector<double> data);
    //void print_matrix();
    vector<int> argmax(int dim);
    Matrix eyes();
};

Matrix operator-(const Matrix &m1, const Matrix &m2);
Matrix operator+(const Matrix &m1, const Matrix &m2);
Matrix operator*(double a, const Matrix& m2);
#endif