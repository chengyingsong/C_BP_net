#include "Matrix.h"
#include "..\C_BP_net\Matrix.h"

Matrix::Matrix(int shape0, int shape1)
{
    //每层都是一个矩阵，输入的数据是(B,784),W是(784,512)
    this->shape0 = shape0;
    this->shape1 = shape1;
    for (int i = 0; i < shape0; i++)
    {
        vector<double> temp(shape1);
        this->data.push_back(temp);
    }
}

Matrix::Matrix() { this->shape0 = 0; this->shape1 = 0; }

Matrix Matrix::transpose()
{
    //对data，从(shape0,shape1)转置成(shape1,shape0)
    Matrix M(shape1, shape0);
    for (int i = 0; i < shape1; i++)
    {
        for (int j = 0; j < shape0; j++)
            M.data[i][j] = data[j][i];
    }
    return M;
}

Matrix Matrix::dot(Matrix b)
{
    //两个矩阵相乘
    assert(shape1 == b.shape0); //检查矩阵shape是否合适
    Matrix M(shape0, b.shape1);
    //M[i][j] = data的第i行和b的第j列相乘相加
    for (int i = 0; i < shape0; i++)
    {
        for (int j = 0; j < b.shape1; j++)
        {
            M.data[i][j] = 0;
            for (int k = 0; k < shape1; k++)
                M.data[i][j] += data[i][k] * b.data[k][j];
        }
    }
    return M;
}

void Matrix::init(double std)
{
    //随机初始化
    int seed = 3; //随机种子
    default_random_engine gen(seed);
    normal_distribution<double> dis(0, std);
    //printf("%d %d\n", this->shape0, this->shape1);
    for (int i = 0; i < this->shape0; i++)
        for (int j = 0; j < this->shape1; j++)
            this->data[i][j] = dis(gen);
}



void Matrix::assign(vector<double> data)
{
    assert(shape0 == 1);
    assert(data.size() == shape1);
    for (int i = 0; i < data.size(); i++)
        this->data[0][i] = data[i];
}



vector<int> Matrix::argmax(int dim)
{
    assert(dim == 0 || dim == 1);
    vector<int> An;
    if (dim == 0)
    {
        for (int i = 0; i < shape0; i++)
        {
            int max_arg = 0;
            for (int j = 1; j < shape1; j++)
            {
                if (data[i][j] > data[i][max_arg]) //i行
                    max_arg = j;
            }
            An.push_back(max_arg);
        }
    }
    else
    {
        for (int i = 0; i < shape1; i++)
        {
            int max_arg = 0;
            for (int j = 1; j < shape0; j++)
            {
                if (data[j][i] > data[max_arg][i]) //i列
                    max_arg = j;
            }
            An.push_back(max_arg);
        }
    }
    return An;
}

Matrix Matrix::eyes()
{
    //把一维的矩阵变成对角矩阵
    assert(shape0 == 1 || shape1 == 1);

    if (shape0 == 1)
    {
        Matrix m(shape1, shape1);
        for (int i = 0; i < shape1; i++)
            m.data[i][i] = data[0][i];
        return m;
    }
    else
    {
        Matrix m(shape0, shape0);
        for (int i = 0; i < shape0; i++)
            m.data[i][i] = data[i][0];
        return m;
    }
}

Matrix operator+(const Matrix &m1, const Matrix &m2)
{
    //只要第二维相同，如果第一维是1，就扩展（广播机制）
    assert(m1.shape1 == m2.shape1);
    assert(m1.shape0 == m2.shape0 || m1.shape0 == 1 || m2.shape0 == 1);
    Matrix m(max(m1.shape0, m2.shape0), m1.shape1);
    if (m1.shape0 == m2.shape0)
    {
        for (int i = 0; i < m1.shape0; i++)
            for (int j = 0; j < m2.shape1; j++)
                m.data[i][j] = m1.data[i][j] + m2.data[i][j];
    }
    else if (m1.shape0 == 1)
    {
        for (int i = 0; i < m2.shape0; i++)
            for (int j = 0; j < m2.shape1; j++)
                m.data[i][j] = m1.data[0][j] + m2.data[i][j];
    }
    else
    {
        for (int i = 0; i < m1.shape0; i++)
            for (int j = 0; j < m1.shape1; j++)
                m.data[i][j] = m1.data[i][j] + m2.data[0][j];
    }

    return m;
}

Matrix operator*(double a, const Matrix& m2)
{
    //重载点乘
    Matrix an(m2.shape0, m2.shape1);
    for (int i = 0; i < m2.shape0; i++)
        for (int j = 0; j < m2.shape1; j++)
            an.data[i][j] = m2.data[i][j] * a;
    return an;
}

Matrix operator-(const Matrix &m1, const Matrix &m2)
{
    assert(m1.shape1 == m2.shape1);
    assert(m1.shape0 == m2.shape0 || m1.shape0 == 1 || m2.shape0 == 1);
    Matrix m(max(m1.shape0, m2.shape0), m1.shape1);
    if (m1.shape0 == m2.shape0)
    {
        for (int i = 0; i < m1.shape0; i++)
            for (int j = 0; j < m2.shape1; j++)
                m.data[i][j] = m1.data[i][j] - m2.data[i][j];
    }
    else if (m1.shape0 == 1)
    {
        for (int i = 0; i < m2.shape0; i++)
            for (int j = 0; j < m2.shape1; j++)
                m.data[i][j] = m1.data[0][j] - m2.data[i][j];
    }
    else
    {
        for (int i = 0; i < m1.shape0; i++)
            for (int j = 0; j < m1.shape1; j++)
                m.data[i][j] = m1.data[i][j] - m2.data[0][j];
    }
    return m;
}

