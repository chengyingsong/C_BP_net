#include"model.h"
#include"Matrix.h"
#include<string>
#include<fstream>
using namespace std;

extern int layer_count;
extern vector<int> layers;
extern vector<Matrix> W;        //权重矩阵
extern vector<Matrix> Bias;     //单元bias
extern vector<Matrix> Avg_W;    //权重矩阵
extern vector<Matrix> Avg_Bias; //单元bias

void ReLU(Matrix& a)
{
    for (int i = 0; i < a.shape0; i++)
    {
        for (int j = 0; j < a.shape1; j++)
            if (a.data[i][j] < 0)
                a.data[i][j] = 0;
    }
}


double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

void tanh(Matrix& a) {
    for (int i = 0; i < a.shape0; i++)
        for (int j = 0; j < a.shape1; j++)
            a.data[i][j] = 2 * sigmoid(2 * a.data[i][j]) - 1;
}

Matrix tanh_derivative(const Matrix& a) {
    Matrix An(a.shape0, a.shape1);
    for (int i = 0; i < a.shape0; i++)
        for (int j = 0; j < a.shape1; j++)
            An.data[i][j] = (1 - a.data[i][j] * a.data[i][j]);  //之后重构一下，元素点乘，广播减
    return An;

}

Matrix ReLu_derivative(Matrix& a)
{
    Matrix An(a.shape0, a.shape1);
    for (int i = 0; i < a.shape0; i++)
        for (int j = 0; j < a.shape1; j++)
            if (a.data[i][j] > 0)
                An.data[i][j] = 1;
            else
                An.data[i][j] = 0;
    return An;
}

void Softmax(Matrix& a)
{
    //shape (1,10)
    for (int i = 0; i < a.shape0; i++)
    {
        double max_x = 0;
        for (int j = 0; j < a.shape1; j++)
            if (a.data[i][j] > max_x)
                max_x = a.data[i][j];
        for (int j = 0; j < a.shape1; j++)
            a.data[i][j] = exp(a.data[i][j]);  //数值下溢了
        //print(a);
        double sum_x = 0;
        for (int j = 0; j < a.shape1; j++)
            sum_x += a.data[i][j];
        for (int j = 0; j < a.shape1; j++)
            a.data[i][j] /= sum_x;
    }
}

vector<Matrix> forward(Matrix& sample_x)
{
    //sample_x shape (1,784)
    // W[0] (784,512),W[1]
    vector<Matrix> H;
    H.push_back(sample_x);
    for (int i = 1; i < layer_count - 1; i++)
    { //H[1] = x W + B (1,512)
        Matrix h = H[i - 1].dot(W[i - 1]) + Bias[i - 1];
        tanh(h);
        H.push_back(h);
    }
    //现在是(1,512),然后softmax
    Matrix y_hat = H[layer_count - 2].dot(W[layer_count - 2]) + Bias[layer_count - 2];
    //printf("before softmax: ");
    //print(y_hat);
    Softmax(y_hat);
    //printf("after softmax: ");
    //print(y_hat);
    H.push_back(y_hat);
    return H;
}


void backprop(vector<Matrix> H, double y)
{
    //TODO:把每一层的梯度放在Avg里
    vector<Matrix> delta_error(H.size() - 1); //H.size = 3,y_hat,h1,x
    Matrix Y(1, layers[layers.size() - 1]);  //Y.shape 1,10
    Y.data[0][y] = 1;
    int index = layer_count - 2; //1
    delta_error[index] = H[index + 1] - Y; //shape (1,10)
    //printf("Predict y_hat: ");
    //print(H[index+1]); 
    Avg_Bias[index] = delta_error[index]; //shape (1,10)
    Avg_W[index] = H[index].transpose().dot(delta_error[index]); //shape(512,10)
    for (int i = index - 1; i >= 0; i--)
    {
        Matrix h_derivative = tanh_derivative(H[i + 1]).eyes();
        delta_error[i] = delta_error[i + 1].dot(W[i + 1].transpose()).dot(h_derivative);
        Avg_Bias[i] = delta_error[i];
        Avg_W[i] = H[i].transpose().dot(delta_error[i]);
    }
}

void save_model(const char* filename) {
    fstream file(filename, ios::out | ios::trunc); //覆盖写入
    assert(file.is_open() == true);
    //先存入W[0],B[0],W[1],B[0]
    for (int i = 0; i < W.size(); i++)
    {
        //存入W[i]
        W[i].save(file);
        file << endl;
        Bias[i].save(file);
    }
    file.close();
}

void load_model(const char* filename) {
    fstream file(filename, ios::in); //读取文件内容
    assert(file.is_open() == true);
    //先存入W[0],B[0],W[1],B[0]
    for (int i = 0; i < W.size(); i++)
    {
        //存入W[i]
        W[i].load(file);
        Bias[i].load(file);
    }
    file.close();
}
