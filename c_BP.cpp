#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <assert.h>
#include "load_data.h"
#include "Matrix.h"
using namespace std;

//三层全连接网络
vector<int> layers = {784, 512, 10};
int layer_count = layers.size();
double eta = 0.001;          //学习率
vector<Matrix> W;        //权重矩阵
vector<Matrix> Bias;     //单元bias
vector<Matrix> Avg_W(layer_count-1);    //权重矩阵
vector<Matrix> Avg_Bias(layer_count-1); //单元bias
vector<double> epoch_acc;

void print(const Matrix& a) {
    for (int i = 0; i < a.shape0; i++)
    {
        for (int j = 0; j < a.shape1; j++)
            printf("%.2lf\t", a.data[i][j]);
        printf("\n");
    }
}


void ReLU(Matrix &a)
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
            a.data[i][j] = 2 *sigmoid(2*a.data[i][j])-1;
}

Matrix tanh_derivative(const Matrix &a) {
    Matrix An(a.shape0,a.shape1);
    for (int i = 0; i < a.shape0; i++)
        for (int j = 0; j < a.shape1; j++)
            An.data[i][j] =(1 - a.data[i][j]*a.data[i][j]);  //之后重构一下，元素点乘，广播减
    return An; 

}

void Softmax(Matrix &a)
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

vector<Matrix> forward(Matrix &sample_x)
{
    //sample_x shape (1,784)
    // W[0] (784,512),W[1]
    vector<Matrix> H;
    H.push_back(sample_x);
    for (int i = 1; i < layer_count - 1; i++)
    { //H[1] = x W + B (1,512)
        Matrix h = H[i - 1].dot(W[i - 1]) + Bias[i - 1];
        tanh(h);
        //printf("inner vector:");
        //print(h);
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

Matrix ReLu_derivative(Matrix &a)
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

void backprop(vector<Matrix> H, double y)
{
    //TODO:把每一层的梯度放在Avg里
    vector<Matrix> delta_error(H.size()-1); //H.size = 3,y_hat,h1,x
    Matrix Y(1, layers[layers.size() - 1]);  //Y.shape 1,10
    Y.data[0][y] = 1;
    int index = layer_count - 2; //1
    delta_error[index] = H[index+1] - Y; //shape (1,10)
    //printf("Predict y_hat: ");
    //print(H[index+1]); 
    Avg_Bias[index] =  delta_error[index]; //shape (1,10)
    Avg_W[index] =  H[index].transpose().dot(delta_error[index]); //shape(512,10)
    for (int i = index -1; i >= 0; i--)
    {
        Matrix h_derivative = tanh_derivative(H[i+1]).eyes();
        delta_error[i] = delta_error[i + 1].dot(W[i+1].transpose()).dot(h_derivative);
        Avg_Bias[i] =  delta_error[i];
        Avg_W[i] = H[i].transpose().dot(delta_error[i]);
    }
}


int main()
{
    printf("Start initial!!!\n");
    for (int i = 0; i < layers.size() - 1; i++)
    {
        //初始化参数矩阵和偏置矩阵
        //printf("i:%d\n", i);
        Matrix w(layers[i], layers[i + 1]);
        Matrix b(1, layers[i + 1]);
        double std = sqrt(2.0 / layers[i + 1]);
        w.init(std); //初始化
        b.init(std);
        W.push_back(w);
        Bias.push_back(b);
    }
    printf("Sucessful!!!\n");
    int batch_size = 50;
    int epoch_size = 10;
    //读入数据，把图片展开成784维的向量
    unordered_map<string, Matrix> Map = load_data();
    Matrix X_train = Map["train_images"];
    Matrix Y_train = Map["train_labels"];
    Matrix X_test = Map["test_images"];
    Matrix Y_test = Map["test_labels"];
    int batch_number = X_train.shape0 / batch_size;

    

    //shuffle
    for (int i = 1; i <= epoch_size; i++)
    {
        printf("Running Epoch %d\n", i);
        //TODO:shuffle等下处理
        for (int j = 0; j < batch_number; j++)
        {
            //printf("Running batch number %d in epoch %d\n", j, i);
            double batch_acc = 0;
            for (int k = 0; k < batch_size; k++)
            {
                int shuffle_index = j * batch_size + k;
                Matrix sample_x(1, layers[0]);
                sample_x.assign(X_train.data[shuffle_index]);
                double sample_y = Y_train.data[shuffle_index][0];
                vector<Matrix> Predict_y = forward(sample_x);
                double y = Predict_y[layer_count - 1].argmax(0)[0];
                //print(Predict_y[2]);
                if (y == sample_y)
                    batch_acc++;
                //printf("pred:%.0lf,real:%.0lf\n", y, sample_y);
                backprop(Predict_y, sample_y);
                for (int k = 0; k < W.size(); k++)
                {
                    double lambda = eta;
                    W[k] = W[k] - lambda * Avg_W[k];
                    Bias[k] = Bias[k] - lambda * Avg_Bias[k];
                }
            }
            //printf("Running batch number %d in epoch %d,acc:%.4lf\n", j, i, batch_acc / batch_size);
        }

        printf("Running feedforward on validation data for epoch %d\n", i);
        double acc = 0;
        for (int k = 0; k < X_test.shape0; k++)
        {
            Matrix sample_x(1, layers[0]);
            sample_x.assign(X_test.data[k]);
            int predict_y = forward(sample_x)[layer_count - 1].argmax(0)[0];
            int y = int(Y_test.data[k][0]);
            if (y == predict_y)
                acc++;
        }
        acc /= X_test.shape0;
        epoch_acc.push_back(acc);
        printf("Accuracy on Validation Set for epoch %d is %lf\n", epoch_size, acc);
    }
    return 0;
}
