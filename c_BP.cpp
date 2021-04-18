#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <assert.h>
#include "load_data.h"
#include "Matrix.h"
#include "model.h"

using namespace std;

extern vector<int> layers;
extern int layer_count;
extern double eta;          //学习率
extern int batch_size;
extern int epoch_size;

//三层全连接网络
extern vector<Matrix> W;        //权重矩阵
extern vector<Matrix> Bias;     //单元bias
extern vector<Matrix> Avg_W;    //权重矩阵
extern vector<Matrix> Avg_Bias; //单元bias
extern vector<double> epoch_acc;

Matrix X_train;
Matrix Y_train;
Matrix X_test;
Matrix Y_test;


double test() {
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
    return acc;
}


void train() {
    int batch_number = X_train.shape0 / batch_size;
    char buf[60];
    vector<int> index(X_train.shape0);
    for (int i = 0; i < index.size(); i++)
        index[i] = i;
    for (int i = 1; i <= epoch_size; i++)
    {
        printf("Running Epoch %d\n", i);
        //shuffle等
        random_shuffle(index.begin(), index.end());
        double best_acc = 0;
        for (int j = 0; j < batch_number; j++)
        {
            //printf("Running batch number %d in epoch %d\n", j, i);
            double batch_acc = 0;
            for (int k = 0; k < batch_size; k++)
            {
                //随机抽取样本训练
                int shuffle_index = index[j * batch_size + k];
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
            printf("Running batch number %d in epoch %d,acc:%.4lf\n", j, i, batch_acc / batch_size);
            snprintf(buf, 60, "weights\\epoch_%d_batch_num_%d_acc_%.4lf.out", j, i, batch_acc / batch_size);
            save_model(buf);
        }
        printf("Running feedforward on validation data for epoch %d\n", i);
        double acc = test();
        printf("Accuracy on Validation Set for epoch %d is %lf\n", i, acc);
        if (acc > best_acc) {
            best_acc = acc;
            snprintf(buf, 60, "weights\\acc_%.4lf.out", acc);
            save_model(buf);
        }
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
    //读入数据，把图片展开成784维的向量
    unordered_map<string, Matrix> Map = load_data();
    X_train = Map["train_images"];
    Y_train = Map["train_labels"];
    X_test = Map["test_images"];
    Y_test = Map["test_labels"];

    

    //shuffle
    load_model("weights\\epoch_11_batch_num_1_acc_0.9392.out");

    train();
    double acc = test();
    printf("final acc:%lf\n", acc);
    return 0;
}
