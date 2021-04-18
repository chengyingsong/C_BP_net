//#include "load_data.h"
#include "..\C_BP_net\load_data.h"

int ReverseInt(int i)
{
    //大小端转换
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, Matrix &labels,int len)
{
    //读取label
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        //cout << "magic number = " << magic_number << endl;
        //cout << "number of images = " << number_of_images << endl;
        assert(len <= number_of_images);
        for (int i = 0; i < len; i++)
        {
            unsigned char label = 0;
            file.read((char *)&label, sizeof(label));
            labels.data[i][0] = (double)label;
        }
    }
    file.close();
}

void read_Mnist_Images(string filename, Matrix &images,int len)
{
    //读取数据集
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&number_of_images, sizeof(number_of_images));
        file.read((char *)&n_rows, sizeof(n_rows));
        file.read((char *)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);
        assert(len <= number_of_images);
       // cout << "magic number = " << magic_number << endl;
       // cout << "number of images = " << number_of_images << endl;
        //cout << "rows = " << n_rows << endl;
        //cout << "cols = " << n_cols << endl;

        for (int i = 0; i < len; i++)
        {
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char *)&image, sizeof(image));
                    if (image  > 0 )
                        images.data[i][c + r * 28] = 1;
                    else images.data[i][c + r * 28] = 0;  //二值化
                    //printf("%.0lf ",images.data[i][c + r*28]);
                }
               // printf("\n");
            }
        }
    }
    else { printf("file not open!!\n"); }
    file.close();
}

unordered_map<string, Matrix> load_data()
{
    extern int train_len;
    extern int test_len;
    Matrix test_labels(test_len, 1);
    Matrix train_labels(train_len, 1);
    Matrix test_images(test_len, 784);
    Matrix train_images(train_len, 784);
    read_Mnist_Label("data\\train-labels.idx1-ubyte", train_labels,train_len);
    read_Mnist_Images("data\\train-images.idx3-ubyte", train_images,train_len);
    read_Mnist_Label("data\\t10k-labels.idx1-ubyte", test_labels,test_len);
    read_Mnist_Images("data\\t10k-images.idx3-ubyte", test_images,test_len);
    unordered_map<string, Matrix> m;
    m.insert({"train_images", train_images});
    m.insert({"test_images", test_images});
    m.insert({"train_labels", train_labels});
    m.insert({"test_labels", test_labels});
    return m;
}