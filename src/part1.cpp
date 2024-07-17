#include <iostream>
#include <cmath>
using namespace std;

void convolution(float* input, float* kernel, float* output, int inputDim, int kernelDim, int padding) {
    int outputDim = inputDim - kernelDim + 2 * padding + 1;

    for (int i = 0; i < outputDim; i++) {
        for (int j = 0; j < outputDim; j++) {
            output[i*outputDim+j] = 0.0f;

            for (int k = 0; k < kernelDim; k++) {
                for (int l = 0; l < kernelDim; l++) {
                    if ((i + k - padding < 0) || (i + k - padding >= inputDim) || (j + l - padding < 0) || (j + l - padding >= inputDim)) {
                        continue;
                    }
                    output[i*outputDim+j] += input[(i + k - padding)*inputDim+(j + l - padding)] * kernel[k*kernelDim+l];
                }
            }
        }
    }
}

void relu(float* input, float* output, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            output[i*m+j] = max(0.0f, input[i*m+j]);
        }
    }
}

void tanh(float* input, float* output, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            output[i*m+j] = tanhf(input[i*m+j]);
        }
    }
}

void maxPooling(float* input, float* output, int inputDim, int pooling) {
    int outputDim = inputDim-pooling+1;

    for (int i = 0; i < outputDim; i++) {
        for (int j = 0; j < outputDim; j++) {
            float maxVal = -INFINITY;

            for (int k = 0; k < pooling; k++) {
                for (int l = 0; l < pooling; l++) {
                    maxVal = fmax(maxVal, input[(i + k)*inputDim + j + l]);
                }
            }

            output[i*outputDim+j] = maxVal;
        }
    }
}

void avgPooling(float* input, float* output, int inputDim, int pooling) {
    int outputDim = inputDim-pooling+1;

    for (int i = 0; i < outputDim; i++) {
        for (int j = 0; j < outputDim; j++) {
            float sum = 0.0;

            for (int k = 0; k < pooling; k++) {
                for (int l = 0; l < pooling; l++) {
                    sum += input[(i + k)*inputDim + j + l];
                }
            }

            output[i*outputDim+j] = sum / (pooling * pooling);
        }
    }
}

void sigmoid(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

void softmax(float* input,float* output, int size) {
    float sum = 0.0;

    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

int main(int argc, char** argv)
{
    int task = atoi(argv[1]);
    
    switch(task)
    {
        case 1:{
            int n = atoi(argv[2]);
            int m = atoi(argv[3]);
            int p = atoi(argv[4]);

            float* input = new float[n*n];
            float* kernel = new float[m*m];
            float* output = new float[(n-m+2*p+1)*(n-m+2*p+1)];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    input[i*n + j] = atof(argv[5 + i*n + j]);
                }
            }

            for(int i = 0; i < m*m; i++) {
                kernel[i] = atof(argv[5 + n*n + i]);
            }

            convolution(input,kernel,output,n,m,p);
            
            for(int i = 0; i < (n-m+2*p+1); i++) {
                for(int j = 0; j < (n-m+2*p+1); j++) {
                    cout << output[i*(n-m+2*p+1) + j] << " ";
                }
                cout << endl;
            }

            break;
        }
        case 2:{
            int act = atoi(argv[2]);
            int n = atoi(argv[3]);
            int m = atoi(argv[4]);

            float* input = new float[n*m];
            float* output = new float[n*m];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    input[i*m + j] = atof(argv[5 + i*m + j]);
                }
            }

            if(act==0) relu(input,output,n,m);
            if(act==1) tanh(input,output,n,m);

            for(int i = 0; i < n; i++) {
                for(int j = 0; j < m; j++) {
                    cout << output[i*m + j] << " ";
                }
                cout << endl;
            }

            break;
        }
        case 3:{
            int mode = atoi(argv[2]);
            int p = atoi(argv[3]);
            int n = atoi(argv[4]);

            float* input = new float[n*n];
            float* output = new float[(n-p+1)*(n-p+1)];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    input[i*n + j] = atof(argv[5 + i*n + j]);
                }
            }

            if(mode==0) maxPooling(input,output,n,p);
            if(mode==1) avgPooling(input,output,n,p);

            for(int i = 0; i < (n-p+1); i++) {
                for(int j = 0; j < (n-p+1); j++) {
                    cout << output[i*(n-p+1) + j] << " ";
                }
                cout << endl;
            }

            break;
        }
        case 4:{
            int mode = atoi(argv[2]);
            int n = argc-3;

            float* input = new float[n];
            float* output = new float[n];

            for (int i = 0; i < n; i++) {
                input[i] = atof(argv[3 + i]);
            }

            if(mode==0) sigmoid(input,output,n);
            if(mode==1) softmax(input,output,n);

            for(int i = 0; i < n; i++) {
                cout << output[i] << " ";
            }
            cout << endl;

            break;
        }
        default:
            break;
    }
    return 0;
}