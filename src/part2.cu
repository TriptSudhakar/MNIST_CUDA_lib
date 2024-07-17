#include <iostream>
#include <cmath>
using namespace std;

__global__ void convolution(float* input, float* kernel, float* output, int inputDim, int kernelDim, int padding) {
    int outputDim = inputDim - kernelDim + 2 * padding + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i < outputDim && j < outputDim)
    {
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

__global__ void relu(float* input, float* output, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < n && j < m) {
        output[i*m+j] = max(0.0f, input[i*m+j]);
    }
}

__global__ void tanh(float* input, float* output, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < n && j < m) {
        output[i*m+j] = tanhf(input[i*m+j]);
    }
}

__global__ void maxPooling(float* input, float* output, int inputDim, int pooling) {
    int outputDim = inputDim-pooling+1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < outputDim && j < outputDim) {
        float maxVal = -INFINITY;
        for (int k = 0; k < pooling; k++) {
            for (int l = 0; l < pooling; l++) {
                maxVal = fmax(maxVal, input[(i + k)*inputDim + j + l]);
            }
        }
        output[i*outputDim + j] = maxVal;
    }
}

__global__ void avgPooling(float* input, float* output, int inputDim, int pooling) {
    int outputDim = inputDim-pooling+1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < outputDim && j < outputDim) {
        float sum = 0.0;
        for (int k = 0; k < pooling; k++) {
            for (int l = 0; l < pooling; l++) {
                sum += input[(i + k)*inputDim + j + l];
            }
        }
        output[i*outputDim + j] = sum / (pooling * pooling);
    }
}

__global__ void sigmoid(float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size){
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

__global__ void exp_sum(float* input, int size, float* sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size){
        *sum = atomicAdd(sum, *sum + exp(input[i]));
    }
}

__global__ void softmax(float* input,float* output, int size, float* sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size){
        output[i] = exp(input[i]) / (*sum);
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

            float* h_input = new float[n*n];
            float* h_kernel = new float[m*m];
            float* h_output = new float[(n-m+2*p+1)*(n-m+2*p+1)];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    h_input[i*n + j] = atof(argv[5 + i*n + j]);
                }
            }

            for(int i = 0; i < m*m; i++) {
                h_kernel[i] = atof(argv[5 + n*n + i]);
            }

            float* d_input;
            float* d_kernel;
            float* d_output;
            cudaMalloc(&d_input, n*n*sizeof(float));
            cudaMalloc(&d_kernel, m*m*sizeof(float));
            cudaMalloc(&d_output, (n-m+2*p+1)*(n-m+2*p+1)*sizeof(float));
            cudaMemcpy(d_input, h_input, n*n*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernel, h_kernel, m*m*sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockDim(16, 16);
            dim3 gridDim((n-m+2*p+blockDim.x)/blockDim.x, (n-m+2*p+blockDim.y)/blockDim.y);
            convolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, n, m, p);
            cudaMemcpy(h_output, d_output, (n-m+2*p+1)*(n-m+2*p+1)*sizeof(float), cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < (n-m+2*p+1); i++) {
                for(int j = 0; j < (n-m+2*p+1); j++) {
                    cout << h_output[i*(n-m+2*p+1) + j] << " ";
                }
                cout << endl;
            }

            break;
        }
        case 2:{
            int act = atoi(argv[2]);
            int n = atoi(argv[3]);
            int m = atoi(argv[4]);

            float* h_input = new float[n*m];
            float* h_output = new float[n*m];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    h_input[i*m + j] = atof(argv[5 + i*m + j]);
                }
            }

            float* d_input;
            float* d_output;
            cudaMalloc(&d_input, n*m*sizeof(float));
            cudaMalloc(&d_output, n*m*sizeof(float));
            cudaMemcpy(d_input, h_input, n*m*sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockDim(16, 16);
            dim3 gridDim((n-m+blockDim.x)/blockDim.x, (n-m+blockDim.y)/blockDim.y);
            if(act==0) relu<<<blockDim, gridDim>>>(d_input, d_output, n, m);
            if(act==1) tanh<<<blockDim, gridDim>>>(d_input, d_output, n, m);
            cudaMemcpy(h_output, d_output, n*m*sizeof(float), cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < m; j++) {
                    cout << h_output[i*m + j] << " ";
                }
                cout << endl;
            }

            break;
        }
        case 3:{
            int mode = atoi(argv[2]);
            int p = atoi(argv[3]);
            int n = atoi(argv[4]);

            float* h_input = new float[n*n];
            float* h_output = new float[(n-p+1)*(n-p+1)];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    h_input[i*n + j] = atof(argv[5 + i*n + j]);
                }
            }

            float* d_input;
            float* d_output;
            cudaMalloc(&d_input, n*n*sizeof(float));
            cudaMalloc(&d_output, (n-p+1)*(n-p+1)*sizeof(float));
            cudaMemcpy(d_input, h_input, n*n*sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockDim(16, 16);
            dim3 gridDim((n/2+blockDim.x-1)/blockDim.x, (n/2+blockDim.y-1)/blockDim.y);
            if(mode==0) maxPooling<<<blockDim, gridDim>>>(d_input, d_output, n, p);
            if(mode==1) avgPooling<<<blockDim, gridDim>>>(d_input, d_output, n, p);
            cudaMemcpy(h_output, d_output, (n-p+1)*(n-p+1)*sizeof(float), cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < (n-p+1); i++) {
                for(int j = 0; j < (n-p+1); j++) {
                    cout << h_output[i*(n-p+1) + j] << " ";
                }
                cout << endl;
            }

            break;
        }
        case 4:{
            int mode = atoi(argv[2]);
            int n = argc-3;

            float* h_input = new float[n];
            float* h_output = new float[n];

            for (int i = 0; i < n; i++) {
                h_input[i] = atof(argv[3 + i]);
            }

            float* d_input;
            float* d_output;
            cudaMalloc(&d_input, n*sizeof(float));
            cudaMalloc(&d_output, n*sizeof(float));
            cudaMemcpy(d_input, h_input, n*sizeof(float), cudaMemcpyHostToDevice);

            if(mode==0) sigmoid<<<32, (n+32-1)/32>>>(d_input, d_output, n);
            if(mode==1)
            {
                float* sum;
                cudaMalloc(&sum, sizeof(float));
                cudaMemset(sum, 0, sizeof(float));
                exp_sum<<<32, (n+32-1)/32>>>(d_input, n, sum);
                softmax<<<32, (n+32-1)/32>>>(d_input, d_output, n, sum);
            }
            cudaMemcpy(h_output, d_output, n*sizeof(float), cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < n; i++) {
                cout << h_output[i] << " ";
            }
            cout << endl;

            break;
        }
        default:
            break;
    }
    return 0;
}