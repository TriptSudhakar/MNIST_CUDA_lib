#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <set>
#include <cuda_runtime.h>
#include <dirent.h>
#include <chrono>
using namespace std;

float* conv1Parameters;
float* conv2Parameters;
float* fc1Parameters;
float* fc2Parameters;

float* conv1Weights;
float* conv2Weights;
float* fc1Weights;
float* fc2Weights;
float* conv1Biases;
float* conv2Biases;
float* fc1Biases;
float* fc2Biases;

float* conv1Output;
float* pool1Output;
float* conv2Output;
float* pool2Output;
float* fc1Output;
float* fc2Output;

__global__ void convolution(const float* input, const float* kernel, const float*biases, float* output,
                                                int inputRows, int inputCols, int inputChannels, int kernelRows, int kernelCols,
                                                int outputRows, int outputCols, int outputChannels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < outputRows && j < outputCols && c < outputChannels) {
        float val = 0.0f;
        for(int m = 0; m < inputChannels; m++){
            for (int k = 0; k < kernelRows; k++) {
                for (int l = 0; l < kernelCols; l++) {
                    val += (input[m * inputCols * inputCols + (i + k) * inputCols + (j + l)] * 
                            kernel[c * inputChannels * kernelRows * kernelCols + m * kernelRows * kernelCols + k * kernelCols + l]);
                }
            }
        }

        output[c * outputRows * outputCols + i * outputCols + j] = val + biases[c];
    }
}

__global__ void maxPooling(const float* input, float* output,
                                 int rows, int cols, int stride,
                                 int outputRows, int outputCols, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < outputRows && j < outputCols && c < channels) {
        int startRow = i * stride;
        int startCol = j * stride;

        float maxVal = input[c * rows * cols + startRow * cols + startCol];
        for (int k = 0; k < stride; k++) {
            for (int l = 0; l < stride; l++) {
                int curRow = startRow + k;
                int curCol = startCol + l;
                if (curRow < rows && curCol < cols) {
                    maxVal = fmaxf(maxVal, input[c * rows * cols + curRow * cols + curCol]);
                }
            }
        }
        output[c * outputRows * outputCols + i * outputCols + j] = maxVal;
    }
}

__global__ void relu(float* input,int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (input[tid] < 0) {
            input[tid] = 0;
        }
    }
}

__global__ void softmax(const float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < size) {
        float sum = 0.0;
        for(int i = 0; i < size; i++) {
            sum += exp(input[i]);
        }

        output[tid] = exp(input[tid]) / sum; 
    }
}

void mnistClassifier(const float *input, float* output) 
{
    // Convolution layer 1
    dim3 blockDim1(16, 16, 1);  
    dim3 gridDim1(2, 2, 32);
    convolution<<<gridDim1, blockDim1>>>(input, conv1Weights, conv1Biases, conv1Output, 28, 28, 1, 5, 5, 24, 24, 20);

    // Pooling layer 1
    dim3 blockDim2(16, 16, 1);  
    dim3 gridDim2(1, 1, 32);
    maxPooling<<<gridDim2, blockDim2>>>(conv1Output, pool1Output, 24, 24, 2, 12, 12, 20);

    // Convolution layer 2
    dim3 blockDim3(16, 16, 1);  
    dim3 gridDim3(1, 1, 64);
    convolution<<<gridDim3, blockDim3>>>(pool1Output, conv2Weights, conv2Biases, conv2Output, 12, 12, 20, 5, 5, 8, 8, 50);

    // Pooling layer 2
    dim3 blockDim4(16, 16, 1);  
    dim3 gridDim4(1, 1, 64);
    maxPooling<<<gridDim4, blockDim4>>>(conv2Output, pool2Output, 8, 8, 2, 4, 4, 50);

    // Fully Connected layer 1
    dim3 blockDim5(16, 16, 1);  
    dim3 gridDim5(1, 1, 512);
    convolution<<<gridDim5, blockDim5>>>(pool2Output, fc1Weights, fc1Biases, fc1Output, 4, 4, 50, 4, 4, 1, 1, 500);

    // ReLU activation for fully connected layer 1
    relu<<<256, 2>>>(fc1Output, 500);

    // Fully Connected layer 2
    dim3 blockDim6(16, 16, 1);  
    dim3 gridDim6(1, 1, 16);
    convolution<<<gridDim6, blockDim6>>>(fc1Output, fc2Weights, fc2Biases, fc2Output, 1, 1, 500, 1, 1, 1, 1, 10);

    // Softmax activation for final layer
    softmax<<<32, 1>>>(fc2Output, output,10);
}

// Function to load weights and biases from file
vector<float> loadParameters(const string& filename) {
    vector<float> params;
    ifstream file(filename);
    float value;
    while (file >> value) {
        params.push_back(value);
    }
    file.close();
    return params;
}

int main() {
    // Load parameters from files
    vector<float> conv1Params = loadParameters("weights/conv1.txt");
    vector<float> conv2Params = loadParameters("weights/conv2.txt");
    vector<float> fc1Params = loadParameters("weights/fc1.txt");
    vector<float> fc2Params = loadParameters("weights/fc2.txt");

    conv1Parameters = (float*)malloc(conv1Params.size() * sizeof(float));
    conv2Parameters = (float*)malloc(conv2Params.size() * sizeof(float));
    fc1Parameters = (float*)malloc(fc1Params.size() * sizeof(float));
    fc2Parameters = (float*)malloc(fc2Params.size() * sizeof(float));

    for (int i = 0; i < conv1Params.size(); ++i) {
        conv1Parameters[i] = conv1Params[i];
    }
    for (int i = 0; i < conv2Params.size(); ++i) {
        conv2Parameters[i] = conv2Params[i];
    }
    for (int i = 0; i < fc1Params.size(); ++i) {
        fc1Parameters[i] = fc1Params[i];
    }
    for (int i = 0; i < fc2Params.size(); ++i) {
        fc2Parameters[i] = fc2Params[i];
    }

    // Extract biases
    int conv1BiasSize = 20; // Bias size for conv1 layer
    int conv2BiasSize = 50; // Bias size for conv2 layer
    int fc1BiasSize = 500;  // Bias size for fc1 layer
    int fc2BiasSize = 10;   // Bias size for fc2 layer
    
    cudaMalloc(&conv1Biases, conv1BiasSize * sizeof(float));
    cudaMalloc(&conv2Biases, conv2BiasSize * sizeof(float));
    cudaMalloc(&fc1Biases, fc1BiasSize * sizeof(float));
    cudaMalloc(&fc2Biases, fc2BiasSize * sizeof(float));
    cudaMalloc(&conv1Weights, (conv1Params.size()-conv1BiasSize) * sizeof(float));
    cudaMalloc(&conv2Weights, (conv2Params.size()-conv2BiasSize) * sizeof(float));
    cudaMalloc(&fc1Weights, (fc1Params.size()-fc1BiasSize) * sizeof(float));
    cudaMalloc(&fc2Weights, (fc2Params.size()-fc2BiasSize) * sizeof(float));

    cudaMemcpy(conv1Biases, conv1Parameters + conv1Params.size() - conv1BiasSize, conv1BiasSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv2Biases, conv2Parameters + conv2Params.size() - conv2BiasSize, conv2BiasSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc1Biases, fc1Parameters + fc1Params.size() - fc1BiasSize, fc1BiasSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc2Biases, fc2Parameters + fc2Params.size() - fc2BiasSize, fc2BiasSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv1Weights, conv1Parameters, (conv1Params.size()-conv1BiasSize) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv2Weights, conv2Parameters, (conv2Params.size()-conv2BiasSize) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc1Weights, fc1Parameters, (fc1Params.size()-fc1BiasSize) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc2Weights, fc2Parameters, (fc2Params.size()-fc2BiasSize) * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&conv1Output, 20 * 24 * 24 * sizeof(float));
    cudaMalloc(&pool1Output, 20 * 12 * 12 * sizeof(float));
    cudaMalloc(&conv2Output, 50 * 8 * 8 * sizeof(float));
    cudaMalloc(&pool2Output, 50 * 4 * 4 * sizeof(float));
    cudaMalloc(&fc1Output, 500 * sizeof(float));
    cudaMalloc(&fc2Output, 10 * sizeof(float));

    int inputRows = 28;
    int inputCols = 28;

    int fileCount = 0;
    const string directory = "pre-proc-img";
    DIR* dir = opendir(directory.c_str());
    if (dir != nullptr) {
        dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_REG) { // Check if it's a regular file
                fileCount++;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory: " << directory << std::endl;
    }

    vector<string> filenames(fileCount);
    float* input;
    cudaMalloc(&input, fileCount * inputRows * inputCols * sizeof(float));

    float* output;
    cudaMalloc(&output, fileCount * 10 * sizeof(float));

    float* h_input = (float*) malloc(fileCount * inputRows * inputCols * sizeof(float));
    int currFile = 0;

    auto start_time = chrono::high_resolution_clock::now();
    dir = opendir(directory.c_str());
    if (dir != nullptr) {
        dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_REG) { // Check if it's a regular file
                filenames[currFile] = entry->d_name;
                string path = directory + "/" + entry->d_name;

                ifstream img(path);
                for (int j = 0; j < inputRows; ++j) {
                    for (int k = 0; k < inputCols; ++k) {
                        img >> h_input[currFile * inputRows * inputCols + j * inputCols + k];
                    }
                }
                img.close();

                currFile++;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory: " << directory << std::endl;
    }

    cudaMemcpy(input, h_input, fileCount * inputRows * inputCols * sizeof(float), cudaMemcpyHostToDevice);
    for(int i=0;i<fileCount;++i) {
        mnistClassifier(input + i * inputRows * inputCols, output + i * 10);
    }

    float* h_output = (float*) malloc(fileCount * 10 * sizeof(float));
    cudaMemcpy(h_output, output, fileCount * 10 * sizeof(float), cudaMemcpyDeviceToHost);

    set<pair<float, int>> s;
    for(int i=0;i<fileCount;++i) {
        string path = "output/" + filenames[i];
        ofstream img(path);

        for(int j=0;j<10;++j) {
            s.insert({h_output[i*10+j], j});
        }

        int cnt = 0;
        for(auto it=s.rbegin();it!=s.rend();++it) {
            if(cnt++ == 5) break;
            img<<(it->first)*100.0<<" class "<<it->second<<endl;
        }
        img.close();

        s.clear();
    }
    auto end_time = chrono::high_resolution_clock::now();
    cout<<"Time taken: "<<chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count()<<" ns"<<endl;
    return 0;
}
