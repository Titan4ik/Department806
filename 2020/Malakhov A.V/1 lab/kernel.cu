#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <ctime>


#define BLOCK_DIM 4 //������ ����������+
#define MATRIX_COUNT 1000000
int M = 32, K = 32;

using namespace std;

__global__ void matrixAdd(int* A, int* B, int M, int K) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = col * M + row;

	if (col < M && row < K) {
		B[index] += A[index];
	}
}


void cpu_test(vector<int*> matrixs) {
	int* C = new int[M * K];
	int size = M * K;
	for (int i = 0; i < M * K; i++)
		C[i] = 0;
	for (int i = 0; i < MATRIX_COUNT; i++) {
		for (int j = 0; j < size; j++) {
			C[j] += matrixs[i][j];
		}
	}

	printf("Result Matrix C:\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < K; j++) {
			printf("%d\t", C[i]);
		}
		printf("\n");
	}
	free(C);
}
void gpu_test(vector<int*> matrixs) {

	int* C = new int[M * K];


	
	for (int i = 0; i < M * K; i++)
		C[i] = 0;



	vector<int*> cuda_matrixs;
	cuda_matrixs.resize(MATRIX_COUNT);
	int* cuda_matrix_result;

	int size = M * K * sizeof(int); 

	for (int i = 0; i < MATRIX_COUNT; i++) {
		cudaMalloc((void**)&cuda_matrixs[i], size); //��������� ������
		cudaMemcpy(cuda_matrixs[i], matrixs[i], size, cudaMemcpyHostToDevice);
	}

	cudaMalloc((void**)&cuda_matrix_result, size);



	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); //����� ���������� ������
	dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (K + dimBlock.y - 1) / dimBlock.y); //������ � ����������� �����

	for (int i = 0; i < MATRIX_COUNT; i++) {
		matrixAdd << <dimGrid, dimBlock >> > (cuda_matrixs[i], cuda_matrix_result, M, K); //����� ����
	}

	cudaDeviceSynchronize(); //�����������

	cudaMemcpy(C, cuda_matrix_result, size, cudaMemcpyDeviceToHost);

	//�����    ����������
	printf("Result Matrix C:\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < K; j++) {
			printf("%d\t", C[i]);
		}
		printf("\n");
	}

	for (int i = 0; i < MATRIX_COUNT; i++) {
		cudaFree(cuda_matrixs[i]);
	}
	cudaFree(cuda_matrix_result);
	free(C);
}

int main() {
	srand(static_cast<unsigned int>(time(0)));
	vector<int*> matrixs;
	for (int i = 0; i < MATRIX_COUNT; i++) {
		printf("%d\n", i);
		matrixs.push_back(new int[M * K]);
		for (int j = 0; j < M * K; j++)
			matrixs[i][j] = rand() % 100 - 50;
	}

	unsigned int start_time1 = clock(); // ��������� �����
	gpu_test(matrixs);
	unsigned int end_time1 = clock(); // �������� �����
	unsigned int search_time1 = end_time1 - start_time1; // ������� �����
	printf("GPU TIME = %d\n", search_time1);
	unsigned int start_time2 = clock(); // ��������� �����
	cpu_test(matrixs);
	unsigned int end_time2 = clock(); // �������� �����
	unsigned int search_time2 = end_time2 - start_time2; // ������� �����
	printf("CPU TIME = %d\n", search_time2);

	for (int i = 0; i < MATRIX_COUNT; i++) {
		free(matrixs[i]);
	}
	system("pause");
}