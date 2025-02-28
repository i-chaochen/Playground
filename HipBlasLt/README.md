
# compile
```bash
./compile.sh
```

# run
```bash
# HIPBLASLT_LOG_MASK=5 HIPBLASLT_LOG_LEVEL=5 ./a.out
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_create] handle[out]=0x320ccb10 
Read matrices from binary files is done.
Allocate memory on the device is done.
Copy data from host to device is done.
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matmul_desc_create] matmulDesc[out]=0x7ffc68bf94e8 computeType=COMPUTE_32F scaleType=R_32F 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matmul_desc_set_attribute] matmulDesc=0x31fed860 attr=MATMUL_DESC_TRANSA buf=0x7ffc68bf946c sizeInBytes=4 bufData=0x6f 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matmul_desc_set_attribute] matmulDesc=0x31fed860 attr=MATMUL_DESC_TRANSB buf=0x7ffc68bf946c sizeInBytes=4 bufData=0x70 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_create] matLayout[out]=0x7ffc68bf9530 type=R_16BF rows=48 cols=1024 ld=48 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_create] matLayout[out]=0x7ffc68bf9528 type=R_16BF rows=176 cols=1024 ld=176 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_create] matLayout[out]=0x7ffc68bf9520 type=R_16BF rows=48 cols=176 ld=48 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_create] matLayout[out]=0x7ffc68bf9518 type=R_16BF rows=48 cols=176 ld=48 
Create matrix descriptors is done.
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matmul] A=0x7f667d700000 Adesc=0x2f1fb160 B=0x7f667d718000 Bdesc=0x2f1fb0e0 C=0x7f667d770000 Cdesc=0x3227f460 D=0x7f667d775000 Ddesc=0x3227f4a0 computeDesc=0x31fed860 algo=0 workSpace=0 workSpaceSizeInBytes=0 stream=0 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Trace][rocblaslt_matmul] A=0x7f667d700000 Adesc=[type=R_16BF rows=48 cols=1024 ld=48] B=0x7f667d718000 Bdesc=[type=R_16BF rows=176 cols=1024 ld=176] C=0x7f667d770000 Cdesc=[type=R_16BF rows=48 cols=176 ld=48] D=0x7f667d775000 Ddesc=[type=R_16BF rows=48 cols=176 ld=48] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transA=OP_N transB=OP_T epilogue=EPILOGUE_DEFAULT biasPointer=0x0] workSpace=0 workSpaceSizeInBytes=0 alpha=1 beta=0 stream=0 
Perform matrix multiplication is done.
Copy the result back to the host is done.
GPU synchronization is done.
hipFree() all done.
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_destory] matLayout=0x2f1fb160 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_destory] matLayout=0x2f1fb0e0 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_destory] matLayout=0x3227f460 
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matrix_layout_destory] matLayout=0x3227f4a0 
hipblasLtMatrixLayoutDestroy() is done.
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_matmul_desc_destroy] matmulDesc=0x31fed860 
hipblasLtMatmulDescDestroy() is done.
[2025-02-28 11:24:46][HIPBLASLT][1862601][Api][rocblaslt_destroy] handle=0x320ccb10 
hipblasLtDestroy() is done.
Printing matrix at 0x32170690
Matrix (48 x 1024):
    0.0000    0.0000   -0.0000 ...
    0.0000   -0.0000    0.0000 ...
   -0.0000   -0.0000    0.0000 ...
  ...

Printed h_A
Printing matrix at 0x322abfa0
Matrix (176 x 1024):
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
  ...

Printed h_B
Printing matrix at 0x2f2ca4e0
Matrix (48 x 176):
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
  ...

Printed h_C
Printing matrix at 0x32268e80
Matrix (48 x 176):
       nan       nan       nan ...
       nan       nan       nan ...
       nan       nan       nan ...
  ...

Printed h_D
Clean up h_D are done.
Matrix multiplication completed successfully.
```

```bash
# HIPBLASLT_LOG_MASK=32 ./a.out
Read matrices from binary files is done.
Allocate memory on the device is done.
Copy data from host to device is done.
Create matrix descriptors is done.
hipblaslt-bench --api_method c -m 48 -n 176 -k 1024 --lda 48 --ldb 176 --ldc 48 --ldd 48  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 --beta 0.000000 --transA N --transB T --batch_count 1 --scaleA 0 --scaleB 0  --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method index --solution_index 218160 --activation_type none 
Perform matrix multiplication is done.
Copy the result back to the host is done.
GPU synchronization is done.
hipFree() all done.
hipblasLtMatrixLayoutDestroy() is done.
hipblasLtMatmulDescDestroy() is done.
hipblasLtDestroy() is done.
Printing matrix at 0x29799d60
Matrix (48 x 1024):
    0.0000    0.0000   -0.0000 ...
    0.0000   -0.0000    0.0000 ...
   -0.0000   -0.0000    0.0000 ...
  ...

Printed h_A
Printing matrix at 0x2982d530
Matrix (176 x 1024):
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
  ...

Printed h_B
Printing matrix at 0x296f2b80
Matrix (48 x 176):
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
    0.0000    0.0000    0.0000 ...
  ...

Printed h_C
Printing matrix at 0x294059a0
Matrix (48 x 176):
       nan       nan       nan ...
       nan       nan       nan ...
       nan       nan       nan ...
  ...

Printed h_D
Clean up h_D are done.
Matrix multiplication completed successfully.
```