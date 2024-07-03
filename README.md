# Dragon-Alpha-v1.2-source [To be continued]
> The source code of Dragon-Alpha-v1.2.

# I. About Cu32
**1.** __What__ __is__ __Cu32__ ? __Cu32__ is a GPU library for FP32 tensor computing. __Cu32__ is designed for training, rather than predicting. It consists of 14 libs:
> - [1]  _Cuda_: It serves as the bridge between JVM (CPU) and GPU. Its functions include data-copy, data-transfer, tensor-initialization, memory-management, and computing-scheduling. It provides Java APIs to manage Cuda memory, event, stream, and device.
> - [2]  _CudaDevice_: It provides functions to manage the information of Cuda Device, such as id, name, SM number, and L2 cache size. The device information are encapsulated into an _CudaDevice_ Java Object.
> - [3]  _Cuda_random_: It uses dozens of threads to generate pseudo random numbers following a specific distribution, such as Gaussian, Uniform, Bernouli, etc. 
> - [4]  _Cuda_image_: It includes functions for _int8_ image processing, like affine-transformation, color-jitter, transpose, and resize. These functions are not limited to 3-channel BGR images, and can also procss Hyper Spectrul Images (HSIs) with many channels.
> - [5]  _Cuda_math_: It includes element-wise functions, like Linear, Quadratic, Sigmoid, Relu, Softmax, Gelu, and BatchNorm. 
> - [6]  _Cuda_expk2_: It serves as an extension library of _Cuda_math_, and involves functions like transpose, split, concat, and padding.
> - [7]  _Cuda_pool2D_: It implements the forward propagation of Average and Max 2D-Pooling layers.
> - [8]  _Cuda_upool2D_: It implements the backward propagation of Average and Max-2D-Pooling layers.
> - [9]  _Cuda_conv3D_: It implements the forward convolution (FConv) of 2D convolutional layers. 
> - [10] _Cuda_dconv3D_deltaX_: It implements the backward-data convolution (BDConv) of 2D convolutional layers. BDConv is used to find the gradient of input feature maps.
> - [11] _Cuda_dconv3D_deltaW_: It implements the backward-filter convolution (BFConv) of 2D convolutional layers. BFConv is used to find the gradient of filters. The source has not been uploade, since this lib contains some algorithms we haven't publicly disclosed.
> - [12] _Cuda_reduce_: It includes some reduction operators, and can be used to find mean, variance, maximum, minimum, etc. To enhance flexibility, these reduction operators are fused with element-wise linear transformations ($Y = \alpha X + \beta$).
> - [13] _Cuda_matMul_: contains 3 types of matrix multiplications, $A * B$, $A * B^T$, and $A^T * B$.
> - [14] _Cuda_batchMatMul_: contains 3 types of batch matrix multiplications, $A * B$, $A * B^T$, and $A^T * B$.
>> These 14 libs have corresponding dlls and VS2017-studio projects. Due to my limited personal ability, I don't have enough time to write comments and optimize all kernel functions. Instead, I pay attention to optimize the kernels with the highest performance up-limit. Specifically, matrix multiplication have good performance when dimensions are multiples of $64$; convolutional operators perform well with $64\times$ channel sizes. I foucused on enhancing the readbility of source code, I believe smart you can understand them without too much comments. To better understand __Cu32__, I suggest you to read the java code of _CudaFloat32EngineBase_, which contains some higher-level operatioal logics of __Cu32__.
 
**2.** __The__ __complication__ __of__ __Cu32__. Under the Apache-2.0 License, you can modify and recompile the source of ___Cu32___. Now, __Cu32__ is only compiled for 64-bit Windows, so kindly recompile it using NVCC for other Operating Systems, like Centos, and Ubuntu.  I recommand nvcc-11.5 compilter for RTX-30X0 (Ampere) GPU, and nvcc-11.8 for RTX-40X0 (Lovelace) GPU, however, I suggest you to try different nvcc versions on your platforms. Except that _Cuda_dconv3D_deltaX_ requires $compute$ and $sm \ge 60$, the other libs only requires $compute$ and $sm \ge 52$. According to my experience, $compute = sm = 52$ or $70$ can usually brings good performance, besides, I encourage you to try different compile conifgurations on your hardware to select the best.

**3.**. __Convolution__ __algorithms__ __in__ __Cu32__.

There are multiple convolution algorithms in ___Cu32___


There are 3 kinds of convolutions in convolutional layers: forward convolution (FConv), backward-data convolution (BDConv), and backward-filrer(BFConv) convolution. The FConv generates the output-feature-maps in forward propagation, 

The convolution algorithms in  ___Cu32___ are listed as follow
  

> - [1] _GEMM_: This algorithm is a variant of direct convolution, which transforms a convolution into a matrix multiplication. _GEMM_ supports both
> - _forward_ and _backward_ propagation. 
> - [2] _GEMMR_: It's an variant of _GEMM_, which transposes the filters from $O_C \times F_H \times F_W \times I_C$ to $F_H \times F_W \times I_C \times O_C$ format enhance bandwidth.
> - [3] _GEMMSK_: It's an variant of _GEMM_, which splits the accumulation tasks along GK axis to enhance paralellism.
> - [4] _GEMMV2_, _GEMMV2R_, _GEMMSKR_: They variants of _GEMM_, _GEMMR_, and _GEMMSK_. They adopt the filter-trimming technique to exclude the padded zeros, and can reduce time complexity especially when dealing with small feature maps.
> - [5] _Im2col-Winograd_: It has been implemented for both forward and backward propagation. It supports unit stride and filters $<= 9*9$
> - [6] _Winograd2D_: It is only used for evaluation, and has not been integrated to Dragon-Alpha.
> - [7] _Kernel-Split_: It is used to find the gradient of input feature maps when $stride > 1$. I have specifically optimized this algorithm for cases with $stride = 2$.
> - [8] _Kernel-SplitV2_: It's an variant of _Kenrel-Split_, with the integration of filter-trimming.
> - [9] _Cross-Add_: It can find the gradient of input feature maps in backward propagation. It is only used when channel is very small. I have not fully optimize this algorithm.

# II. About Alpha

Please make sure: the JDK version is greater than 8.0<br>
**3.** To complie the CUDA-C++ source code of cu32, make sure:  compute >= 52, sm >= 52 <br>
**4.** Kindly read “Arxiv.pdf” first, to briefly understand Alpha.<br>
**5.** Alpha has only been executed on GTX 1050, RTX 3060ti GPU, and presently its applications can only be executed on CUDA GPU.<br>
**6.** Since I am the only-one programmer to build Alpha, I must pay my main attention to the code instead of the document, to complete Alpha’s prototype in time. If you have some questions, just see the source-code. Sorry, my personal abilities are really limited.<br> 

# II. Files
- **Arxiv.pdf**  an article talking about the background, characteristics, architecture and experiments of Alpha, preprinted on arxiv.org, at: https://arxiv.org/abs/2305.08819.<br>
- **exec**  the executable files of Alpha.
  - **lib**  Java libraries of Alpha, which are jar-files complied by Java-code. Obviously, you need to add such jar files to your projects.<br>
  - **native-lib**  native libraries of Dragon-Alpha. They are dynamic-linked-libraries, and integrated to Alpha at the bottom through JNI.<br>
    - **cuda_float32**  contains the executable files of cu32. Presently, cu32 has only been complied for 64-bit Windows, and will be compiled for Linux in the near future<br>
  - **icon**  Alpha’s logo. If you like it, set it for Alpha’s home directory. <br> 
  - **data**  Alphas’ built-in data-sets, including MINIST, cifar-10, cifar-100 and Soccer. Please decompress before use them.<br> 
  - **src**  the source-code of Dragon-Alpha<br>
  - **alpha_src**  the Java source-code of Alpha. You rename this directory to ‘src’ and integrate it to your own Java-project. I suggest using NetBeans to read such source-code, since I use NetBeans to build Alpha.<br>
  - **zutil_src**  ZUtil is an auxiliary library for Alpha. Since I wrote it in my sophomore year, it may have some unreasonable aspects in programming-style and architecture. I only uploaded a part of it, so kindly use ZUTIL-STD-1.1.jar instead of the source-code.<br>
  - **cu32_src**  the C++ source-code of cu32, consists of 13 Visual Studio (VS2017) projects. To open such projects on your PC, please make sure your VS can build CUDA projects. You also need to add jdk.include&jdk.lib to such projects (such as jni.h, jvm.lib). Since I use CUDA v11.3, I suggest you to use such version too.<br>
- **experiments**  the related experimental code&data related to Arxiv.pdf.<br>
  - **alpha-code**  the experimental code of Alpha, you can take it as examples, to create your own Alpha-app. Before using Alpha’s API, you must specify the home-path of Alpha, in order to load the relative native libraries.<br>
  ![image](https://github.com/GilgameshXYZ123/Dragon-Alpha/assets/65615049/2586a7d0-0226-4bae-a575-5d9e2c8bdf66)
  - **pytorch-code**  the experimental code of PyTorch.<br>
  - **experiment-data**  console output to track some metrics for both Alpha and PyTorch, in order to make a comparison.<br>
  - **test_cuda**  some related code to test Alpha&cu32. You can take it as examples of using Alpha’s operators.<br>
  
# About me
- My name Zhang Zhiyi, and I often use Gilgamesh as my internet name.<br>
- I was born in April, 2000, majored Computer-Science in my college, and now study Pattern-Recognition after graduate.<br>
- First, let’s talk about the reason why I create Alpha instead of using PyTorch. I prefer Java to Python, but failed to find a Java-based DL framework as excellent as PyTorch, in the past few years. Also, I want to learn more about the principles and details of DL, and like implementing them to improve my abilities. So, I started to build my own Java DL framework. Dragon-Alpha can be regraded as a continuation of Dragon, which is my graduation project.<br>
- It took me over 1 year to build Alpha-v1.2. In such progress, I have been suffering while enjoying, and finally benefited a lot. PyTorch&cuDNN is my opponent but also my mentor. I tried to learn its advantage, and pondered how to make some breakthrough while keeping Alpha’s own characteristics. Now, the Alpha’s prototype has been completed, and the relative paper has been written.<br>
At present, Alpha is not as polished as PyTorch, but it could be good start and have a long way to reach perfection. I am grateful to all those who provided me with support and encouragement.<br>
- It’s my honour to share the source code Alpha&cu32. Sincerely, I request and need all of you to use and improve it. If you have some related good advice and achievement, please contact me at gilgamesh@mail.ustc.edu.cn.<br>



