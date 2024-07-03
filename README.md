# Dragon-Alpha-v1.2-source [To be continued]
> The source code of Dragon-Alpha-v1.2.

# I. About Cu32
**1.** ___Cu32___ is a GPU acceleration library for FP32 tensor computing, and consists of 14 components:<br>
> - [1] _Cuda_: serves as the bridge between JVM (CPU) and GPU. Its functions includes data-copy, data-transfer, tensor-initialization, memory-management, and computing scheduling. It also
> provides Java APIs to manage Cuda memory, event, stream, and device.
> - [2] _CudaDevice_: get and manage the information of Cuda Device, like id, name, SM number, and L2 cache size.
> - [3] _Cuda_random_: generates pseudo random numbers, using many threads.
> - [4] _Cuda_image_: includes functions to process image in int8 datatype. These functions are not limited to 3-channel BGR images, and can also process HSIs (Hyper Spectrul Images).
> - [5] _Cuda_math_: includes element-wise functions, like Relu, Softmax, Gelu, and BatchNorm.
> - [6] _Cuda_expk2_: The extension library of Cuda_math and Cuda.
> - [7] _Cuda_pool2D_: implements the forward propagation of Average-2D-Pooling and Max-2D-Pooling layers.
> - [8] _Cuda_upool2D_: implements the backward propagation of Average-2D-Pooling and Max-2D-Pooling layers.
> - [9] _Cuda_conv3D_: implements the forward propgation of 2D convolutional layers.
> - [10] _Cuda_dconv3D_deltaX_: implements the backward propagation of 2D convolutional layers, and this library is used to find the gradient of input feature maps.
> - [11] _Cuda_dconv3D_deltaW_: implements the backward propagation of 2D convolutional layers, and this library is used to find the gradient of filters. This lib contains some algorithms we haven't publicly disclosed, so the source has not been uploaded.
> - [12] _Cuda_reduce_: includes some reduction operators, and can be used to find mean, variance, maximum, minimum, etc.
> - [13] _Cuda_mat_: contains 3 types of matrix multiplications:  $A * B$, $A * B^T$, $A^T * B$
> - [14] _Cuda_batchMatMul_: contains 3 types of batch matrix multiplications:  $A * B$, $A * B^T$, $A^T * B$.
>> These 14 components have corresponding dlls and VS2017-studio projects. Due to my limited personal ability, I don't have enough time to optimize all kernel functions. Instead, I have to pay attention to optimize the kernel functions that have the highest performance up-limit. Therefore, matrix multiplication operators have good performance when dimensions are multiples of $64$, while convolution operators perform well with $64x$ channel. <br>
>> I paid attention to readbility when writing ___Cu32___. I believe smar you can understand them, without too much comments. To better understand ___Cu32___, please read the code of CudaFloat32EngineBase.java, which contains the Java APIs of ___Cu32___ and some higher-level operatioal logics.
  
**2.**  Under the Apache-2.0 License, you can modify and recompile the source of ___Cu32___ :
> - ___Cu32___ is now only complied for 64-bit Windows. Kindly recompile it for other Operating Systems like Centos, and Ubuntu.
> - I recommand nvcc-11.5 compilter for RTX-30XX GPU, and nvcc-11.8 for RTX-40XX GPU. To  generate the fastest code, I suggest you to try different versions of nvcc on your platforms.
> - Except for _Cuda_dconv3D_deltaX_ requires $compute$ and $sm >= 60$, the other libs can be compiled with $compute$ and $sm >= 52$.
> - I recommand $compute = sm = 52$ or $70$ configurations. Besides, I encourage you to try different compile conifgurations on your hardware to select the best configuration.
> - Since ___Cu32___ has been integrated to Dragon-Alpha through JNI, there are relevant head files included in progjects. However, you can extract the key code and rewrite them for other purpose.

**3.**. The convolution algorithms in  ___Cu32___ are as follows:
> - [1] _GEMM_: It supports both forward and backward propagation.
> - [2] _GEMMR_: It's an variant of _GEMM_, which transposes the filters from to enhance bandwidth.
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



