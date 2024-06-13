/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

import java.io.Serializable;
import java.util.Objects;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.util.lang.annotation.Desc;

/**
 *
 * @author Gilgamesh
 */
public final class CudaDevice implements Serializable
{
    public CudaDevice() { this(Cuda.getDeviceId()); }
    public CudaDevice(int device_id) { this.setId(device_id); }
    
    public static CudaDevice[] getDevices() {
        CudaDevice[] devices = new CudaDevice[Cuda.getDeviceCount()];
        for(int i=0;i<devices.length;i++) devices[i] = new CudaDevice(i);
        return devices;
    }
    
    //<editor-fold defaultstate="collapsed" desc="CudaDeviceProperties">
    private native void setProperty(int device_id);//passed
   
    protected int id;
    
    //<editor-fold defaultstate="collapsed" desc="properties for a device">
    @Desc("ASCII string identifying device")
    protected String name;//passed
    @Desc("16-byte universally unique identifier")
    protected byte[] uuid;//char[16] passed
    @Desc("8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms")
    protected byte[] luid;//char[8] passed
    @Desc("LUID device node mask. Value is undefined on TCC and non-Windows platforms")
    protected int luidDeviceNodeMask;
    
    @Desc("Major compute capability")    
    protected int major;//passed
    @Desc("Minor compute capability")
    protected int minor;//passed
    
    @Desc("PCI bus ID of the device")
    protected int pciBusID;//passed
    @Desc("PCI device ID of the device")
    protected int pciDeviceID;//passed
    @Desc("PCI domain ID of the device")
    protected int pciDomainID;//passed
    @Desc("Unique identifier for a group of devices on the same multi-GPU board")
    protected int multiGpuBoardGroupID;//pased
    
    @Desc("Compute mode (See ::cudaComputeMode)")
    protected int computeMode;//passed
    @Desc("Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount")
    protected boolean deviceOverlap;//passed
    @Desc("Device is integrated as opposed to discrete")
    protected boolean integrated;//passed
    @Desc("Specified whether there is a run time limit on kernels")
    protected boolean kernelExecTimeoutEnabled;//passed
    @Desc("Device shares a unified address space with the host")
    protected boolean unifiedAddressing;
    @Desc("Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer")
    protected boolean canMapHostMemory;//passed
    @Desc("Device can access host registered memory at the same virtual address as the CPU")
    protected boolean canUseHostPointerForRegisteredMem;//passed
    @Desc("Host can directly access managed memory on the device without migration")
    protected boolean directManagedMemAccessFromHost;//passed
    @Desc("Device can possibly execute multiple kernels concurrently")
    protected boolean concurrentKernels;//passed
    @Desc("Device has ECC support enabled")
    protected boolean ECCEnabled;//passed
    @Desc("1 if device is a Tesla device using TCC driver, 0 otherwise")
    protected boolean tccDriver;//passed
    @Desc("Device supports stream priorities")
    protected boolean streamPrioritiesSupported;//passed
    @Desc("Device supports caching globals in L1")
    protected boolean globalL1CacheSupported;//passed
    @Desc("Device supports caching locals in L1")
    protected boolean localL1CacheSupported;//passed
    @Desc("Device supports allocating managed memory on this system")
    protected boolean managedMemory;//passed
    @Desc("Device supports allocating managed memory on this system")
    protected boolean multiGpuBoard;//passed
    @Desc("Link between the device and the host supports native atomic operations")
    protected boolean hostNativeAtomicSupported;
    @Desc("Device supports coherently accessing pageable memory without calling cudaHostRegister on it")
    protected boolean pageableMemoryAccess;//passed
    @Desc("Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel")
    protected boolean cooperativeLaunch;//passed
    @Desc("Device can coherently access managed memory concurrently with the CPU")
    protected boolean concurrentManagedAccess;//passed
    @Desc("Device supports Compute Preemption")
    protected boolean computePreemptionSupported;//passed
    
    @Desc("Clock frequency in kilohertz")
    protected int clockRate;//passed
    @Desc("Peak memory clock frequency in kilohertz")
    protected int memoryClockRate;//passed
    @Desc("Global memory bus width in bits")
    protected int memoryBusWidth;//passed
    @Desc("Global memory available on device in bytes")
    protected long totalGlobalMemory;//passed
    @Desc("Constant memory available on device in bytes")
    protected long totalConstMemory;//passed
    @Desc("Size of L2 cache in bytes")
    protected int l2CacheSize;//passed
    @Desc("Device's maximum l2 persisting lines capacity setting in bytes")
    protected int persistingL2CacheMaxSize;//passed
    @Desc("Number of multiprocessors on device")
    protected int multiProcessorCount;//passed
    @Desc("Number of asynchronous engines")
    protected int asyncEngineCount;//ppased
    @Desc("Maximum size of each dimension of a grid")
    protected int[] maxGridSize;//int[3] passed
    @Desc(" Ratio of single precision performance (in floating-point operations per second) to double precision performance")
    protected int singleToDoublePrecisionPerfRatio;//passed
    @Desc("The maximum value of ::cudaAccessPolicyWindow::num_bytes")
    protected int accessPolicyMaxWindowSize;//passed
    @Desc("Maximum pitch in bytes allowed by memory copies")
    protected long memoryPitch;//passed
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="properties for a multiprocessor">
    @Desc("Maximum resident threads per multiprocessor")
    protected int maxThreadsPerMultiProcessor;//passed
    @Desc("Shared memory available per multiprocessor in bytes")
    protected long sharedMemPerMultiprocessor;//passed
    @Desc("32-bit registers available per multiprocessor")
    protected int regsPerMultiprocessor;//passed
    @Desc("Maximum number of resident blocks per multiprocessor")
    protected int maxBlocksPerMultiProcessor;//passed
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="properties for a block">
    @Desc("Shared memory available per block in bytes")
    protected long sharedMemPerBlock;//passed
    @Desc("Shared memory reserved by CUDA driver per block in bytes")
    protected long reservedSharedMemPerBlock;//passed
    @Desc("32-bit registers available per block ")
    protected long sharedMemPerBlockOptin;//passed
    @Desc("32-bit registers available per block")
    protected int regsPerBlock;//passed
    @Desc("Maximum number of threads per block")
    protected int maxThreadsPerBlock;//passed
    @Desc("Maximum size of each dimension of a block")
    protected int[] maxThreadsDim;//int[3] passed
    @Desc("Warp size in threads")
    protected int warpSize;//passed
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="properties for texture memory">
    @Desc("Alignment requirement for textures")
    protected long textureAlignment;//passed
    @Desc("Pitch alignment requirement for texture references bound to pitched memory")
    protected long texturePitchAlignment;//passed
    @Desc("Maximum 1D texture size")
    protected int maxTexture1D;//passed
    @Desc("Maximum 1D mipmapped texture size")
    protected int maxTexture1DMipmap;//passed
    @Desc("Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead")
    protected int maxTexture1DLinear;//passed
    @Desc("Maximum 2D texture dimensions")
    protected int[] maxTexture2D;//int[2] passed
    @Desc("Maximum 2D mipmapped texture dimensions")
    protected int[] maxTexture2DMipmap;//int[2] passed
    @Desc("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory")
    protected int[] maxTexture2DLinear;//int[3] passed
    @Desc("Maximum 2D texture dimensions if texture gather operations have to be performed")
    protected int[] maxTexture2DGather;//int[2] passed
    @Desc("Maximum 3D texture dimensions")
    protected int[] maxTexture3D;//int[3]; passed
    @Desc("Maximum alternate 3D texture dimensions")
    protected int[] maxTexture3DAlt;//int[3] passed
    @Desc(" Maximum 1D layered texture dimensions")
    protected int[] maxTexture1DLayered;//int[2] passed
    @Desc("Maximum 2D layered texture dimensions")
    protected int[] maxTexture2DLayered;//int[3] passed
    @Desc("Maximum Cubemap layered texture dimensions")
    protected int[] maxTextureCubemapLayered;//int[2]
    //</editor-fold>          
    //<editor-fold defaultstate="collapsed" desc="properties for surface memory">
    @Desc("Alignment requirements for surfaces")
    protected long surfaceAlignment;
    @Desc("Maximum 1D surface size")
    protected int maxSurface1D;//passed  
    @Desc("Maximum 2D surface dimensions")
    protected int[] maxSurface2D;//int[2] passed     
    @Desc("Maximum 3D surface dimensions")
    protected int[] maxSurface3D;//int[3] passed
    @Desc("Maximum 1D layered surface dimensions")
    protected int[] maxSurface1DLayered;//int[2] passed
    @Desc("Maximum 2D layered surface dimensions")
    protected int[] maxSurface2DLayered;//int[3] passed
    @Desc("Maximum Cubemap surface dimensions")
    protected int maxSurfaceCubemap;//passed
    @Desc("Maximum Cubemap layered surface dimensions")
    protected int[] maxSurfaceCubemapLayered;//int[2]
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="CudaDeviceProperties-Getters">
    public final int getId() { return id; }
    public final String getName() { return name; }
    public final byte[] getUuid() { return uuid; }
    public final byte[] getLuid() { return luid; }
    public final int getLuidDeviceNodeMask() { return luidDeviceNodeMask; }
    public final int getMajor() { return major; }
    public final int getMinor() { return minor; }
    public final int getPciBusID() { return pciBusID; }
    public final int getPciDeviceID() { return pciDeviceID; }
    public final int getPciDomainID() { return pciDomainID; }
    public final int getMultiGpuBoardGroupID() { return multiGpuBoardGroupID; }
    public final int getComputeMode() { return computeMode; }
    public final boolean isDeviceOverlap() { return deviceOverlap; }
    public final boolean isIntegrated() { return integrated; }
    public final boolean isKernelExecTimeoutEnabled() { return kernelExecTimeoutEnabled; }
    public final boolean isUnifiedAddressing() { return unifiedAddressing; }
    public final boolean isCanMapHostMemory() { return canMapHostMemory; }
    public final boolean isCanUseHostPointerForRegisteredMem() { return canUseHostPointerForRegisteredMem; }
    public final boolean isDirectManagedMemAccessFromHost() { return directManagedMemAccessFromHost; }
    public final boolean isConcurrentKernels() { return concurrentKernels; }
    public final boolean isECCEnabled() { return ECCEnabled; }
    public final boolean isTccDriver() { return tccDriver; }
    public final boolean isStreamPrioritiesSupported() { return streamPrioritiesSupported; }
    public final boolean isGlobalL1CacheSupported() { return globalL1CacheSupported; }
    public final boolean isLocalL1CacheSupported() { return localL1CacheSupported; }
    public final boolean isManagedMemory() { return managedMemory; }
    public final boolean isMultiGpuBoard() { return multiGpuBoard; }
    public final boolean isHostNativeAtomicSupported() { return hostNativeAtomicSupported; }
    public final boolean isPageableMemoryAccess() { return pageableMemoryAccess; }
    public final boolean isCooperativeLaunch() { return cooperativeLaunch; }
    public final boolean isConcurrentManagedAccess() { return concurrentManagedAccess; }
    public final boolean isComputePreemptionSupported() { return computePreemptionSupported; }
    public final int getClockRate() { return clockRate; }
    public final int getMemoryClockRate() { return memoryClockRate; }
    public final int getMemoryBusWidth() { return memoryBusWidth; }
    public final long getTotalGlobalMemory() { return totalGlobalMemory; }
    public final long getTotalConstMemory() { return totalConstMemory; }
    public final int getL2CacheSize() { return l2CacheSize; }
    public final int getPersistingL2CacheMaxSize() { return persistingL2CacheMaxSize; }
    public final int multiProcessorCount() { return multiProcessorCount; }
    public final int getAsyncEngineCount() { return asyncEngineCount; }
    public final int[] getMaxGridSize() { return maxGridSize; }
    public final int getSingleToDoublePrecisionPerfRatio() { return singleToDoublePrecisionPerfRatio; }
    public final int getAccessPolicyMaxWindowSize() { return accessPolicyMaxWindowSize; }
    public final long getMemoryPitch() { return memoryPitch; }
    public final int getMaxThreadsPerMultiProcessor() { return maxThreadsPerMultiProcessor; }
    public final long getSharedMemPerMultiprocessor() { return sharedMemPerMultiprocessor; }
    public final int getRegsPerMultiprocessor() { return regsPerMultiprocessor; }
    public final int getMaxBlocksPerMultiProcessor() { return maxBlocksPerMultiProcessor; }
    public final long getSharedMemPerBlock() { return sharedMemPerBlock; }
    public final long getReservedSharedMemPerBlock() { return reservedSharedMemPerBlock; }
    public final long getSharedMemPerBlockOptin() { return sharedMemPerBlockOptin; }
    public final int getRegsPerBlock() { return regsPerBlock; }
    public final int getMaxThreadsPerBlock() { return maxThreadsPerBlock; }
    public final int[] getMaxThreadsDim() { return maxThreadsDim; }
    public final int getWarpSize() { return warpSize; }
    public final long getTextureAlignment() { return textureAlignment; }
    public final long getTexturePitchAlignment() { return texturePitchAlignment; }
    public final int getMaxTexture1D() { return maxTexture1D; }
    public final int getMaxTexture1DMipmap() { return maxTexture1DMipmap; }
    public final int getMaxTexture1DLinear() { return maxTexture1DLinear; }
    public final int[] getMaxTexture2D() { return maxTexture2D; }
    public final int[] getMaxTexture2DMipmap() { return maxTexture2DMipmap; }
    public final int[] getMaxTexture2DLinear() { return maxTexture2DLinear; }
    public final int[] getMaxTexture2DGather() { return maxTexture2DGather; }
    public final int[] getMaxTexture3D() { return maxTexture3D; }
    public final int[] getMaxTexture3DAlt() { return maxTexture3DAlt; }
    public final int[] getMaxTexture1DLayered() { return maxTexture1DLayered; }
    public final int[] getMaxTexture2DLayered() { return maxTexture2DLayered; }
    public final int[] getMaxTextureCubemapLayered() { return maxTextureCubemapLayered; }
    public final long getSurfaceAlignment() { return surfaceAlignment; }
    public final int getMaxSurface1D() { return maxSurface1D; }
    public final int[] getMaxSurface2D() { return maxSurface2D; }
    public final int[] getMaxSurface3D() { return maxSurface3D; }
    public final int[] getMaxSurface1DLayered() { return maxSurface1DLayered; }
    public final int[] getMaxSurface2DLayered() { return maxSurface2DLayered; }
    public final int getMaxSurfaceCubemap() { return maxSurfaceCubemap; }
    public final int[] getMaxSurfaceCubemapLayered() { return maxSurfaceCubemapLayered; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: CudaComputeMode" >
    public static class ComputeMode
    {
        public static final int Default = 0;//Default compute mode (Multiple threads can use ::cudaSetDevice() with this device)
        public static final int Exclusive = 1;//Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device)
        public static final int Prohibited = 2;//Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device)
        public static final int ExclusiveProcess = 3;//Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device)
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public synchronized void setId(int device_id) {
        this.id = device_id;
        this.setProperty(device_id);
    }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append("{ ");
        sb.append("id = ").append(id);
        sb.append(", name = ").append(name).append(" }");
    }
    @Override
    public String toString() {
        StringBuilder sb=new StringBuilder(512);
        this.append(sb);
        return sb.toString();
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 79 * hash + this.id;
        hash = 79 * hash + Objects.hashCode(this.name);
        return hash;
    }
    
    @Override
    public boolean equals(Object o) {
        if(!(o instanceof CudaDevice)) return false;
        CudaDevice dev = (CudaDevice) o;
        return (dev.id == id) && dev.name.equals(name);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Init-Code">
    static
    {
        if(CudaFloat32EngineBase.__TEST_MODE__())
        synchronized(CudaDevice.class) {
            System.load("D:\\virtual disc Z-Gilgamesh\\Dargon-alpha\\engine-cuda-base\\CudaDevice\\x64\\Release\\CudaDevice.dll");
        }
    }
    //</editor-fold>
}
