/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda;

import java.io.File;
import java.util.Objects;
import z.dragon.engine.EngineBase;
import z.dragon.engine.EngineCore;
import z.dragon.engine.Result;
import z.dragon.engine.Result.IndexedResult;
import z.dragon.engine.Syncer;
import z.dragon.engine.cuda.impl.math.Cuda_matMul;
import z.dragon.engine.cuda.impl.CudaDevice;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaStreamPool;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.dragon.engine.cuda.impl.PinnedMempool;
import z.dragon.engine.cuda.impl.math.Cuda_batchMatMul;
import z.dragon.engine.cuda.impl.math.Cuda_conv3D;
import z.dragon.engine.cuda.impl.math.Cuda_dconv3D_deltaW;
import z.dragon.engine.cuda.impl.math.Cuda_dconv3D_deltaX;
import z.dragon.engine.cuda.impl.math.Cuda_depthwise_conv3D;
import z.dragon.engine.cuda.impl.math.Cuda_depthwise_dconv3D_deltaW;
import z.dragon.engine.cuda.impl.math.Cuda_depthwise_dconv3D_deltaX;
import z.dragon.engine.cuda.impl.math.Cuda_function;
import z.dragon.engine.cuda.impl.math.Cuda_image;
import z.dragon.engine.cuda.impl.math.Cuda_pool2D;
import z.dragon.engine.cuda.impl.math.Cuda_random;
import z.dragon.engine.cuda.impl.math.Cuda_reduce;
import z.dragon.engine.cuda.impl.math.Cuda_upool2D;
import z.dragon.engine.cuda.impl.math.FloatFunc;
import z.dragon.engine.cuda.impl.math.FloatFunc.FloatFuncConfig;
import z.util.math.ExMath;
import z.util.math.Num;
import z.util.math.vector.Vector;
import z.util.function.Int2Float;
import z.dragon.engine.cuda.impl.math.Cuda_depthwise_dconv3D_deltaW.DWConvDW_GZ_Decider;

/**
 *
 * @author Gilgamesh
 */
public class CudaFloat32EngineBase extends EngineBase { 
    //<editor-fold defaultstate="collapsed" desc="native-lib">
    private static final String PATH = "native-lib\\cuda_float32";
    
    private static boolean NATIVE_LOAD = false;
    private static final boolean TEST_MODE = false;
    
    public static boolean __TEST_MODE__() { return TEST_MODE; }
    public static boolean __NATIVE_LOAD__() { return NATIVE_LOAD; }
    public static void __SET_NATIVE_LOAD__(boolean flag) { NATIVE_LOAD = flag; }
    
    public static synchronized void load_native_lib(String alpha_home) {
        File nativeLib = new File(alpha_home, PATH);
        for(File lib : nativeLib.listFiles((File file) -> { return file.getName().endsWith(".dll");}))
            System.load(lib.getAbsolutePath());
        System.setProperty("ALPHA_HOME", alpha_home);
    }
    //</editor-fold>
     
    protected CudaDevice dev;
    protected CudaStreamPool streamPool;
    protected PinnedMempool bufPool;// = new PinnedMemoryPool(MEM_1MB * 128);
    
    //<editor-fold defaultstate="collapsed" desc="Init-Code"> 
    protected int Init_smooth_maxPart(int x) {
        if(x <= 32) x = (x + 3) >> 2 << 2;//padding to 4x;
        else        x = (x + 7) >> 3 << 3;//padding to 8x
        return ExMath.clip(x, 16, 128);
    }
    //</editor-fold>
    public CudaFloat32EngineBase() { this(new CudaDevice(), 32); }
    public CudaFloat32EngineBase(int deviceId, int streamPool_maxsize) { this(new CudaDevice(deviceId), streamPool_maxsize); }
    public CudaFloat32EngineBase(CudaDevice device, int streamPool_maxsize) {
        super("cuda_float32", 2, "cuda_int32", "cuda_int8");
        if(device == null) throw new NullPointerException("device");

        this.dev = device;
        this.streamPool = CudaStreamPool.instance(dev, streamPool_maxsize, 16);
         
        //param initailze-------------------------------------------------------
        final int SM_count = dev.multiProcessorCount();
        conv3D_dW_GemmSK_maxPart      = Init_smooth_maxPart(SM_count);
        conv3D_dW_WinogradSHW_maxPart = Init_smooth_maxPart((int) (1.21f * SM_count)); 
        dwconv3D_dW_GemmSK_maxPart    = Init_smooth_maxPart((int) (3.28f * SM_count));
        matMul_sk_maxPart             = Init_smooth_maxPart(SM_count);
        matMulT1_sk_maxPart           = Init_smooth_maxPart(SM_count);
        matMulT2_sk_maxPart           = Init_smooth_maxPart(SM_count);
    }
    
    public CudaFloat32EngineBase tf32(boolean flag) { 
        matMul_tf32   = flag;
        matMulT1_tf32 = flag;
        matMulT2_tf32 = flag;
        batchMatMul_tf32   = flag;
        batchMatMulT1_tf32 = flag;
        batchMatMulT2_tf32 = flag;
        return this;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public CudaStreamPool streamPool() {return streamPool;}
    public synchronized CudaFloat32EngineBase streamPool(CudaStreamPool streamPool) {  this.streamPool = streamPool; return this; }
    
    public PinnedMempool buf_mempool() { return bufPool; }
    public synchronized CudaFloat32EngineBase buf_mempool(PinnedMempool mempool) { this.bufPool = mempool; return this; }
    
    public CudaDevice device() { return dev; }
    
    @Override
    public void append(StringBuilder sb) {
        super.append(sb);
        sb.delete(sb.length() - 3, sb.length() - 1);
        if(bufPool != null) {//add properties of bufPool
            bufPool.meta_data().forEach((k, v) -> {  
                sb.append("\nbufPool.").append(k).append(" = ").append(v);
            });
        }
        sb.append(" }");
    }
    
    @Override
    public synchronized void clear() {
        streamPool.clear();
        bufPool.clear();
    }
    
    @Override
    public void finalize() throws Throwable {
        super.finalize();
        this.clear();
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 59 * hash + Objects.hashCode(this.dev);
        return hash;
    }
    
    @Override
    public boolean equals(Object o) {
        if(!(o instanceof CudaFloat32EngineBase)) return false;
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) o;
        return Objects.equals(cu32.dev, dev);
    }
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="extra: int32">
    //<editor-fold defaultstate="collapsed" desc="tensor<int32>: get">
    @Override
    public void get1D_int32(long address, int[] value, int length) {
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.get1D_int(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get1D_v2_int(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void get2D_int32(long address, int[] value, int height, int width, int stride) {
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.get2D_int(stream, address, value, height, width, stride); 
            streamPool.returnStream(stream);
            return;
        }
      
        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get2D_v2_int(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor<int32>: set">
    @Override
    public void set1D_int32(long address, int[] value, int length) {
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.set1D_int(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set1D_v2_int(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void set2D_int32(long address, int[] value, int height, int width, int stride) {
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.set2D_int(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }

        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set2D_v2_int(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="extra: int8">
    //<editor-fold defaultstate="collapsed" desc="tensor<int8>: constant">
    @Override
    public Syncer set1D_int8(long address, int value, int length) {
        long stream_address = streamPool.getStream();
        Cuda.set1D_char(stream_address, address, value, length);
        return new StreamSyncer(streamPool, stream_address);
    }

    @Override
    public Syncer set2D_int8(long address, int value, int height, int width, int stride) {
        long stream_address = streamPool.getStream();
        Cuda.set2D_char(stream_address, address, value, height, width, stride);
        return new StreamSyncer(streamPool, stream_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor<int8>: get">
    @Override
    public void get1D_int8(long address, byte[] value, int length)  {
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.get1D_char(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc( ((length + 3) >> 2) << 2 );//padding length to 4x
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get1D_v2_char(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void get2D_int8(long address, byte[] value, int height, int width, int stride) {
        //[common version]------------------------------------------------------
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.get2D_char(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }
        
        //[width = 3, stride = 4, used for pictures(JPEG) width 3 channels]-----
        if((width == 3) && (stride == 4)) {
            long[] bufBlock = bufPool.malloc(height * stride);
            long buf_size = bufBlock[0], buf_address = bufBlock[1];
            Cuda.get2D_v2_char_W3S4(stream, address, value, buf_address, height);
            streamPool.returnStream(stream);
            bufPool.free(buf_size, buf_address);
            return;
        }
        
        //[fast version]--------------------------------------------------------
        long[] bufBlock = bufPool.malloc(height * stride);//stride % 4 == 0
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get2D_v2_char(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor<int8>: set">
    @Override
    public void set1D_int8(long address, byte[] value, int length) {
        long stream = streamPool.getStream();
        if(bufPool == null) {
            Cuda.set1D_char(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc( ((length + 3) >> 2) << 2 );//padding length to 4x
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set1D_v2_char(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void set2D_int8(long address, byte[] value, int height, int width, int stride) {
        //[common version]------------------------------------------------------
        if(bufPool == null) {
           long stream = streamPool.getStream();
            Cuda.set2D_char(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }
        
        //[width = 3, stride = 4, used for pictures (JPEG) width 3 channels]----
        if((width == 3) && (stride == 4)  && (height % 4 == 0)) {
            long stream = streamPool.getStream();
            
            long[] block1 = bufPool.malloc(height * 3);
            long[] block2 = bufPool.malloc(height * 4);//padding length to 4x
            long buf1_size = block1[0], buf1_address = block1[1];
            long buf2_size = block2[0], buf2_address = block2[1];
            
            Cuda.set2D_v2_char_W3S4(stream, address, value, buf1_address, buf2_address, height);
            
            bufPool.free(buf1_size, buf1_address);
            bufPool.free(buf2_size, buf2_address);
            streamPool.returnStream(stream);
            return;
        }
        
        //[fast version]--------------------------------------------------------
        long stream = streamPool.getStream();
        
        long[] block = bufPool.malloc(height * stride);//padding length to 4x
        long buf_size = block[0], buf_address = block[1];
        
        Cuda.set2D_v2_char(stream, address, value, buf_address, height, width, stride);
        
        bufPool.free(buf_size, buf_address);
        streamPool.returnStream(stream);
    }
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Syncer & Result">
    //<editor-fold defaultstate="collapsed" desc="class: StreamSyncer">
    public static final class StreamSyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long stream;
        private final long event;
        private boolean called = false;
        
        public final boolean called() { return called; }
        public final long stream() { return stream; }
        
        StreamSyncer(CudaStreamPool streamPool, long stream) {
            this.streamPool = streamPool;
            this.stream = stream;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize();
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStream(stream);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStream(stream);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: Stream2Syncer_1">
    public static final class Stream2Syncer_1 implements Syncer { //only sync stream1
        private final CudaStreamPool streamPool;
        private final long stream1;
        private final long stream2;
        private final long event;
        private boolean called = false;
        
        public final boolean called() { return called; }
        public final long stream1() { return stream1; }
        public final long stream2() { return stream2; }
        
        Stream2Syncer_1(CudaStreamPool streamPool, long stream1, long stream2) {
            this.streamPool = streamPool;
            this.stream1 = stream1;
            this.stream2 = stream2;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream1);
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return; 
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamBlockSyncer">
    public static final class StreamBlockSyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long stream;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block;
        
        public final boolean called() { return called; }
        public final long stream() { return stream; }
        public final long[] block() { return block; }
        
        StreamBlockSyncer(CudaStreamPool streamPool, long stream,
                EngineCore core, long[] block)
        {
            this.streamPool = streamPool;
            this.stream = stream;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
            
            this.core = core;
            this.block = block;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStream(stream);
            core.free(block[0], block[1]);
        }
        
        @Override
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStream(stream);
            core.free(block[0], block[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamBlock2Syncer">
    public static final class StreamBlock2Syncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long stream;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public final boolean called() { return called; }
        public final long stream() { return stream; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
        
        StreamBlock2Syncer(CudaStreamPool streamPool, long stream,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.stream = stream;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStream(stream);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStream(stream);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: Stream2Block2Syncer_1">
    public static final class Stream2Block2Syncer_1 implements Syncer {//only sync stream1
        private final CudaStreamPool streamPool;
        private final long stream1;
        private final long stream2;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public final boolean called() { return called; }
        public final long stream1() { return stream1; }
        public final long stream2() { return stream2; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
          
        Stream2Block2Syncer_1(CudaStreamPool streamPool, long stream1, long stream2,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.stream1 = stream1;
            this.stream2 = stream2;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream1);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }

        @Override
        protected void finalize() throws Throwable {
            super.finalize();
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: Stream2Block3Syncer_1">
    public static final class Stream2Block3Syncer_1 implements Syncer {//only sync stream1
        private final CudaStreamPool streamPool;
        private final long stream1;
        private final long stream2;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        private final long[] block3;
        
        public final boolean called() { return called; }
        public final long stream1() { return stream1; }
        public final long stream2() { return stream2; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
        public final long[] block3() { return block3; }
        
        Stream2Block3Syncer_1(CudaStreamPool streamPool, long stream1, long stream2,
                EngineCore core, long[] block1, long[] block2, long[] block3)
        {
            this.streamPool = streamPool;
            this.stream1 = stream1;
            this.stream2 = stream2;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream1);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
            this.block3 = block3;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
            core.free(block3[0], block3[1]);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
            core.free(block3[0], block3[1]);
        }
    }
    //</editor-fold>
  
    static final void sync_streamArray_2_stream0(long[] streams) {
        long event1 = Cuda.newEvent_DisableTiming();//streams -> streams[0]
        Cuda.eventRecord(event1, streams, streams.length);
        Cuda.streamWaitEvent_default(streams[0], event1);
        Cuda.deleteEvent(event1);
    }
    
    //<editor-fold defaultstate="collapsed" desc="class: StreamArraySyncer">
    public static final class StreamArraySyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        public final boolean called() { return called; }
        public long[] streams() { return streams; }
        
        StreamArraySyncer(CudaStreamPool streamPool, long[] streams) {
            this.streamPool = streamPool;
            this.streams = streams;
            
            sync_streamArray_2_stream0(streams);//streams -> streams[0]
            this.event = Cuda.newEvent_DisableTiming();//wait streams[0]
            Cuda.eventRecord(event, streams[0]);
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize();
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStreamArray(streams);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlockSyncer">
    public static final class StreamArrayBlockSyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block;
        
        public final boolean called() { return called; }
        public long[] streams() { return streams; }
        public long[] block() { return block; }
        
        StreamArrayBlockSyncer(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block)
        {
            //stream sync-------------------------------------------------------
            this.streamPool = streamPool;
            this.streams = streams;
            
            sync_streamArray_2_stream0(streams);//streams -> streams[0]
            this.event = Cuda.newEvent_DisableTiming();//wait streams[0]
            Cuda.eventRecord(event, streams[0]);
            
            //resource----------------------------------------------------------
            this.core = core;
            this.block = block;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); if(called) return; 
            Cuda.deleteEvent(event);
            core.free(block[0], block[1]);//release block1.mem_address
            streamPool.returnStreamArray(streams);
        }
             
        @Override
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            core.free(block[0], block[1]);//release block1.mem_address
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlock2Syncer">
    public static final class StreamArrayBlock2Syncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public final boolean called() { return called; }
        public final long[] streams() { return streams; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
        
        StreamArrayBlock2Syncer(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block1, long[] block2)
        {
            //stream sync-------------------------------------------------------
            this.streamPool = streamPool;
            this.streams = streams;
            
            sync_streamArray_2_stream0(streams);//streams -> streams[0]
            this.event = Cuda.newEvent_DisableTiming();//wait streams[0]
            Cuda.eventRecord(event, streams[0]);
            
            //resource----------------------------------------------------------
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
             
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if(called) return; 
                Cuda.deleteEvent(event);
            }
            core.free(block1[0], block1[1]);//release block1.mem_address
            core.free(block2[0], block2[1]);//release block2.mem_address
            streamPool.returnStreamArray(streams);
        }
        
        @Override
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            core.free(block1[0], block1[1]);//release block1.mem_address
            core.free(block2[0], block2[1]);//release block2.mem_address
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlock2Syncer_1">
    public static final class StreamArrayBlock2Syncer_1 implements Syncer {//only sync streams[0]
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public final boolean called() { return called; }
        public final long[] streams() { return streams; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
          
        StreamArrayBlock2Syncer_1(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStreamArray(streams);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStreamArray(streams);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlock4Syncer_1">
    public static final class StreamArrayBlock4Syncer_1 implements Syncer {//only sync streams[0]
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        private final long[] block3;
        private final long[] block4;
        
        public final boolean called() { return called; }
        public final long[] streams() { return streams; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
        public final long[] block3() { return block3; }
        public final long[] block4() { return block4; }
          
        StreamArrayBlock4Syncer_1(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block1, long[] block2, long[] block3, long[] block4)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
            this.block3 = block3;
            this.block4 = block4;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStreamArray(streams);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
            core.free(block3[0], block3[1]);
            core.free(block4[0], block4[1]);
        }
        
        @Override 
        public final synchronized void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStreamArray(streams);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
            core.free(block3[0], block3[1]);
            core.free(block4[0], block4[1]);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: BiasedForwardSyncer">
    public static final class BiasedForwardSyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        public final boolean called() { return called; }
        public final long[] streams() { return streams; }
        
        BiasedForwardSyncer(CudaStreamPool streamPool, long[] streams) {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);//streams[0], add bias after the transformation
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); if(called) return; 
            Cuda.deleteEvent(event);
            streamPool.returnStreamArray(streams);
        }
        
        @Override 
        public final synchronized void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: BiasedBlockForwardSyncer">
    public static final class BiasedForwardBlockSyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block;
        
        public final boolean called() { return called; }
        public final long[] streams() { return streams; }
        public final long[] block() { return block; }
        
        BiasedForwardBlockSyncer(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block) 
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);//streams[0], add bias after the transformation
            
            this.core = core;
            this.block = block;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            streamPool.returnStreamArray(streams);
            core.free(block[0], block[1]);
        }
        
        @Override 
        public final void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            streamPool.returnStreamArray(streams);
            core.free(block[0], block[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: SplitKSyncer">
    public static final class SplitKSyncer implements Syncer {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        
        private final EngineCore core;
        private final long[] block;
        private boolean called = false;
        
        public final boolean called() { return called; }
        public final long[] streams() { return streams; }
        public final long[] block() { return block; }
        
        SplitKSyncer(EngineCore core, long[] block,
                CudaStreamPool streamPool, long[] streams)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);//bufSum -> streams[0]
            
            this.core = core;
            this.block = block;
        }
        
        @Override
        protected void finalize() throws Throwable {
            super.finalize();
            synchronized(this) {
                if (called) return; 
                Cuda.deleteEvent(event);
            }
            core.free(block[0], block[1]);//release resources
            streamPool.returnStreamArray(streams);
        }
             
        @Override
        public final synchronized void sync() {
            synchronized(this) {
                if (called) return;
                Cuda.eventSync_Del(event);
                called = true;
            }
            core.free(block[0], block[1]);//release resources
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: StreamBlockResult">
    public static final class StreamBlockResult extends Result<Float> {
        private final CudaStreamPool streamPool;
        private final long stream;
        
        private final EngineCore core;
        private final long[] block;
        
        public final long stream() { return stream; } 
        public final long[] block() { return block; }
        
        StreamBlockResult(CudaStreamPool streamPool, long stream_address,
                EngineCore core, long[] block)
        {
            this.streamPool = streamPool;
            this.stream = stream_address;
            
            this.core = core;
            this.block = block;
        }

        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            if(isDone()) return;
            streamPool.returnStream(stream);
            core.free(block[0], block[1]);
        }

        @Override
        protected final Float waitResult() {
            float[] result = new float[1];
            Cuda.get1D(stream, block[1], result, 1);//V_address = block1
            
            streamPool.returnStream(stream);
            core.free(block[0], block[1]);
            return result[0];
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: FloatIndexedResult">
    public static final class FloatIndexedResult extends IndexedResult<Float> {
        private final CudaStreamPool streamPool;
        private final long stream;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public final long stream() { return stream; }
        public final long[] block1() { return block1; }
        public final long[] block2() { return block2; }
        
        FloatIndexedResult(CudaStreamPool streamPool, long stream_address,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.stream = stream_address;
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }

        @Override
        protected void finalize() throws Throwable {
            super.finalize(); 
            if(isDone()) return;
            streamPool.returnStream(stream);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
        }
        
        @Override
        protected final IndexedValue<Float> waitResult() {
            float[] result = new float[1];
            int[] index = new int[1];
            Cuda.get1D(stream, block1[1], result, 1);
            Cuda.get1D_int(stream, block2[1], index, 1);
            
            streamPool.returnStream(stream);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
            
            return new IndexedValue<>(index[0], result[0]);
        }
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Operations">
    @Override 
    public long malloc(long memsize) {
        long address;
        synchronized(Cuda.class) {
            Cuda.setDevice(dev.getId());
            address = Cuda.malloc(memsize);
        }
        return address;
    }
    
    @Override 
    public void free(long address)  {
        synchronized(Cuda.class) {
            Cuda.setDevice(dev.getId());
            Cuda.free(address);
        }
    }
    
    @Override 
    public Syncer memset(long address, int value, long memsize)  {
        long stream_address = streamPool.getStream();
        Cuda.memsetAsync(stream_address, address, value, memsize);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer memcpy(long dst_address, long src_address, long memsize) {
        long stream_address = streamPool.getStream();
        Cuda.memcpyAsyncDeviceToDevice(stream_address, dst_address, src_address, memsize);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    //<editor-fold defaultstate="collapsed" desc="tensor = constant">
    @Override 
    public Syncer set1D(long address, float value, int length)  {
        long stream_address = streamPool.getStream();
        Cuda.set1D(stream_address, address, value, length);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer set2D(long address, float value, int height, int width, int stride) {
        long stream_address = streamPool.getStream();
        Cuda.set2D(stream_address, address, value, height, width, stride);
        return new StreamSyncer(streamPool, stream_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor: get">
    @Override 
    public void get1D(long address, float[] value, int length) {
        long stream = streamPool.getStream();
        if(bufPool == null) {
            Cuda.get1D(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get1D_v2(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    
    @Override 
    public void get2D(long address, float[] value, int height, int width, int stride) {
        long stream = streamPool.getStream();
        if(bufPool == null) {
            Cuda.get2D(stream, address, value, height, width, stride); 
            streamPool.returnStream(stream);
            return;
        }
      
        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get2D_v2(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = value: set">
    @Override 
    public void set1D(long address, float[] value, int length) {
        long stream = streamPool.getStream();
        if(bufPool == null) {//no buffer memthod
            Cuda.set1D(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L); //pinned memory: faster
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set1D_v2(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    
    @Override 
    public void set2D(long address, float[] value, int height, int width, int stride) {
        long stream = streamPool.getStream();
        if(bufPool == null) {
            Cuda.set2D(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set2D_v2(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = another tensor">
    @Override 
    public Syncer setFrom1Dto2D(long src_address, int src_length, 
            long dst_address, int dst_height, int dst_width, int dst_stride) 
    {
        long stream_address = streamPool.getStream();
        Cuda.setFrom1Dto2D(stream_address,
                src_address, src_length,
                dst_address, dst_height, dst_width, dst_stride);
         return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer setFrom2Dto1D(long src_address, int src_height, int src_width, int src_stride, 
            long dst_address, int dst_length) 
    {
        long stream_address = streamPool.getStream();
        Cuda.setFrom2Dto1D(stream_address,
                src_address, src_height, 
                src_width, src_stride, dst_address, dst_length);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer setFrom2Dto2D(long src_address, int src_height, int src_width, int src_stride, 
            long dst_address, int dst_height, int dst_width, int dst_stride) 
    {
        long stream_address = streamPool.getStream();
        Cuda.setFrom2Dto2D(stream_address,
                src_address, src_height, src_width, src_stride, 
                dst_address, dst_height, dst_width, dst_stride);
        return new StreamSyncer(streamPool, stream_address);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    //<editor-fold defaultstate="collapsed" desc="gappedMemcpy2D">
    @Override
    public Syncer gappedMemcpy2D(
            long X_address, int Xstart, int strideX, 
            long Y_address, int Ystart, int strideY,
            int width, int length)
    {
        long stream = streamPool.getStream();
        Cuda_expk2.gappedMemcpy2D(stream,
                X_address, Xstart, strideX,
                Y_address, Ystart, strideY,
                width, length);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="indexedMemcpy2D">
    @Override
    public Syncer srcIndexedMemcpy(long Y_address, 
            long X_address, long Index_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_expk2.srcIndexedMemcpy(stream,
                X_address, Index_address,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer dstIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_expk2.dstIndexedMemcpy(stream,
                X_address, Index_address, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="transpose(2D -> 4D)">
    @Override
    public Syncer transpose(
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim, 
            int dimIndex1, int dimIndex2, 
            int strideX, int strideY, 
            int length) 
    {
        long stream = streamPool.getStream();
        if(Xdim.length == 4) {
            Cuda_expk2.transpose4D(stream, 
                    X_address, Y_address,
                    Xdim[1], Xdim[2], Xdim[3], 
                    Ydim[1], Ydim[2], Ydim[3], 
                    dimIndex1, dimIndex2,
                    strideX, strideY, length);
        }
        else if(Xdim.length == 3) {
            Cuda_expk2.transpose3D(stream,
                    X_address, Y_address, 
                    Xdim[1], Xdim[2], 
                    Ydim[1], Ydim[2], 
                    dimIndex1, dimIndex2,
                    strideX, strideY, length);
        }
        else {//X.dim = 2
            Cuda_expk2.transpose2D(stream,
                    X_address, Y_address,
                    Xdim[1], Ydim[1],
                    strideX, strideY,
                    length);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="rot180">
    @Override
    public Syncer rot180(long Y_address,
            long X_address,
            int IH, int IW, int IC, 
            int length) 
    {
        long stream = streamPool.getStream();
        Cuda_expk2.rot180(stream,
                X_address, Y_address, 
                IH, IW, IC, 
                length);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pad(2D -> 4D)">
    @Override
    public Syncer pad(//X.ndim = Y.ndim = p0.length
            long Y_address, int[] Ydim, 
            long X_address, int[] Xdim,
            int[] p0)
    {
        long stream = streamPool.getStream();
        if(Xdim.length == 4) {//ndim =4: [N, H, W, C]
            int IN = Xdim[0], ON = Ydim[0], pn0 = p0[0];
            int IH = Xdim[1], OH = Ydim[1], ph0 = p0[1];
            int IW = Xdim[2], OW = Ydim[2], pw0 = p0[2];
            int IC = Xdim[3], OC = Ydim[3], pc0 = p0[2];
            Cuda_expk2.pad4D(stream,
                    X_address, IN, IH, IW, IC,
                    Y_address, ON, OH, OW, OC,
                    pn0, ph0, pw0, pc0);
        }
        else if(Xdim.length == 3) {//ndim=3: [N, W, C]
            int IN = Xdim[0], ON = Ydim[0], pn0 = p0[0];
            int IW = Xdim[1], OW = Ydim[1], pw0 = p0[1];
            int IC = Xdim[2], OC = Ydim[2], pc0 = p0[2];
            Cuda_expk2.pad3D(stream,
                    X_address, IN, IW, IC, 
                    Y_address, ON, OW, OC, 
                    pn0, pw0, pc0);
        }
        else {//Xdim == 2, [N, C]. memcpy function for 1D case
            int IN = Xdim[0], ON = Ydim[0], pn0 = p0[0];
            int IC = Xdim[1], OC = Ydim[1], pc0 = p0[1];
            Cuda_expk2.pad2D(stream, 
                    X_address, IN, IC, 
                    Y_address, ON, OC, 
                    pn0, pc0);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="trim(2D -> 4D)">
    @Override
    public Syncer trim(//X.ndim = Y.ndim = p0.length
            long Y_address, int[] Ydim, 
            long X_address, int[] Xdim, 
            int[] p0) 
    {
        long stream = streamPool.getStream();
        if(Xdim.length == 4) {//ndim =4: [N, H, W, C]
            int IN = Xdim[0], ON = Ydim[0], pn0 = p0[0];
            int IH = Xdim[1], OH = Ydim[1], ph0 = p0[1];
            int IW = Xdim[2], OW = Ydim[2], pw0 = p0[2];
            int IC = Xdim[3], OC = Ydim[3], pc0 = p0[2];
            Cuda_expk2.trim4D(stream,
                    X_address, IN, IH, IW, IC, 
                    Y_address, ON, OH, OW, OC, 
                    pn0, ph0, pw0, pc0);
        }
        else if(Xdim.length == 3) {//ndim=3: [N, W, C]
            int IN = Xdim[0], ON = Ydim[0], pn0 = p0[0];
            int IW = Xdim[1], OW = Ydim[1], pw0 = p0[1];
            int IC = Xdim[2], OC = Ydim[2], pc0 = p0[2];
            Cuda_expk2.trim3D(stream, 
                    X_address, IN, IW, IC,
                    Y_address, ON, OW, OC,
                    pn0, pw0, pc0);
        }
        else if(Xdim.length == 2) {//Xdim == 2, [N, C]. memcpy function for 1D case
            int IN = Xdim[0], ON = Ydim[0], pn0 = p0[0];
            int IC = Xdim[1], OC = Ydim[1], pc0 = p0[1];
            Cuda_expk2.trim2D(stream, 
                    X_address, IN, IC,
                    Y_address, ON, OC, 
                    pn0, pc0);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //<editor-fold defaultstate="collapsed" desc="Normal matMul">
    //<editor-fold defaultstate="collapsed" desc="matMul">
    protected int matMul_sk_threshold = 512;
    protected int matMul_sk_maxPart = 16;
    
    protected float matMul_sk_minBlock_perSM = 1.2f;//64 / 39 = 1.641
    protected float matMul_sk_maxBlock_perSM = 4.0f;
    
    public int matMul_splitK_threshold() { return matMul_sk_threshold; }
    public CudaFloat32EngineBase matMul_splitK_threshold(int threshold) {
        if(!Num.isPowerOf2(threshold)) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must be a power of 2", threshold));
        if(threshold <= 0) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must > 0", threshold));
        matMul_sk_threshold = threshold;
        return this;
    }
   
    public int matMul_splitK_maxPart() { return matMul_sk_maxPart; }
    public CudaFloat32EngineBase matMul_splitK_maxPart(int maxPart) {
        if(maxPart < 4) throw new IllegalArgumentException(String.format(
                "maxPartNum { got %d } must >= 4", maxPart));
        matMul_sk_maxPart = maxPart;
        return this;
    }
   
    public float matMul_splitK_minBlock_perSM() { return matMul_sk_minBlock_perSM; } 
    public CudaFloat32EngineBase matMul_splitK_mainBlock_perSM(int minBlock) {
        if(minBlock < 0) throw new IllegalArgumentException(String.format(
                "minBlock { got %d } must >= 1", minBlock));
        matMul_sk_minBlock_perSM = minBlock;
        return this;
    }
    
    public float matMul_splitK_maxBlock_perSM() { return matMul_sk_maxBlock_perSM; } 
    public CudaFloat32EngineBase matMul_splitK_maxBlock_perSM(float maxBlock) {
        if(maxBlock < 0) throw new IllegalArgumentException(String.format(
                "maxBlock { got %f } must >= 1", maxBlock));
        matMul_sk_maxBlock_perSM = maxBlock;
        return this;
    }
    
    protected boolean matMul_tf32 = false;
    public boolean matMul_tf32() { return matMul_tf32; }
    public CudaFloat32EngineBase matMul_tf32(boolean flag) { matMul_tf32 = flag; return this; }
    
    @Override
    public Syncer matMul(long C_address, 
            long A_address, long B_address, 
            int N, int M, int K) 
    {
        float fGridZ = 1; int b0 = 1;//splitK decision
        if(K >= matMul_sk_threshold) {
            b0 = Cuda_matMul.blockNum(N, M);//matMul: [N, K] * [K, M] = [N, M]
            float expect_block_perSM = matMul_sk_minBlock_perSM + 0.4f * K / (K + 1024.0f);
            if(b0 < dev.multiProcessorCount() * expect_block_perSM) {
                int b1 = Cuda_matMul.blockNum(K, M);//deltaB[K, M] = matMulT1: A^T[K, N] * deltaC[N, M]
                int b2 = Cuda_matMul.blockNum(N, K);//deltaA[N, K] = matMulT2: deltaC[N, M] * B^T[M, K]
                fGridZ = (b1 + b2) / (b0 * 2.0f);
            }
        }
        
        int length = Cuda_matMul.nstream(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        
        //======[no need to split]==============================================
        if(fGridZ < 1.8f) {
            if (matMul_tf32 && (N > 127) && (M > 127) && (K % 8 == 0)) 
                Cuda_matMul.matMul_mma(streamArray, length, 
                        A_address, B_address, C_address,
                        N, M, K);
            else /*default method*/
                Cuda_matMul.matMul(streamArray, length, 
                        A_address, B_address, C_address,
                        N, M, K);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //======[split K to improve parallelism]================================
        else { 
            int GridZ = (int) fGridZ;
            
            //restrict on GridZ-------------------------------------------------
            int GridZ2 = (K >> 8);//K_slice = 512
            if(GridZ > GridZ2) GridZ = GridZ2;
            
            float coef = K / (K + 4096.0f);
            int GridZ3 = (int) (dev.multiProcessorCount() * matMul_sk_maxBlock_perSM * coef / b0);
            if(GridZ > GridZ3) GridZ = GridZ3;
            
            if(GridZ < 2) GridZ = 2; else GridZ = (GridZ + 3) >> 2 << 2;//padding to 2x
            if(GridZ > matMul_sk_maxPart) GridZ = matMul_sk_maxPart;
            //restrict on GridZ-------------------------------------------------
            
            int part = GridZ - 1;
            int sizeC = N * M;
            long[] block = core.malloc(sizeC * part);
            long Cbuf_address = block[1];
            
            //------[stage1: local matMul]--------------------------------------
            Cuda_matMul.matMulSK(streamArray, length, GridZ,
                    A_address, 
                    B_address, 
                    C_address, Cbuf_address, 
                    N, M, K);
            
            //------[stage2: global reduction]----------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, streamArray.length);
            
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            Cuda_matMul.SKbuf_summary(stream, 
                    Cbuf_address, C_address, 
                    part, sizeC);
            
            Cuda.deleteEvent(event);
            return new SplitKSyncer(core, block, streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMul with bias">
    @Override
    public Syncer matMul_biased(long C_address, 
            long A_address, long B_address,
            int N, int M, int K,
            long Bias_address,//lengthv = C.lengthv = N*M, stride = C.mem_stride = M
            int lengthv, int width) 
    {
        float fGridZ = 1; int b0 = 1;//splitK decision
        if(K >= matMul_sk_threshold) {
            b0 = Cuda_matMul.blockNum(N, M);//matMul: [N, K] * [K, M] = [N, M]
            float expect_block_perSM = matMul_sk_minBlock_perSM + K / (K + 1024.0f);
            if(b0 < dev.multiProcessorCount() * expect_block_perSM) {
                int b1 = Cuda_matMul.blockNum(K, M);//deltaB[K, M] = matMulT1: A^T[K, N] * deltaC[N, M]
                int b2 = Cuda_matMul.blockNum(N, K);//deltaA[N, K] = matMulT2: deltaC[N, M] * B^T[M, K]
                fGridZ = (b1 + b2) / (b0 * 2.0f);
            }
        }
        
        int length = Cuda_matMul.nstream(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        
        //======[no need to split]==============================================
        if(fGridZ < 1.8f) {
            //-----[stage1: Matrix Multiply]------------------------------------
            if (matMul_tf32 && (N > 127) && (M > 127) && (K % 8 == 0)) 
                Cuda_matMul.matMul_mma(streamArray, length, 
                        A_address, B_address, C_address, 
                        N, M, K);
            else /*default method*/
                Cuda_matMul.matMul(streamArray, length, 
                        A_address, B_address, C_address, 
                        N, M, K);
        
            //-----[stage2: add Bias]-------------------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, length);
            Cuda.streamWaitEvent_default(streamArray[0], event);
            Cuda_function.linear_dual2D_row(streamArray[0], C_address,
                    Bias_address, M, 
                    1.0f, 1.0f, 0.0f, 
                    C_address, 
                    lengthv, width, M);
            
            Cuda.deleteEvent(event);
            return new BiasedForwardSyncer(streamPool, streamArray);
        }
        //======[split K to improve parallelism]================================
        else {
            int GridZ = (int) fGridZ;
           
            //restrict on GridZ-------------------------------------------------
            int GridZ2 = (K >> 8);//K_slice = 512
            if(GridZ > GridZ2) GridZ = GridZ2;
            
            float coef = K / (K + 4096.0f);
            int GridZ3 = (int) (dev.multiProcessorCount() * matMul_sk_maxBlock_perSM * coef / b0);
            if(GridZ > GridZ3) GridZ = GridZ3;
            
            if(GridZ < 2) GridZ = 2; else GridZ = (GridZ + 3) >> 2 << 2;//padding to 2x
            if(GridZ > matMul_sk_maxPart) GridZ = matMul_sk_maxPart;
            //restrict on GridZ-------------------------------------------------
            
            int part = GridZ - 1;
            int sizeC = N * M;
            long[] block = core.malloc(sizeC * part);
            long Cbuf_address = block[1];
            
            //------[stage1: local matMulT1]------------------------------------
            Cuda_matMul.matMulSK(streamArray, length, GridZ,
                    A_address, 
                    B_address, 
                    C_address, Cbuf_address, 
                    N, M, K);
            
            //------[stage2: global reduction]----------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, streamArray.length);
            
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            Cuda_matMul.SKbuf_summary(stream, 
                    Cbuf_address, C_address, 
                    part, sizeC);
            
            Cuda.deleteEvent(event);
            
            //-----[stage3: add Bias]-------------------------------------------
            Cuda_function.linear_dual2D_row(stream, C_address,
                    Bias_address, M, 
                    1.0f, 1.0f, 0.0f, 
                    C_address, 
                    lengthv, width, M);
            
            return new SplitKSyncer(core, block, streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT1">
    protected int matMulT1_sk_threshold = 512;
    protected int matMulT1_sk_maxPart = 16;
    
    protected float matMulT1_sk_minBlock_perSM = 1.14f;
    protected float matMulT1_sk_maxBlock_perSM = 4.0f;
    
    public int matMulT1_splitK_threshold() { return matMulT1_sk_threshold; }
    public CudaFloat32EngineBase matMulT1_splitK_threshold(int threshold) {
        if(!Num.isPowerOf2(threshold)) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must be a power of 2", threshold));
        if(threshold <= 0) throw new IllegalArgumentException(String.format(
                "thread { got %d } must > 0", threshold));
        matMulT1_sk_threshold = threshold;
        return this;
    }
    
    public int matMulT1_maxPart() { return matMulT1_sk_maxPart; }
    public CudaFloat32EngineBase matMulT1_maxPart(int maxPart) {
        if(maxPart < 4) throw new IllegalArgumentException(String.format(
                "maxPartNum { got %d } must >= 4", maxPart));
        matMulT1_sk_maxPart = maxPart;
        return this;
    }
    
    public float matMulT1_splitK_minBlock_perSM() { return matMulT1_sk_minBlock_perSM; } 
    public CudaFloat32EngineBase matMulT1_splitK_minBlock_perSM(float minBlock) {
        if(minBlock < 0) throw new IllegalArgumentException(String.format(
                "minBlock { got %f } must >= 1", minBlock));
        matMulT1_sk_minBlock_perSM = minBlock;
        return this;
    }
    
    public float matMulT1_splitK_maxBlock_perSM() { return matMulT1_sk_maxBlock_perSM; } 
    public CudaFloat32EngineBase matMulT1_splitK_maxBlock_perSM(float maxBlock) {
        if(maxBlock < 0) throw new IllegalArgumentException(String.format(
                "maxBlock { got %f } must >= 1", maxBlock));
        matMulT1_sk_maxBlock_perSM = maxBlock;
        return this;
    }
    
    protected boolean matMulT1_tf32 = false;
    public boolean matMulT1_tf32() { return matMulT1_tf32; }
    public CudaFloat32EngineBase matMulT1_tf32(boolean flag) { matMulT1_tf32 = flag; return this; }
    
    @Override
    public Syncer matMulT1(long C_address,
            long A_address, long B_address, 
            int N, int M, int K)//[K, N]^T * [K, M] = [N, K]
    {
        float fGridZ = 1; int b1 = 1;
        if(K >= matMulT1_sk_threshold) {
            b1 = Cuda_matMul.blockNum(N, M);//matMulT1: [K, N]^T * [K, M] = [N, K]
            float expect_block_perSM = matMulT1_sk_minBlock_perSM + K / (K + 1024.0f);
            if(b1 < dev.multiProcessorCount() * expect_block_perSM) {
                int b0 = Cuda_matMul.blockNum(K, N);//deltaA[K, N] = matMulT2: B[K, M] * deltaC^T[M, N]
                int b2 = Cuda_matMul.blockNum(K, M);//deltaB[K, M] = matMul:   A[K, N] * deltaC[N, M]
                fGridZ = (b0 + b2) / (b1 * 2.0f);
            }
        }
        
        int length = Cuda_matMul.nstream(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        
        //======[no need to split]==============================================
        if(fGridZ < 1.8f) {
            if (matMulT1_tf32 && (N > 127) && (M > 127) && (K % 8 == 0)) 
                Cuda_matMul.matMulT1_mma(streamArray, length,
                        A_address, B_address, C_address,
                        N, M, K);
            else /*default method*/
                Cuda_matMul.matMulT1(streamArray, length,
                        A_address, B_address, C_address,
                        N, M, K);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //======[split K to improve parallelism]================================
        else {
            int GridZ = (int) fGridZ;
            
            //restrict on GridZ-------------------------------------------------
            int GridZ2 = (K >> 8);//K_slice = 512
            if(GridZ > GridZ2) GridZ = GridZ2;
            
            float coef = K / (K + 4096.0f);
            int GridZ3 = (int) (dev.multiProcessorCount() * matMulT1_sk_maxBlock_perSM * coef / b1);
            if(GridZ > GridZ3) GridZ = GridZ3;
            
            if(GridZ < 2) GridZ = 2; else GridZ = (GridZ + 3) >> 2 << 2;//pading to 2x
            if(GridZ > matMulT1_sk_maxPart) GridZ = matMulT1_sk_maxPart;
            //restrict on GridZ-------------------------------------------------
            
            int part = GridZ - 1;
            int sizeC = N * M;
            long[] block = core.malloc(sizeC * part);
            long Cbuf_address = block[1];
            
            //------[stage1: local matMulT1]------------------------------------
            Cuda_matMul.matMulT1SK(streamArray, length, GridZ,
                    A_address, 
                    B_address,
                    C_address, Cbuf_address, 
                    N, M, K);
            
            //------[stage2: global reduction]----------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, streamArray.length);
            
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            Cuda_matMul.SKbuf_summary(stream, 
                    Cbuf_address, C_address, 
                    part, sizeC);
            
            Cuda.deleteEvent(event);
            return new SplitKSyncer(core, block, streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT2">
    protected int matMulT2_sk_threshold = 512;
    protected int matMulT2_sk_maxPart = 16;
    
    protected float matMulT2_sk_minBlock_perSM = 1.14f;//64 / 39 = 1.641
    protected float matMulT2_sk_maxBlock_perSM = 4.0f;
    
    public int matMulT2_splitK_threshold() { return matMulT2_sk_threshold; }
    public CudaFloat32EngineBase matMulT2_splitK_threshold(int threshold) {
        if(!Num.isPowerOf2(threshold)) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must be a power of 2", threshold));
        if(threshold <= 0) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must > 0", threshold));
        matMulT2_sk_threshold = threshold;
        return this;
    }
   
    public int matMulT2_splitK_maxPart() { return matMulT2_sk_maxPart; }
    public CudaFloat32EngineBase matMulT2_splitK_maxPart(int maxPart) {
        if(maxPart < 4) throw new IllegalArgumentException(String.format(
                "maxPartNum { got %d } must >= 4", maxPart));
        matMulT2_sk_maxPart = maxPart;
        return this;
    }
   
    public float matMulT2_splitK_minBlock_perSM() { return matMulT2_sk_minBlock_perSM; } 
    public CudaFloat32EngineBase matMulT2_splitK_mainBlock_perSM(int minBlock) {
        if(minBlock < 0) throw new IllegalArgumentException(String.format(
                "minBlock { got %d } must >= 1", minBlock));
        matMulT2_sk_minBlock_perSM = minBlock;
        return this;
    }
    
    public float matMulT2_splitK_maxBlock_perSM() { return matMulT2_sk_maxBlock_perSM; } 
    public CudaFloat32EngineBase matMulT2_splitK_maxBlock_perSM(float maxBlock) {
        if(maxBlock < 0) throw new IllegalArgumentException(String.format(
                "maxBlock { got %f } must >= 1", maxBlock));
        matMulT2_sk_maxBlock_perSM = maxBlock;
        return this;
    }
    
    protected boolean matMulT2_tf32 = false;
    public boolean matMulT2_tf32() { return matMulT2_tf32; }
    public CudaFloat32EngineBase matMulT2_tf32(boolean flag) { matMulT2_tf32 = flag; return this; }
    
    @Override
    public Syncer matMulT2(long C_address,
            long A_address, long B_address, 
            int N, int M, int K)
    {
        float fGridZ = 1; int b2 = 1;//splitK decision
        if(K >= matMulT2_sk_threshold) {
            b2 = Cuda_matMul.blockNum(N, K);//deltaA[N, K] = matMulT2: deltaC[N, M] * B^T[M, K]
            float expect_block_perSM = matMul_sk_minBlock_perSM + K / (K + 1024.0f);
            if(b2 < dev.multiProcessorCount() * expect_block_perSM) {
                int b0 = Cuda_matMul.blockNum(N, M);//matMul: [N, K] * [K, M] = [N, M]
                int b1 = Cuda_matMul.blockNum(K, M);//deltaB[K, M] = matMulT1: A^T[K, N] * deltaC[N, M]
                fGridZ = (b1 + b2) / (b0 * 2.0f);
            }
        }
        
        int length = Cuda_matMul.nstream(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
            
        //======[no need to split]==============================================
        if (fGridZ < 1.8f) { 
            if (matMulT2_tf32 && (N > 127) && (M > 127) && (K % 8 == 0))
                Cuda_matMul.matMulT2_mma(streamArray, length, 
                        A_address, B_address, C_address,
                        N, M, K);
            else /*default method*/
                Cuda_matMul.matMulT2(streamArray, length, 
                        A_address, B_address, C_address,
                        N, M, K);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //======[split K to improve parallelism]================================
        else { 
            int GridZ = (int) fGridZ;
            
            //restrict on GridZ-------------------------------------------------
            int GridZ2 = (K >> 8);//K_slice = 512
            if(GridZ > GridZ2) GridZ = GridZ2;
            
            float coef = K / (K + 4096.0f);
            int GridZ3 = (int) (dev.multiProcessorCount() * matMul_sk_maxBlock_perSM * coef / b2);
            if(GridZ > GridZ3) GridZ = GridZ3;
            
            if(GridZ < 2) GridZ = 2; else GridZ = (GridZ + 3) >> 2 << 2;//padding to 2x
            if(GridZ > matMulT2_sk_maxPart) GridZ = matMulT2_sk_maxPart;
            //restrict on GridZ-------------------------------------------------
            
            int part = GridZ - 1;
            int sizeC = N * M;
            long[] block = core.malloc(sizeC * part);
            long Cbuf_address = block[1];
            
            //------[stage1: local matMulT1]------------------------------------
            Cuda_matMul.matMulT2SK(streamArray, length, GridZ,
                    A_address,
                    B_address,
                    C_address, Cbuf_address,
                    N, M, K);
            
            //------[stage2: global reduction]----------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, streamArray.length);
            
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            Cuda_matMul.SKbuf_summary(stream, 
                    Cbuf_address, C_address, 
                    part, sizeC);
            
            Cuda.deleteEvent(event);
            return new SplitKSyncer(core, block, streamPool, streamArray);
        }
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Batch matMul">
    //<editor-fold defaultstate="collapsed" desc="batchMatMul">
    private boolean batchMatMul_useTexture = true;
    public boolean batchMatMul_useTexture() { return batchMatMul_useTexture; }
    public CudaFloat32EngineBase batchMatMul_useTexture(boolean flag) { batchMatMul_useTexture = flag; return this; }
    
    protected boolean batchMatMul_tf32 = false;
    public boolean batchMatMul_tf32() { return batchMatMul_tf32; }
    public CudaFloat32EngineBase batchMatMul_tf32(boolean flag) { batchMatMul_tf32 = flag; return this; }

    @Override
    public Syncer batchMatMul(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int BK, int AK) 
    {
        int length = Cuda_batchMatMul.streamSize(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        if (batchMatMul_tf32 && (N > 127) && (M > 127) && (BK % 2 == 0) && (BK > 7)) 
            Cuda_batchMatMul.batchMatMul_mma(batchMatMul_useTexture,
                    streamArray, length,
                    A_address, B_address, C_address,
                    Batch, N, M, BK, AK);
        else if (batchMatMul_useTexture)
            Cuda_batchMatMul.batchMatMul_texture(streamArray, length,
                    A_address, B_address, C_address,
                    Batch, N, M, BK, AK);
        else /*default method*/
            Cuda_batchMatMul.batchMatMul(streamArray, length, 
                    A_address, B_address, C_address,
                    Batch, N, M, BK, AK);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT1">
    protected boolean batchMatMulT1_tf32 = false;
    public boolean batchMatMulT1_tf32() { return batchMatMulT1_tf32; }
    public CudaFloat32EngineBase batchMatMulT1_tf32(boolean flag) { batchMatMulT1_tf32 = flag; return this; }
    
    @Override
    public Syncer batchMatMulT1(long C_address, 
            long A_address, long B_address, 
            int Batch, int CN, int AN, int M, int K)
    {
        int length = Cuda_batchMatMul.streamSize(CN, M);
        long[] streamArray = streamPool.getStreamArray(length);
        if (batchMatMulT1_tf32 && (CN > 127) && (M > 127) && (K > 7))
            Cuda_batchMatMul.batchMatMulT1_mma(streamArray, length,
                    A_address, B_address, C_address,
                    Batch, CN, AN, M, K);
        else /*default method*/
            Cuda_batchMatMul.batchMatMulT1(streamArray, length,
                    A_address, B_address, C_address,
                    Batch, CN, AN, M, K);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT2">
    private boolean batchMatMulT2_useTexture = true;
    public boolean batchMatMulT2_useTexture() { return batchMatMulT2_useTexture; }
    public CudaFloat32EngineBase batchMatMulT2_useTexture(boolean flag) { batchMatMulT2_useTexture = flag; return this; }
    
    protected boolean batchMatMulT2_tf32 = false;
    public boolean batchMatMulT2_tf32() { return batchMatMulT2_tf32; }
    public CudaFloat32EngineBase batchMatMulT2_tf32(boolean flag) { batchMatMulT2_tf32 = flag; return this; }
    
    @Override
    public Syncer batchMatMulT2(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int CM, int BM, int K)
    {
        int length = Cuda_batchMatMul.streamSize(N, CM);
        long[] streamArray = streamPool.getStreamArray(length);
        boolean useTexture = batchMatMulT2_useTexture && (N > 47) && (CM > 47);
        
        if (batchMatMulT2_tf32 && (N > 127) && (CM > 127) && (K > 7)) 
            Cuda_batchMatMul.batchMatMulT2_mma(batchMatMulT2_useTexture,
                    streamArray, length,
                    A_address, B_address, C_address,
                    Batch, N, CM, BM, K);
        else if(useTexture) 
            Cuda_batchMatMul.batchMatMulT2_texture(streamArray, length,
                    A_address, B_address, C_address,
                    Batch, N, CM, BM, K);
        else /*default method*/
            Cuda_batchMatMul.batchMatMulT2(streamArray, length,
                    A_address, B_address, C_address, 
                    Batch, N, CM, BM, K);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Convolution3D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: conv3D">
    //<editor-fold defaultstate="collapsed" desc="Configurations">
    protected boolean conv3D_remode = true;
    public boolean conv3D_remode() { return conv3D_remode; }
    public CudaFloat32EngineBase conv3D_remode(boolean flag) { conv3D_remode = flag; return this; }
    
    protected boolean conv3D_Gemm_useTexture = true;
    public boolean conv3D_Gemm_useTexture() { return conv3D_Gemm_useTexture; }
    public CudaFloat32EngineBase conv3D_GEMM_useTexture(boolean flag) { conv3D_Gemm_useTexture = flag; return this; }
    
    protected float conv3D_GemmV2_Q = 1.25f;
    public float conv3D_GemmV2_Q() { return conv3D_GemmV2_Q; }
    public CudaFloat32EngineBase conv3D_GEMMV2_Q(float Q) {
        if(Q < 1.1f) throw new IllegalArgumentException(String.format("Q { got %f } must >= 1.1", Q));
        conv3D_GemmV2_Q = Q;
        return this;
    }
    
    protected float conv3D_GemmR_V2_Q  = 1.03923f;//sqrt(1.08)
    protected float conv3D_GemmR_V2_Q2 = 1.08f; 
    public float conv3D_GemmR_V2_Q() { return conv3D_GemmR_V2_Q; }
    public CudaFloat32EngineBase conv3D_GemmR_V2_Q(float Q) {
        if(Q < 1.03f) throw new IllegalArgumentException(String.format("Q { got %f } must >= 1.03", Q));
        conv3D_GemmR_V2_Q = Q; 
        conv3D_GemmR_V2_Q2 = Q * Q; 
        return this;
    }
    
    protected boolean conv3D_Im2colWinograd_s8 = true;
    public boolean conv3D_Im2colWinograd_s8() { return conv3D_Im2colWinograd_s8; } 
    public CudaFloat32EngineBase conv3D_Im2colWinograd_s8(boolean flag) { conv3D_Im2colWinograd_s8 = flag; return this; }
    
    protected boolean conv3D_Im2colWinograd_s16 = true;
    public boolean conv3D_Im2colWinograd_s16() { return conv3D_Im2colWinograd_s16; } 
    public CudaFloat32EngineBase conv3D_Im2colWinograd_s16(boolean flag) { conv3D_Im2colWinograd_s16 = flag; return this;}
    //</editor-fold>
    
    public static final int CONV3D_GEMMR                = 0;
    public static final int CONV3D_GEMMR_V2             = 1;
    public static final int CONV3D_GEMMR_W1             = 2;
    public static final int CONV3D_IM2COL_WINOGRAD_S8R  = 3;
    public static final int CONV3D_IM2COL_WINOGRAD_S16R = 4;
    //<editor-fold defaultstate="collapsed" desc="Decide Algorithm">
    protected static final float wgrad_gemmc_speed   = 0.96f;
    
    protected static final float wgrad_s4_f3x2_speed = 1.15f;//F(3, 2)
    protected static final float wgrad_s4_f2x3_speed = 1.20f;//F(2, 3)
    protected static final float wgrad_s4_f6x2_speed = 1.20f;//F(6, 2), F_ruse(3, 2)
    protected static final float wgrad_s4_f4x3_speed = 1.30f;//F(4, 3), F_ruse(2, 3)
    
    protected static final float wgrad_s8_f7x2_speed = 1.24f;//F(7, 2)
    protected static final float wgrad_s8_f6x3_speed = 1.55f;//F(6, 3)
    protected static final float wgrad_s8_f5x4_speed = 1.75f;//F(5, 4)
    protected static final float wgrad_s8_f4x5_speed = 1.72f;//F(4, 5)
    protected static final float wgrad_s8_f3x6_speed = 1.50f;//F(3, 6)
    protected static final float wgrad_s8_f2x7_speed = 1.20f;//F(2, 7)
    protected static final float wgrad_s8_f8x5_speed = 1.82f;//F(8, 5), F_ruse(4, 5)
    protected static final float wgrad_s8_f6x6_speed = 1.60f;//F(6, 6), F_ruse(3, 6)
    protected static final float wgrad_s8_f4x7_speed = 1.35f;//F(4, 7), F_ruse(4, 7)

    protected static final float wgrad_s16_fAx7_speed = 1.95f;//F(10, 7)
    protected static final float wgrad_s16_f9x8_speed = 2.15f;//F( 9, 8)
    protected static final float wgrad_s16_f8x9_speed = 2.10f;//F( 8, 9)
    protected static final float wgrad_s16_f7xA_speed = 1.95f;//F(10, 7)
    protected static final float wgrad_s16_f6xB_speed = 1.90f;//F(11, 6)
    protected static final float wgrad_s16_f5xC_speed = 1.85f;//F(12, 5)
    protected static final float wgrad_s16_fIx8_speed = 2.20f;//F(18, 8), F_ruse(9, 8)
    protected static final float wgrad_s16_fGx9_speed = 2.25f;//F(16, 9), F_ruse(8, 9)
    
    protected static final Int2Float WGrad_s8_F7x2_Speed = new Int2Float() {
        private float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) { 
            float q2 = 1.0f * (W % 7) / W, q1 = 1.0f - q2;
            return q1 * wgrad_s8_f7x2_speed + q2 * wgrad_gemmc_speed;
        }
    };
    protected static final Int2Float WGrad_s8_F6x3_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            float speed = 0; for(;;) {
                final int r6 = W  % 6; speed += (W  - r6) * wgrad_s8_f6x3_speed; if(r6 == 0) break;
                final int r4 = r6 & 3; speed += (r6 - r4) * wgrad_s4_f4x3_speed; if(r4 == 0) break;
                final int r2 = r4 & 1; speed += (r4 - r2) * wgrad_s4_f2x3_speed; if(r2 == 0) break;
                speed += r2 * wgrad_gemmc_speed; break; 
            } return speed / W;
        }
    };
    protected static final Int2Float WGrad_s8_F5x4_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            float speed = 0; for(;;) {
                final int r5 = W  % 5; speed += (W  - r5) * wgrad_s8_f5x4_speed; if (r5 == 0) break;
                final int r3 = r5 % 3; speed += (r5 - r3) * wgrad_s4_f3x2_speed; if (r3 == 0) break;
                speed += r3 * wgrad_gemmc_speed; break;
            } return speed / W;
        }
    };
    protected static final Int2Float WGrad_s8_F4x5_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            float q2 = 1.0f * (W & 3) / W, q1 = 1.0f - q2;
            return q1 * wgrad_s8_f4x5_speed + q2 * wgrad_gemmc_speed;
        }
    };
    protected static final Int2Float WGrad_s8_F3x6_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        public float __apply(int W) {
            float speed = 0; for(;;) {
                final int r6 = W  % 6; speed += (W  - r6) * wgrad_s8_f6x6_speed; if (r6 == 0) break;
                final int r3 = r6 % 3; speed += (r6 - r3) * wgrad_s8_f3x6_speed; if (r3 == 0) break;
                final int r2 = r3 & 1; speed += (r3 - r2) * wgrad_s4_f2x3_speed; if (r2 == 0) break;
                speed += r2 * wgrad_gemmc_speed; break;
            } return speed / W;
        }
    };
    protected static final Int2Float WGrad_s8_F2x7_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); } 
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            float speed = 0; for(;;) {
                final int r4 = W  & 3; speed += (W  - r4) * wgrad_s8_f4x7_speed; if (r4 == 0) break;
                final int r2 = r4 & 1; speed += (r4 - r2) * wgrad_s8_f2x7_speed; if (r2 == 0) break;
                speed += r2 * wgrad_gemmc_speed; break;
            } return speed / W;
        }
    };
    protected static final Int2Float[] WGrad_s8_Derailleur = {//idx = FW - 2
        WGrad_s8_F7x2_Speed,//FW = 2
        WGrad_s8_F6x3_Speed,//FW = 3
        WGrad_s8_F5x4_Speed,//FW = 4
        WGrad_s8_F4x5_Speed,//FW = 5
        WGrad_s8_F3x6_Speed,//FW = 6
        WGrad_s8_F2x7_Speed,//FW = 7
    };
    protected static final int wgrad_s8_FW_min = 2; 
    protected static final int wgrad_s8_FW_max = wgrad_s8_FW_min + WGrad_s8_Derailleur.length - 1;
    
    protected static final Int2Float WGrad_S16_FAx7_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            float speed = 0; for(;;) {
                final int rA = W  % 10; speed += (W  - rA) * wgrad_s16_fAx7_speed; if (rA == 0) break;
                final int r4 = rA &  3; speed += (rA - r4) * wgrad_s8_f4x7_speed; if (r4 == 0) break;
                final int r2 = r4 &  1; speed += (r4 - r2) * wgrad_s8_f2x7_speed; if (r2 == 0) break;
                speed += r2 * wgrad_gemmc_speed; break;
            } return speed / W;
        }
    };
    protected static final Int2Float WGrad_S16_F9x8_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) { 
            float speed = 0; for(;;) {
                final int rI = W  % 18; speed += (W  - rI) * wgrad_s16_fIx8_speed; if (rI == 0) break;
                final int r9 = rI %  9; speed += (rI - r9) * wgrad_s16_f9x8_speed; if (r9 == 0) break;
                final int r5 = r9 %  5; speed += (r9 - r5) * wgrad_s8_f5x4_speed; if (r5 == 0) break;
                final int r3 = r5 %  3; speed += (r5 - r3) * wgrad_s4_f3x2_speed; if (r3 == 0) break;
                speed += r3 * wgrad_gemmc_speed; break;
            } return speed / W;
        }
    };
    protected static final Int2Float WGrad_S16_F8x9_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            float speed = 0; for(;;) {
                final int rG = W  & 15; speed += (W  - rG) * wgrad_s16_fGx9_speed; if (rG == 0) break;
                final int r8 = rG &  7; speed += (rG - r8) * wgrad_s16_f8x9_speed; if (r8 == 0) break;
                final int r4 = r8 &  3; speed += (r8 - r4) * wgrad_s4_f4x3_speed; if (r4 == 0) break;
                final int r2 = r4 &  1; speed += (r4 - r2) * wgrad_s4_f2x3_speed; if (r2 == 0) break;
                speed += r2 * wgrad_gemmc_speed; break;
            } return speed / W;
        }
    };
    protected static final Int2Float[] Winograd_s16_Derailleur = {//idx = FW - 5
        WGrad_S16_FAx7_Speed,//FW = 7
        WGrad_S16_F9x8_Speed,//FW = 8
        WGrad_S16_F8x9_Speed,//FW = 9
    };
    protected static final int wgrad_s16_FW_min = 7;
    protected static final int wgrad_s16_FW_max = wgrad_s16_FW_min + Winograd_s16_Derailleur.length - 1;
    
    protected int conv3D_decide_algorithm(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw) 
    {
        if((sh == 1) && (sw == 1)) {
            if((FH == 1) && (FW == 1) && (ph == 0) && (pw == 0)) return CONV3D_GEMMR_W1;
            
            //[speed of GemmRV2]------------------------------------------------
            float GemmRV2 = 0; 
            if ((N > 63) && (OC > 63) && (Cuda_conv3D.GEMM_GM_slice(OC, N*OH*OW, FH*FW*IC) <= 0)) {
                final float psu = Cuda_conv3D.psu_s1(IH, IW, OH, OW, FH, FW);
                GemmRV2 = (float) Math.sqrt(psu) / conv3D_GemmR_V2_Q;
            }
            
            //[speed of Im2colWinograd]-----------------------------------------
            boolean Im2col_Winograd = ((IC & 7) == 0) && (N*OH >= 32);
            
            float Im2col_Winograd_s8 = 0;
            if (conv3D_Im2colWinograd_s8 && Im2col_Winograd && 
                    (FW + OW > 8) && (OC >= 64) && 
                    (FW >= wgrad_s8_FW_min && FW <= wgrad_s8_FW_max) && (FH >= 1 && FH <= 9)) 
                Im2col_Winograd_s8 = WGrad_s8_Derailleur[FW - wgrad_s8_FW_min].apply(OW);
            
            float Im2col_Winograd_s16 = 0; 
            if (conv3D_Im2colWinograd_s16 && Im2col_Winograd &&
                    (FW + OW > 16) && (OC >= 32) && 
                    (FW >= wgrad_s16_FW_min && FW <= wgrad_s16_FW_max) && (FH >= 1 && FH <= 9) &&
                    (pw <= (FW >> 1)) && (OW - IW - pw + ((FW-1) >> 1) <= 0))
                Im2col_Winograd_s16 = Winograd_s16_Derailleur[FW - wgrad_s16_FW_min].apply(OW);
            
            //[decide algorithm]------------------------------------------------
            int Algo = CONV3D_GEMMR; float speed = 1.0f;//default algorithm
            if (speed < Im2col_Winograd_s16) { Algo = CONV3D_IM2COL_WINOGRAD_S16R; speed = Im2col_Winograd_s16; }
            if (speed < Im2col_Winograd_s8) { Algo = CONV3D_IM2COL_WINOGRAD_S8R; speed = Im2col_Winograd_s8; }
            if (speed < GemmRV2) Algo = CONV3D_GEMMR_V2; 
            return Algo;
        }
        else {//down-sampling: stride >= 2
            boolean V2 = (N > 63) && (OC > 63) && 
                    (Cuda_conv3D.psu(IH, IW, OH, OW, FH, FW, sh, sw) > conv3D_GemmR_V2_Q2) && 
                    (Cuda_conv3D.GEMM_GM_slice(OC, N*OH*OW, FH*FW*IC) <= 0);
            return (V2 ? CONV3D_GEMMR_V2 : CONV3D_GEMMR);
        }
    }
    //</editor-fold>
    
    @Override 
    public Syncer conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {
        final int GM = N * OH * OW;
        //<editor-fold defaultstate="collapsed" desc="Kernel Remode Area: W -> CW">
        if(conv3D_remode && (OC > 31) && (GM > 31)) {
            //------[Stage1: remode W -> CW]------------------------------------
            final int GK = FH * FW * IC;
            final int CW_size = GK * OC;//CW[FH, FW, IC, OC]
            final long block[] = core.malloc(CW_size);
            final long CW_address = block[1];
            
            long stream = streamPool.getStream();
            Cuda_conv3D.kernel_remode(stream, W_address, CW_address, FH, FW, OC, IC);
            
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
            
            //------[Stage2: conv3D]--------------------------------------------
            final int algo = conv3D_decide_algorithm(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            final int length;
            if      (algo == CONV3D_IM2COL_WINOGRAD_S8R)  length = Cuda_conv3D.Im2colWinograd_s8_nstream(FW, OH, OW, N, OC);
            else if (algo == CONV3D_IM2COL_WINOGRAD_S16R) length = Cuda_conv3D.Im2colWinograd_s16_nstream(FW, OH, OW, N, OC);
            else if (algo == CONV3D_GEMMR_V2)             length = Cuda_conv3D.GEMMV2_nstream(N, OC);
            else  /*(algo == CONV3D_GEMMR)*/              length = Cuda_conv3D.GEMM_nstream(GM, OC);
            
            long[] streamArray = streamPool.getStreamArray(length);
            Cuda.streamsWaitEvent_default(streamArray, length, event);//wait for stage1
            
            if (algo == CONV3D_GEMMR_W1)
                Cuda_conv3D.CONV3D_GemmR_W1(streamArray, length,
                        X_address, IH, IW, 
                        W_address, CW_address, 
                        Y_address,
                        N, IC, OC);
            else if (algo == CONV3D_IM2COL_WINOGRAD_S8R &&
                Cuda_conv3D.CONV3D_Im2col_Winograd_s8_R_texture(conv3D_Gemm_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW, 
                        Y_address, OH, OW,
                        N, IC, OC, 
                        ph, pw));
            else if (algo == CONV3D_IM2COL_WINOGRAD_S16R && 
                Cuda_conv3D.CONV3D_Im2col_Winograd_s16_R_texture(conv3D_Gemm_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC,
                        ph, pw));
            else if (algo == CONV3D_GEMMR_V2)
                Cuda_conv3D.conv3D_GemmV2R(conv3D_Gemm_useTexture, streamArray, length,
                        X_address, IH, IW,
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            else if (conv3D_Gemm_useTexture) /*(algo == CONV3D_GEMMR)*/
                Cuda_conv3D.CONV3D_GemmR_texture(streamArray, length,
                        X_address, IH, IW,
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            else /*(algo == CONV3D_GEMMR)*/
                Cuda_conv3D.CONV3D_GemmR(streamArray, length,
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);

            Cuda.deleteEvent(event);
            streamArray = Vector.append(stream, streamArray);
            return new StreamArrayBlockSyncer(streamPool, streamArray, core, block);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="Common Area">
        else {
            boolean V2 = (N > 127) && (OC > 127) && //No padding and [FH = FW = 1], V2 = false
                    Cuda_conv3D.psu(IH, IW, OH, OW, FH, FW, sh, sw) >= conv3D_GemmV2_Q;
            int length = (V2 ? Cuda_conv3D.GEMMV2_nstream(N, OC) : Cuda_conv3D.GEMM_nstream(GM, OC));
            long[] streamArray  = streamPool.getStreamArray(length);
            boolean useTexture = conv3D_Gemm_useTexture && (OC > 15);
            
            if ((FH == 1) && (FW == 1)) 
                Cuda_conv3D.conv3D_W1(streamArray, length,
                        X_address, IH, IW,
                        W_address,
                        Y_address,
                        N, IC, OC);
            else if ((ph == 0) && (pw == 0)) 
                Cuda_conv3D.conv3D_np(streamArray, length, 
                    X_address, IH, IW,
                    W_address, FH, FW, 
                    Y_address, OH, OW,
                    N, IC, OC, 
                    sh, sw);
            else if (V2)
                Cuda_conv3D.conv3DV2(useTexture, streamArray, length,
                       X_address, IH, IW,
                       W_address, FH, FW,
                       Y_address, OH, OW,
                       N, IC, OC,
                       sh, sw, ph, pw);
            else if (useTexture)
                Cuda_conv3D.conv3D_texture(streamArray, length,
                       X_address, IH, IW,
                       W_address, FH, FW,
                       Y_address, OH, OW,
                       N, IC, OC,
                       sh, sw, ph, pw);
            else /*Nornal GEMM*/
                Cuda_conv3D.conv3D(streamArray, length,
                       X_address, IH, IW,
                       W_address, FH, FW,
                       Y_address, OH, OW,
                       N, IC, OC,
                       sh, sw, ph, pw);

            return new StreamArraySyncer(streamPool, streamArray);
        }
        //</editor-fold>
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: conv3D with bias">
    @Override 
    public Syncer conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW,         
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw,
            long Bias_address, //stride = OC, lengthv = N*OH*OW*OC
            int lengthv, int width)    
    {
        final int GM = N * OH * OW;
        //<editor-fold defaultstate="collapsed" desc="Kernel Remode Area: W -> CW">
        if(conv3D_remode && (OC > 31) && (GM > 31)) {
            //------[Stage1: remode W -> CW]------------------------------------
            final int GK = FH * FW * IC;
            final int CW_size = GK * OC;//CW[FH, FW, IC, OC]
            final long block[] = core.malloc(CW_size);
            final long CW_address = block[1];
            
            long stream = streamPool.getStream();
            Cuda_conv3D.kernel_remode(stream, W_address, CW_address, FH, FW, OC, IC);
            
            long event1 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event1, stream);
            
            //------[Stage2: conv3D]--------------------------------------------
            final int algo = conv3D_decide_algorithm(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            final int length;
            if      (algo == CONV3D_IM2COL_WINOGRAD_S8R)  length = Cuda_conv3D.Im2colWinograd_s8_nstream(FW, OH, OW, N, OC);
            else if (algo == CONV3D_IM2COL_WINOGRAD_S16R) length = Cuda_conv3D.Im2colWinograd_s8_nstream(FW, OH, OW, N, OC);
            else if (algo == CONV3D_GEMMR_V2)             length = Cuda_conv3D.GEMMV2_nstream(N, OC);
            else  /*(algo == CONV3D_GEMMR)*/              length = Cuda_conv3D.GEMM_nstream(GM, OC);
            
            long[] streamArray = streamPool.getStreamArray(length);
            Cuda.streamsWaitEvent_default(streamArray, length, event1);//wait for stage1
            
            if (algo == CONV3D_GEMMR_W1)
                Cuda_conv3D.CONV3D_GemmR_W1(streamArray, length,
                        X_address, IH, IW,
                        W_address, CW_address,
                        Y_address,
                        N, IC, OC);
            else if (algo == CONV3D_IM2COL_WINOGRAD_S8R && 
                Cuda_conv3D.CONV3D_Im2col_Winograd_s8_R_texture(conv3D_Gemm_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW, 
                        Y_address, OH, OW,
                        N, IC, OC, 
                        ph, pw));
            else if (algo == CONV3D_IM2COL_WINOGRAD_S16R && 
                Cuda_conv3D.CONV3D_Im2col_Winograd_s16_R_texture(conv3D_Gemm_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC,
                        ph, pw));
            else if (algo == CONV3D_GEMMR_V2)
                Cuda_conv3D.conv3D_GemmV2R(conv3D_Gemm_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            else if (conv3D_Gemm_useTexture) /*(algo == CONV3D_GEMMR)*/
                Cuda_conv3D.CONV3D_GemmR_texture(streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            else /*(algo == CONV3D_GEMMR)*/
                Cuda_conv3D.CONV3D_GemmR(streamArray, length,
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            Cuda.deleteEvent(event1);
            
            //------[Stage3: add bias]------------------------------------------
            long event2 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event2, streamArray, length);
            Cuda.streamWaitEvent_default(stream, event2);//wait for stage2
            
            Cuda_function.linear_dual2D_row(stream, Y_address,
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f, 
                    Y_address,
                    lengthv, width, OC);
            
            Cuda.deleteEvent(event2);
            streamArray = Vector.append(stream, streamArray);//sync stream0
            return new BiasedForwardBlockSyncer(streamPool, streamArray, core, block);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="Common Area">
        else {
            boolean V2 = (N > 127) && (OC > 127) && //No padding and [FH = FW = 1], V2 = false
                    (Cuda_conv3D.psu(IH, IW, OH, OW, FH, FW, sh, sw)) >= conv3D_GemmV2_Q;
            int length = (V2 ? Cuda_conv3D.GEMMV2_nstream(N, OC) : Cuda_conv3D.GEMM_nstream(OH, OW, N, OC));
            long[] streamArray = streamPool.getStreamArray(length);
            boolean useTexture = conv3D_Gemm_useTexture && (OC > 15);
            
            //------[Stage1: conv3D]--------------------------------------------
            if ((FH == 1) && (FW == 1))
                Cuda_conv3D.conv3D_W1(streamArray, length,
                        X_address, IH, IW,
                        W_address,
                        Y_address,
                        N, IC, OC);
            else if ((ph == 0) && (pw == 0))//no padding
                Cuda_conv3D.conv3D_np(streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw);
            else if (V2) 
                Cuda_conv3D.conv3DV2(useTexture, streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            else if (useTexture)
                Cuda_conv3D.conv3D_texture(streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            else /*Normal GEMM*/
                Cuda_conv3D.conv3D(streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            
            //------[Stage2: add Bias]------------------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, length);
            Cuda.streamWaitEvent_default(streamArray[0], event);

            Cuda_function.linear_dual2D_row(streamArray[0], Y_address,
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f,
                    Y_address,
                    lengthv, width, OC);

            Cuda.deleteEvent(event);
            return new BiasedForwardSyncer(streamPool, streamArray);
        }
        //</editor-fold>
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: deconv3D with bias">
    @Override
    public Syncer deconv3D_biased(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            long W_address, int FH, int FW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw,
            long Bias_address,//stride = OC, lengthv = N*OH*OW*OC
            int lengthv, int width) 
    {   
        final int algo = conv3D_deltaX_decide_algorithm(IH, IW, OH, OW, FH, FW, N, OC, IC, sh, sw, ph, pw);
        //<editor-fold defaultstate="collapsed" desc="crossAdd kernel: when OC is small">
        if(algo == DECONV3D_DX_CROSS_ADD) {
            //------[Stage1: deconv3D]------------------------------------------
            final int length = Cuda_dconv3D_deltaX.CrossAdd_nstream(IH, IW, N, IC);
            long[] streamArray = streamPool.getStreamArray(length);
            
            Cuda_dconv3D_deltaX.DCONV3D_deltaX_crossAdd(streamArray, length, 
                    X_address, IH, IW,
                    W_address, FH, FW,
                    Y_address, OH, OW,
                    N, OC, IC,
                    sh, sw, ph, pw);
           
            //------[Stage2: add bias]------------------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, length);
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            
            Cuda_function.linear_dual2D_row(stream, Y_address, 
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f, 
                    Y_address, 
                    lengthv, width, OC);
            
            Cuda.deleteEvent(event);
            return new BiasedForwardSyncer(streamPool, streamArray);
        }
         //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="dense kernel: sh = sw = 1">
        else if ((sh == 1) && (sw == 1)) {
            final int length;
            if      (algo == DECONV3D_DX_IM2COL_WINOGRAD_S8)  length = Cuda_dconv3D_deltaX.Im2colWinograd_s8_nstream(FW, OH, OW, N, OC);
            else if (algo == DECONV3D_DX_IM2COL_WINOGRAD_S16) length = Cuda_dconv3D_deltaX.Im2colWinograd_s16_nstream(FW, OH, OW, N, OC);
            else if (algo == DECONV3D_DX_ZERO_PADDING_S1_V2)  length = Cuda_dconv3D_deltaX.GEMMV2_nstream_s1(N, OC);
            else  /*(algo == DECONV3D_DX_ZERO_PADDING_S1)*/   length = Cuda_dconv3D_deltaX.GEMM_nstream_s1(OH, OW, N, OC);
            
           //------[Stage1: deconv3D]------------------------------------------
            long[] streamArray = streamPool.getStreamArray(length);
            boolean useTexture = conv3D_dX_s1_useTexture && (OC > 15);
            
            if (algo == DECONV3D_DX_ZERO_PADDING_W1) 
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_W1(streamArray, length, 
                        X_address, 
                        W_address, 
                        Y_address, OH, OW,
                        N, OC, IC);
            else if (algo == DECONV3D_DX_IM2COL_WINOGRAD_S8 && 
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_Im2col_Winograd_s8_texture(useTexture, streamArray, length,
                        X_address, IH, IW, 
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, OC, IC,
                        ph, pw));
            else if (algo == DECONV3D_DX_IM2COL_WINOGRAD_S16 &&
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_Im2col_Winograd_s16_texture(useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, FH, FW, 
                        Y_address, OH, OW,
                        N, OC, IC, 
                        ph, pw));
            else if (algo == DECONV3D_DX_ZERO_PADDING_S1_V2) 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_V2_s1(useTexture, streamArray, length,
                        X_address, IH, IW, 
                        W_address, FH, FW, 
                        Y_address, OH, OW, 
                        N, OC, IC, 
                        ph, pw);
            else if (useTexture) 
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_s1_texture(streamArray, length, 
                        X_address, IH, IW, 
                        W_address, FH, FW, 
                        Y_address, OH, OW, 
                        N, OC, IC,
                        ph, pw);
            else /*(algo == DECONV3D_DX_ZERO_PADDING_S1)*/
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_s1(streamArray, length, 
                        X_address, IH, IW, 
                        W_address, FH, FW, 
                        Y_address, OH, OW, 
                        N, OC, IC,
                        ph, pw);
            
            //------[Stage2: add bias]------------------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, length);
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            
            Cuda_function.linear_dual2D_row(stream, Y_address, 
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f, 
                    Y_address, 
                    lengthv, width, OC);
            
            Cuda.deleteEvent(event);
            return new BiasedForwardSyncer(streamPool, streamArray);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="sparse kernel: sh * sw >= 2">
        else {
            //------[Stage1: remode W -> CW]-------------------------------------
            final int CFH = (FH + sh - 1) / sh;
            final int CFW = (FW + sw - 1) / sw;
            final int CW_size = (IC * CFH * CFW * OC * sh * sw);//CW[sh, sw, CFH, CFW, OC, IC]
            final long block[] = core.malloc(CW_size);
            final long CW_address = block[1];
            
            long stream = streamPool.getStream();
            Cuda_dconv3D_deltaX.ks_remodev2(stream, 
                    W_address, FH, FW,
                    CW_address, CFH, CFW,
                    IC, OC, sh, sw);
            
            long event1 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event1, stream);//streamArray wait stream
            
            //------[Stage2: deconv3D]------------------------------------------
            final int length = (algo == DECONV3D_DX_KERNEL_SPLIT_IMS2R_V2 ? 
                    Cuda_dconv3D_deltaX.GEMMV2_nstream_s1(N, OC): 
                    Cuda_dconv3D_deltaX.KernelSplit_nstream(OH, OW, N, OC, sh, sw));
            long[] streamArray = streamPool.getStreamArray(length);
            Cuda.streamsWaitEvent_default(streamArray, length, event1);
            
            boolean useTexture = conv3D_dX_ks_useTexture && (OC > 15);
            if (algo == DECONV3D_DX_KERNEL_SPLIT_IMS2R_V2) 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksV2_Ims2R(useTexture, streamArray, length,
                        X_address, IH, IW,
                        CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, OC, IC,
                        ph, pw);
            else if (algo == DECONV3D_DX_KERNEL_SPLIT_IMS2R) {
                if(useTexture) 
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksIms2R_texture(streamArray, length,
                            X_address, IH, IW,
                            CW_address, FH, FW,
                            Y_address, OH, OW,
                            N, OC, IC,
                            ph, pw);
                else 
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksIms2R(streamArray, length,
                            X_address, IH, IW,
                            CW_address, FH, FW,
                            Y_address, OH, OW,
                            N, OC, IC,
                            ph, pw);
            }
            else if (algo == DECONV3D_DX_KERNEL_SPLIT_IMSR) {
                if (useTexture)
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksImsR_texture(streamArray, length,
                            X_address, IH, IW,
                            CW_address, FH, FW,
                            Y_address, OH, OW,
                            N, OC, IC,
                            sh, sw, ph, pw);
                else 
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksImsR(streamArray, length,
                            X_address, IH, IW,
                            CW_address, FH, FW,
                            Y_address, OH, OW,
                            N, OC, IC,
                            sh, sw, ph, pw);
            }
            else /*algo == DECONV3D_DX_KERNEL_SPLIT*/
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_kernelSplit(streamArray, length,
                        X_address, IH, IW,
                        CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, OC, IC,
                        sh, sw, ph, pw);
            
            Cuda.deleteEvent(event1);
             
            //------[Stage3: add bias]------------------------------------------
            long event2 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event2, streamArray, length);
            Cuda.streamWaitEvent_default(stream, event2);
            
            Cuda_function.linear_dual2D_row(stream, Y_address, 
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f, 
                    Y_address, 
                    lengthv, width, OC);
            
            Cuda.deleteEvent(event2);
            streamArray = Vector.append(stream, streamArray);//sync stream0
            return new BiasedForwardBlockSyncer(streamPool, streamArray, core, block);    
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: conv3D_deltaW"> 
    //<editor-fold defaultstate="collapsed" desc="Configurations: common"> 
    protected int conv3D_dW_sk_threshold = 512;
    public int conv3D_deltaW_SplitK_threshold() { return conv3D_dW_sk_threshold; }
    public CudaFloat32EngineBase conv3D_deltaW_SplitK_threshold(int threshold) {
        if(!Num.isPowerOf2(threshold)) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must be a power of 2", threshold));
        if(threshold < 256) throw new IllegalArgumentException(String.format(
                "threshold { got %d } must >= 256", threshold));
        conv3D_dW_sk_threshold = threshold;
        return this;
    }
    
    protected float conv3D_dW_GemmV2SK_Q  = 1.11803f;//sqrt(1.25f)
    protected float conv3D_dW_GemmV2SK_Q2 = 1.25f;
    public float conv3D_deltaW_GemmV2SK_Q() { return conv3D_dW_GemmV2SK_Q; }
    public CudaFloat32EngineBase conv3D_deltaW_GemmV2SK_Q(float Q) {
        if(Q < 1.05f) throw new IllegalArgumentException(String.format("Q { got %f } must >= 1.05", Q));
        conv3D_dW_GemmV2SK_Q  = Q;
        conv3D_dW_GemmV2SK_Q2 = Q * Q;
        return this;
    }
    
    protected boolean conv3D_dW_WinogradV2_SHW_s4 = true;
    public boolean conv3D_deltaW_WinogradV2_SHW_s4() { return conv3D_dW_WinogradV2_SHW_s4; }
    public CudaFloat32EngineBase conv3D_deltaW_WinogradV2_SHW_s4(boolean flag) { conv3D_dW_WinogradV2_SHW_s4 = flag; return this; }
    
    protected boolean conv3D_dW_WinogradV2_SHW_s8 = true;
    public boolean conv3D_deltaW_WinogradV2_SHW_s8() { return conv3D_dW_WinogradV2_SHW_s8; }
    public CudaFloat32EngineBase conv3D_deltaW_WinogradV2_SHW_s8(boolean flag) { conv3D_dW_WinogradV2_SHW_s8 = flag; return this; }
    
    protected boolean conv3D_dW_WinogradV2_SHW_s16 = true;
    public boolean conv3D_deltaW_WinogradV2_SHW_s16() { return conv3D_dW_WinogradV2_SHW_s16; }
    public CudaFloat32EngineBase conv3D_deltaW_WinogradV2_SHW_s16(boolean flag) { conv3D_dW_WinogradV2_SHW_s16 = flag; return this; }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Configurations: GemmSK"> 
    protected int conv3D_dW_GemmSK_maxPart = 16;
    public int conv3D_deltaW_GemmSK_maxPart() { return conv3D_dW_GemmSK_maxPart; } 
    public CudaFloat32EngineBase conv3D_deltaW_GemmSK_maxPart(int maxPart) {
        if(maxPart < 4) throw new IllegalArgumentException(String.format(
                "maxPart { got %d } must >= 4", maxPart));
        conv3D_dW_GemmSK_maxPart = maxPart;
        return this;
    }
    
    protected float conv3D_dW_GemmSK_minBlock_perSM = 1.5f;
    public float conv3D_deltaW_GemmSK_minBlock_perSM() { return conv3D_dW_GemmSK_minBlock_perSM; } 
    public CudaFloat32EngineBase conv3D_deltaW_GemmSK_minBlock_perSM(float minBlock) {
        if(minBlock < 0) throw new IllegalArgumentException(String.format(
                "minBlock { got %f } must >= 1", minBlock));
        conv3D_dW_GemmSK_minBlock_perSM = minBlock;
        return this;
    }
    
    protected float conv3D_dW_GemmSK_maxBlock_perSM = 4.0f;
    public float conv3D_deltaW_GemmSK_maxBlock_perSM() { return conv3D_dW_GemmSK_maxBlock_perSM; } 
    public CudaFloat32EngineBase conv3D_deltaW_GemmSK_maxBlock_perSM(float maxBlock) {
        if(maxBlock < 0) throw new IllegalArgumentException(String.format(
                "maxBlock { got %f } must >= 1", maxBlock));
        conv3D_dW_GemmSK_maxBlock_perSM = maxBlock;
        return this;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Configurations: WinogradSHW"> 
    protected int conv3D_dW_WinogradSHW_maxPart = 16;
    public int conv3D_deltaW_WinogradSHW_maxPart() { return conv3D_dW_WinogradSHW_maxPart; } 
    public CudaFloat32EngineBase conv3D_deltaW_WinogradSHW_maxPart(int maxPart) {
        if(maxPart < 4) throw new IllegalArgumentException(String.format(
                "maxPart { got %d } must >= 4", maxPart));
        conv3D_dW_WinogradSHW_maxPart = maxPart;
        return this;
    }
    
    protected float conv3D_dW_WinogradSHW_minBlock_perSM = 1.5f;
    public float conv3D_deltaW_WinogradSHW_minBlock_perSM() { return conv3D_dW_WinogradSHW_minBlock_perSM; } 
    public CudaFloat32EngineBase conv3D_deltaW_WinogradSHW_minBlock_perSM(float minBlock) {
        if(minBlock < 0) throw new IllegalArgumentException(String.format(
                "minBlock { got %f } must >= 1", minBlock));
        conv3D_dW_WinogradSHW_minBlock_perSM = minBlock;
        return this;
    }
    
    protected float conv3D_dW_WinogradSHW_maxBlock_perSM = 4.0f;
    public float conv3D_deltaW_WinogradSHW_maxBlock_perSM() { return conv3D_dW_WinogradSHW_maxBlock_perSM; } 
    public CudaFloat32EngineBase conv3D_deltaW_WinogradSHW_maxBlock_perSM(float maxBlock) {
        if(maxBlock < 0) throw new IllegalArgumentException(String.format(
                "maxBlock { got %f } must >= 1", maxBlock));
        conv3D_dW_WinogradSHW_maxBlock_perSM = maxBlock;
        return this;
    }
    //</editor-fold>
    
    public static final int DECONV3D_DW_GEMMSK              = 0;
    public static final int DECONV3D_DW_GEMMSK_V2           = 1;
    public static final int DECONV3D_DW_GEMMSK_W1           = 2;
    public static final int DECONV3D_DW_WINOGRAD_V2_SHW_S4  = 3;
    public static final int DECONV3D_DW_WINOGRAD_V2_SHW_S8  = 4;
    public static final int DECONV3D_DW_WINOGRAD_V2_SHW_S16 = 5;
    //<editor-fold defaultstate="collapsed" desc="Decide Algorithm">
    static final int dc3dW_s4_nbase  = 28;//cache block size = 64 * 64, 2 + 28 = 39, 30 / 10 = 3 
    static final int[] dc3dW_s4_r = {//r = dc3dW_s4_r[n - 2]
        3,//n = 2, r = 3, F(2, 3), algo = 2 + 28 = 30
        2,//n = 3, r = 2, F(3, 2), algo = 2 + 28 = 31
    };
    
    static final int dc3dW_s8_nbase  = 38;//cache block size = 64 * 32, 2 + 38 = 40, 40 / 10 = 4
    static final int[] dc3dW_s8_r = {//r = dc3dW_s8_r[n - 3]
        6,//n = 3, r = 6, F(3, 6), algo = 3 + 38 = 41
        5,//n = 4, r = 5, F(4, 5), algo = 4 + 38 = 42
        4,//n = 5, r = 4, F(5, 4), algo = 5 + 38 = 43
        6,//n = 6, r = 6, F(6, 6), algo = 6 + 38 = 44
        2,//n = 7, r = 2, F(7, 2), algo = 7 + 38 = 45
        5,//n = 8, r = 5, F(8, 5), algo = 8 + 38 = 46
    };
    
    static final int dc3dW_s16_nbase = 45;//cache block size = 32 * 32, 5 + 45 = 50, 50 / 10 = 5
    static final int[] dc3dW_s16_r = {//r = dc3dW_s8_r[n - 7]
        0xC,//n = 5, r = 12, F(5, 12), algo = 5 + 45 = 50    
        0xB,//n = 6, r = 11, F(6, 11), algo = 6 + 45 = 51
        0xA,//n = 7, r = 10, F(7, 10), algo = 7 + 45 = 52
        0x9,//n = 8, r =  9, F(8,  9), algo = 8 + 45 = 53
        0x8,//n = 9, r =  8, F(9,  8), algo = 9 + 45 = 54
    };
  
    protected static final Int2Float WGradSHW_s4_F2x3_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r3 = W % 3; 
            float speed = (W - r3) * wgrad_s4_f2x3_speed; 
            if (r3 != 0) speed += r3 * wgrad_gemmc_speed;
            return speed / W;
        }
    };
    protected static final Int2Float[] WGradSHW_s4_Derailleur = {//idx = n - 2, algo = n + base
        WGradSHW_s4_F2x3_Speed,//FW % 2 == 0, n = 2
        null,                  //FW % 3 == 0, n = 3
    };
    protected static final int wgradSHW_s4_FW_min = 2; 
    protected static final int wgradSHW_s4_FW_max = wgradSHW_s4_FW_min + WGradSHW_s4_Derailleur.length - 1;
    
    protected static final Int2Float WGradSHW_s8_F3x6_Speed = new Int2Float() {//FW % 3 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r6 = W % 6; 
            float speed = (W - r6) * wgrad_s8_f3x6_speed;
            if (r6 != 0) {
                if((r6 & 1) == 0) speed += r6 * wgrad_s4_f3x2_speed;
                else speed += r6 * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s8_F4x5_Speed = new Int2Float() {//FW % 4 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r5 = W % 5; 
            float speed = (W - r5) * wgrad_s8_f4x5_speed; 
            if (r5 != 0) {
                if(r5 % 3 == 0) speed += r5 * wgrad_s4_f4x3_speed;
                else speed += r5 * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s8_F5x4_Speed = new Int2Float() {//FW % 5 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r4 = W & 3; 
            float speed = (W - r4) * wgrad_s8_f5x4_speed; 
            if (r4 != 0) speed += r4 * wgrad_gemmc_speed;
            return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s8_F6x6_Speed = new Int2Float() {//FW % 6 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r6 = W % 6; 
            float speed = (W - r6) * wgrad_s8_f6x6_speed;
            if(r6 != 0) {
                if      (r6 % 3  == 0) speed += r6 * wgrad_s8_f6x3_speed;
                else if((r6 & 1) == 0) speed += r6 * wgrad_s4_f3x2_speed;
                else speed += r6 * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s8_F7x2_Speed = new Int2Float() {//FW % 7 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++)  cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r2 = W & 1; 
            float speed = (W - r2) * wgrad_s8_f7x2_speed; 
            if (r2 != 0) speed += r2 * wgrad_gemmc_speed;
            return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s8_F8x5_Speed = new Int2Float() {//FW % 8 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r5 = W % 5; 
            float speed = (W - r5) * wgrad_s8_f8x5_speed; 
            if (r5 != 0) {
                if(r5 % 3 == 0) speed += r5 * wgrad_s4_f4x3_speed;
                else speed += r5 * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float[] WGradSHW_s8_Derailleur = {//idx = n - 3, algo = n + base
        WGradSHW_s8_F3x6_Speed,//FW % 3 == 0, n = 3
        WGradSHW_s8_F4x5_Speed,//FW % 4 == 0, n = 4
        WGradSHW_s8_F5x4_Speed,//FW % 5 == 0, n = 5
        WGradSHW_s8_F6x6_Speed,//FW % 6 == 0, n = 6
        WGradSHW_s8_F7x2_Speed,//FW % 7 == 0, n = 7
        WGradSHW_s8_F8x5_Speed,//FW % 8 == 0, n = 8
    };
    protected static final int wgradSHW_s8_FW_min = 3; 
    protected static final int wgradSHW_s8_FW_max = wgradSHW_s8_FW_min + WGradSHW_s8_Derailleur.length - 1;
   
    protected static final Int2Float WGradSHW_s16_F5xC_Speed = new Int2Float() {//FW % 5 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int rC = W % 12; 
            float speed = (W - rC) * wgrad_s16_f5xC_speed; 
            if (rC != 0) {
                if((rC & 3) == 0) speed += rC * wgrad_s8_f5x4_speed;
                else speed += rC * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s16_F6xB_Speed = new Int2Float() {//FW % 6 == 0
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int rB = W % 11; 
            float speed = (W - rB) * wgrad_s16_f6xB_speed; 
            if (rB != 0) {
                if       (rB % 6  == 0) speed += rB * wgrad_s8_f6x6_speed;
                else if  (rB % 3  == 0) speed += rB * wgrad_s8_f6x3_speed;
                else if ((rB & 1) == 0) speed += rB * wgrad_s4_f3x2_speed;
                else speed += rB * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s16_F7xA_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int rA = W % 10; 
            float speed = (W - rA) * wgrad_s16_f7xA_speed; 
            if (rA != 0) {
                if ((rA & 1) == 0) speed += rA * wgrad_s8_f7x2_speed;
                else speed += rA * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s16_F8x9_Speed = new Int2Float() {
        float[] cc = new float[256]; { for(int i=0; i<cc.length; i++) cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r9 = W % 9; 
            float speed = (W - r9) * wgrad_s16_f8x9_speed; 
            if (r9 != 0) {
                if      (r9 % 5 == 0) speed += r9 * wgrad_s8_f8x5_speed;
                else if (r9 % 7 == 0) speed += r9 * wgrad_s8_f4x7_speed;
                else if (r9 % 3 == 0) speed += r9 * wgrad_s4_f4x3_speed;
                else speed += r9 * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float WGradSHW_s16_F9x8_Speed = new Int2Float() {
       float[] cc = new float[256]; { for(int i=0; i<cc.length; i++)  cc[i] = __apply(i + 1); }
        @Override public float apply(int W) { return (W <= cc.length ? cc[W - 1] : __apply(W)); }
        float __apply(int W) {
            final int r8 = W & 7; 
            float speed = (W - r8) * wgrad_s16_f9x8_speed; 
            if (r8 != 0) {
                if       (r8 % 6  == 0) speed += r8 * wgrad_s8_f3x6_speed;
                else if ((r8 & 1) == 0) speed += r8 * wgrad_s4_f3x2_speed;
                else speed += r8 * wgrad_gemmc_speed;
            } return speed / W;
        }
    };
    protected static final Int2Float[] WGradSHW_s16_Derailleur = {//idx = n - 5, algo = n + base
        WGradSHW_s16_F5xC_Speed,//FW % 5 == 0, n = 5
        WGradSHW_s16_F6xB_Speed,//FW % 6 == 0, n = 6
        WGradSHW_s16_F7xA_Speed,//FW % 7 == 0, n = 7
        WGradSHW_s16_F8x9_Speed,//FW % 8 == 0, n = 8
        WGradSHW_s16_F9x8_Speed,//FW % 9 == 0, n = 9
    };
    protected static final int wgradSHW_s16_FW_min = 5; 
    protected static final int wgradSHW_s16_FW_max = wgradSHW_s16_FW_min + WGradSHW_s16_Derailleur.length - 1;
    
    protected int conv3D_deltaW_decide_algorithm(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if((sh == 1) && (sw == 1)) {
            if ((FH == 1) && (FW == 1) && (ph == 0) && (pw == 0)) return DECONV3D_DW_GEMMSK_W1;
            
            //[speed of GemmV2]-------------------------------------------------
            final float psu = Cuda_dconv3D_deltaW.psu_s1(IH, IW, OH, OW, FH, FW);
            final float psu2 = (float) Math.sqrt(psu), psu4 =(float) Math.sqrt(psu2); 
            float GemmV2 = 0; if((OC >= 64) && (IC >= 64)) GemmV2 = psu2 / conv3D_dW_GemmV2SK_Q;
            
            //[speed of Winograd]-----------------------------------------------
            boolean winograd = ((N & 7) == 0) && (IC >= 32);
            
            float winograd_s16 = 0; int Algo_Winograd_s16 = -1;
            if(conv3D_dW_WinogradV2_SHW_s16 && winograd && (OC >= 32) && (OW >= 8)) { int n = 0;
                if      ((FW % 9 == 0) && (OW >=  8) && (pw <= 4) && (FW - IW + OW - pw <= 5)) n = 9;//F(9,  8)
                else if ((FW % 8 == 0) && (OW >=  9) && (pw <= 4) && (FW - IW + OW - pw <= 5)) n = 8;//F(8,  9)
                else if ((FW % 7 == 0) && (OW >= 10) && (pw <= 3) && (FW - IW + OW - pw <= 4)) n = 7;//F(7, 10)
                else if ((FW % 6 == 0) && (OW >= 11) && (pw <= 3) && (FW - IW + OW - pw <= 4) && (OC >= 64)) n = 6;//F(6, 11)
                else if ((FW % 5 == 0) && (OW >= 12) && (pw <= 2) && (FW - IW + OW - pw <= 3) && (OC >= 64)) n = 5;//F(5, 12)
                if (n != 0) { 
                    winograd_s16 = WGradSHW_s16_Derailleur[n - wgradSHW_s16_FW_min].apply(OW) * psu4;
                    Algo_Winograd_s16 = n + dc3dW_s16_nbase;
                }
            }
            
            float Winograd_s8 = 0; int Algo_Winograd_s8 = -1;
            if(conv3D_dW_WinogradV2_SHW_s8 && winograd && (OC >= 64) && (OW >= 4)) { int n = 0;
                if      ((FW % 8 == 0) && (OW >= 5)) n = 8;//F(8, 5)
                else if ((FW % 5 == 0) && (OW >= 4)) n = 5;//F(5, 4)
                else if ((FW % 4 == 0) && (OW >= 5)) n = 4;//F(4, 5)
                else if ((FW % 6 == 0) && (OW >= 6)) n = 6;//F(6, 6)
                else if ((FW % 3 == 0) && (OW >= 6)) n = 3;//F(3, 6)
                else if ((FW % 7 == 0) && (OW >= 2)) n = 7;//F(7, 2)
                if (n != 0) { 
                    Winograd_s8 = WGradSHW_s8_Derailleur[n - wgradSHW_s8_FW_min].apply(OW) * psu4;
                    Algo_Winograd_s8 = n + dc3dW_s8_nbase;
                }
            }
            
            float Winograd_s4 = 0; int Algo_Winograd_s4 = -1;
            if(conv3D_dW_WinogradV2_SHW_s4 && winograd && (OC >= 64) && (IC >= 64) && (OW >= 3)) { int n = 0;
                if      ((FW % 2 == 0) && (OW >= 3)) n = 2;//F(2, 3)
               //else if ((FW % 3 == 0) && (OW >= 2)) { n = 3; r = 2; OWr = OW % 3; }//F(3, 2)
                if (n != 0) {
                    Winograd_s4 = WGradSHW_s4_Derailleur[n - wgradSHW_s4_FW_min].apply(OW) * psu4;
                    Algo_Winograd_s4 = n + dc3dW_s4_nbase; 
                }
            }
            
            int Algo = DECONV3D_DW_GEMMSK; float speed = 1.0f;//default algorithm
            if (speed < winograd_s16) { Algo = Algo_Winograd_s16; speed = winograd_s16; }
            if (speed < Winograd_s8) { Algo = Algo_Winograd_s8; speed = Winograd_s8; }
            if (speed < Winograd_s4) { Algo = Algo_Winograd_s4; speed = Winograd_s4; }
            if (speed < GemmV2) Algo = DECONV3D_DW_GEMMSK_V2;
            return Algo;
        }
        else {//down-sampling: stride >= 2
            boolean V2 = (OC > 63) && (IC > 63) && //No padding and [FH = FW = 1], V2 = false
                    Cuda_dconv3D_deltaW.psu(IH, IW, OH, OW, FH, FW, sh, sw) >= conv3D_dW_GemmV2SK_Q;
            return (V2 ? DECONV3D_DW_GEMMSK_V2 : DECONV3D_DW_GEMMSK);
        }
    }
    
    protected int conv3D_deltaW_decide_algorithm_s8(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if((sh == 1) && (sw == 1)) {
            if ((FH == 1) && (FW == 1) && (ph == 0) && (pw == 0)) return DECONV3D_DW_GEMMSK_W1;
            
            //[speed of GemmV2]-------------------------------------------------
            final float psu = Cuda_dconv3D_deltaW.psu_s1(IH, IW, OH, OW, FH, FW);
            final float psu2 = (float) Math.sqrt(psu), psu4 =(float) Math.sqrt(psu2); 
            float GemmV2 = 0; if((OC >= 64) && (IC >= 64)) GemmV2 = psu2 / conv3D_dW_GemmV2SK_Q;
            
            //[speed of Winograd]-----------------------------------------------
            boolean Winograd = ((N & 7) == 0) && (IC >= 32);
            
            float Winograd_s8 = 0; int Algo_Winograd_s8 = -1;
            if(conv3D_dW_WinogradV2_SHW_s8 && Winograd && (OC >= 64) && (OW >= 4)) { int n = 0;
                if      ((FW % 8 == 0) && (OW >= 5)) n = 8;//F(8, 5)
                else if ((FW % 5 == 0) && (OW >= 4)) n = 5;//F(5, 4)
                else if ((FW % 4 == 0) && (OW >= 5)) n = 4;//F(4, 5)
                else if ((FW % 6 == 0) && (OW >= 6)) n = 6;//F(6, 6)
                else if ((FW % 3 == 0) && (OW >= 6)) n = 3;//F(3, 6)
                else if ((FW % 7 == 0) && (OW >= 2)) n = 7;//F(7, 2)
                if (n != 0) { 
                    Winograd_s8 = WGradSHW_s8_Derailleur[n - wgradSHW_s8_FW_min].apply(OW) * psu4;
                    Algo_Winograd_s8 = n + dc3dW_s8_nbase;
                }
            }
            
            int Algo = DECONV3D_DW_GEMMSK; float speed = 1.0f;//default algorithm
            if (speed < Winograd_s8) { Algo = Algo_Winograd_s8; speed = Winograd_s8; }
            if (speed < GemmV2) Algo = DECONV3D_DW_GEMMSK_V2;
            return Algo;
        }
        else {//down-sampling: stride >= 2
            boolean V2 = (OC > 63) && (IC > 63) && //No padding and [FH = FW = 1], V2 = false
                    Cuda_dconv3D_deltaW.psu(IH, IW, OH, OW, FH, FW, sh, sw) >= conv3D_dW_GemmV2SK_Q;
            return (V2 ? DECONV3D_DW_GEMMSK_V2 : DECONV3D_DW_GEMMSK);
        }
    }
    
    protected int conv3D_deltaW_decide_algorithm_GEMM(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int IC, int OC,
            int sh, int sw)
    {
        boolean V2 = (OC > 63) && (IC > 63) && //No padding and [FH = FW = 1], V2 = false
                Cuda_dconv3D_deltaW.psu(IH, IW, OH, OW, FH, FW, sh, sw) >= conv3D_dW_GemmV2SK_Q;
        return (V2 ? DECONV3D_DW_GEMMSK_V2: DECONV3D_DW_GEMMSK);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="GridZ: GemmSK">
    protected int conv3D_dW_GemmSK_GridZ(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int N, int IC, int OC, int GK)
    {
        final int SM_count = dev.multiProcessorCount();
        float expect_block_perSM = conv3D_dW_GemmSK_minBlock_perSM * (GK / (GK + 1024.0f));
        
        final int b2 = Cuda_dconv3D_deltaW.GEMM_nblock(FH, FW, OC, IC);
        if(b2 >= SM_count * expect_block_perSM) return 1;//enough parallelism
        
        //split tasks to improve parallelism------------------------------------
        final int b0 = Cuda_conv3D.GEMM_nblock(OH, OW, N, OC);
        final int b1 = Cuda_dconv3D_deltaX.GEMM_nblock(IH, IW, N, IC);
        int GZ = (b0 + b1) / (b2 << 1);
                    
        final int GZ2 = GK >> 8; if(GZ > GZ2) GZ = GZ2;//min_GK_slice = 512
        final int GZ3 = (int) (SM_count * conv3D_dW_GemmSK_maxBlock_perSM / b2); if(GZ > GZ3) GZ = GZ3;
                
        if(GZ < 2) GZ = 2; else GZ = (GZ + 3) >> 2 << 2;//padding to 2x
        if(GZ > conv3D_dW_GemmSK_maxPart) GZ = conv3D_dW_GemmSK_maxPart;
        return GZ;
    }
    
    protected int conv3D_dW_GemmSKV2_GridZ(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int N, int IC, int OC, int GK)
    {
        final int SM_count = dev.multiProcessorCount();
        float expect_block_perSM = conv3D_dW_GemmSK_minBlock_perSM * (GK / (GK + 1024.0f));
        
        final int b2 = Cuda_dconv3D_deltaW.GEMMV2_nblock(FH, FW, OC, IC);
        if (b2 >= SM_count * expect_block_perSM) return 1;//enough parallelism
        
        //split tasks to improve parallelism------------------------------------
        final int b0 = Cuda_conv3D.GEMMV2_nblock(OH, OW, N, OC);
        final int b1 = Cuda_dconv3D_deltaX.GEMMV2_nblock(IH, IW, N, IC);
        int GZ = (b0 + b1) / (b2 << 1);
                    
        final int GZ2 = GK >> 8; if(GZ > GZ2) GZ = GZ2;//min GK_slice = 512
        final int GZ3 = (int) (SM_count * conv3D_dW_GemmSK_maxBlock_perSM / b2);  if(GZ > GZ3) GZ = GZ3;
                
        if(GZ < 2) GZ = 2; else GZ = (GZ + 3) >> 2 << 2;//padding to 2x
        if(GZ > conv3D_dW_GemmSK_maxPart) GZ = conv3D_dW_GemmSK_maxPart;
        return GZ;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="GridZ: WinogradV2SHW">
    protected int conv3D_dW_WinogradV2SHW_64x64_GridZ(//cache block size =  64 * 64
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int n, int r)//F(n, r)
    {
	final int GK = N * OH * OW;
	final int SM_count = dev.multiProcessorCount();
	final float intensity = Cuda_conv3D.Im2colWinograd_64x32_intensity(n, r);
        if (intensity == -1) throw new IllegalArgumentException(String.format("F(%d, %d) is not supported", n, r));
                
	//-----[Stage0]---------------------------------------------------------
	final int b0 = Cuda_conv3D.Im2colWinograd_64x64_nblock(OH, OW, N, OC, n);//increase with GK
	final int b1 = Cuda_conv3D.Im2colWinograd_64x64_nblock(IH, IW, N, IC, n);//increase with GK
	final int b2 = Cuda_conv3D.Im2colWinograd_64x64_nblock(FH, FW, IC, OC, n);
	int GZ = (int) ((b0 + b1) / (b2 * 1.45f)); if (GZ <= 2) {
            float expect_block_perSM = 2.0f + conv3D_dW_WinogradSHW_maxBlock_perSM * (GK / (GK + 1024.0f));
            float expect_parallelism = (SM_count << 1) * expect_block_perSM;
            if (b2 >= expect_parallelism) return 1;//enough parallelism
	}

	//-----[Stage2]---------------------------------------------------------
        int GZ2, mblock = conv3D_dW_WinogradSHW_maxPart * b2; float nblock;
	if ((intensity > 14.5f) && (mblock > (nblock = 6.5641f * SM_count))) GZ2 = (int) (nblock / b2);
	else {
            float maxBlock_perSM = conv3D_dW_WinogradSHW_maxBlock_perSM * (b2 <= 32 ? 2 : 1);
            GZ2 = (int) ((SM_count * maxBlock_perSM + b2 - 1) / b2);
	}
        float size = (GK / 512f) * (FH * FW * IC / 512f) * (OC / 1024f);
        float coef = size / SM_count; if (coef < 0.35f) coef = 0.35f; else if (coef > 1f) coef = 1f;
        GZ2 = (int) (GZ2 * coef); if (GZ > GZ2) GZ = GZ2;

	//-----[Stage3]---------------------------------------------------------
	final int GZ3 = GK >> 8; if (GZ > GZ3) GZ = GZ3;//min: GK_slice = 512

	if (GZ < 2) GZ = 2;//padding to 2x
	else if (GZ < 32) GZ = (GZ + 3) >> 2 << 2;//padding to 4x
	else GZ = (GZ + 7) >> 3 << 3;//padding to 8x

	if (GZ > conv3D_dW_WinogradSHW_maxPart) GZ = conv3D_dW_WinogradSHW_maxPart;
	return GZ;
    }
    
    protected int conv3D_dW_WinogradV2SHW_64x32_GridZ(//cache block size = 64 * 32
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int n, int r)//F(n, r)
    {
	final int GK = N * OH * OW;
	final int SM_count = dev.multiProcessorCount();
	final float intensity = Cuda_conv3D.Im2colWinograd_64x32_intensity(n, r);
        if (intensity == -1) throw new IllegalArgumentException(String.format("F(%d, %d) is not supported", n, r));
                
	//-----[Stage0]---------------------------------------------------------
	final int b0 = Cuda_conv3D.Im2colWinograd_64x32_nblock(OH, OW, N, OC, n);//increase with GK
	final int b1 = Cuda_conv3D.Im2colWinograd_64x32_nblock(IH, IW, N, IC, n);//increase with GK
	final int b2 = Cuda_conv3D.Im2colWinograd_64x32_nblock(FH, FW, IC, OC, n);
	int GZ = (int) ((b0 + b1) / (b2 * 1.45f)); 
        if (GZ <= 2) {
            float expect_block_perSM = 2.0f + conv3D_dW_WinogradSHW_minBlock_perSM * (GK / (GK + 1024.0f));
            float expect_parallelism = (SM_count << 1) * expect_block_perSM;
            if (b2 >= expect_parallelism) return 1;//enough parallelism
	}

	//-----[Stage2]---------------------------------------------------------
	int GZ2, mblock = conv3D_dW_WinogradSHW_maxPart * b2; float nblock;
	if      ((intensity > 14.5f) && (mblock > (nblock = 6.5641f * SM_count))) GZ2 = (int) (nblock / b2);
	else if ((intensity > 13.5f) && (mblock > (nblock = 7.3846f * SM_count))) GZ2 = (int) (nblock / b2);
	else if                         (mblock > (nblock = 9.8462f * SM_count))  GZ2 = (int) (nblock / b2);
	else {
            float maxBlock_perSM = conv3D_dW_WinogradSHW_maxBlock_perSM * (b2 <= 32 ? 2 : 1);
            GZ2 = (int) ((SM_count * maxBlock_perSM + b2 - 1) / b2);
	}
        float size = (N * OH * OW / 512f) * (FH * FW * IC / 512f) * (OC / 1024f);
        float coef = size / SM_count; if (coef < 0.35f) coef = 0.35f; else if (coef > 1f) coef = 1f;
        GZ2 = (int) (GZ2 * coef); if (GZ > GZ2) GZ = GZ2;

	//-----[Stage3]---------------------------------------------------------
	final int GZ3 = GK >> 8; if (GZ > GZ3) GZ = GZ3;//min: GK_slice = 512

	if (GZ < 2) GZ = 2; //padding to 2x
	else if (GZ < 32) GZ = (GZ + 3) >> 2 << 2;//padding to 4x
	else GZ = (GZ + 7) >> 3 << 3;//padding to 8x

	if (GZ > conv3D_dW_WinogradSHW_maxPart) GZ = conv3D_dW_WinogradSHW_maxPart;
	return GZ;
    }
        
    protected int conv3D_dW_WinogradV2SHW_32x32_GridZ(//cache block size = 32 * 32
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int n, int r)//F(n, r)
    {
	final int GK = N * OH * OW;
	final int SM_count = dev.multiProcessorCount();
                
	//-----[Stage0]---------------------------------------------------------
	final int b0 = Cuda_conv3D.Im2colWinograd_64x32_nblock(OH, OW, N, OC, n);//increase with GK
	final int b1 = Cuda_conv3D.Im2colWinograd_64x32_nblock(IH, IW, N, IC, n);//increase with GK
	final int b2 = Cuda_conv3D.Im2colWinograd_64x32_nblock(FH, FW, IC, OC, n);
	int GZ = (int) ((b0 + b1) / (b2 * 1.45f)); 
        if (GZ <= 2) {
            float expect_block_perSM = 2.0f + conv3D_dW_WinogradSHW_minBlock_perSM * (GK / (GK + 1024.0f));
            float expect_parallelism = (SM_count << 1) * expect_block_perSM;
            if (b2 >= expect_parallelism) return 1;//enough parallelism
	}

	//-----[Stage2]---------------------------------------------------------
	final float maxBlock_perSM = conv3D_dW_WinogradSHW_maxBlock_perSM * (b2 <= 32 ? 2 : 1);
	int GZ2 = (int) ((SM_count * maxBlock_perSM + b2 - 1) / b2);
        float size = (N * OH * OW / 512f) * (FH * FW * IC / 512f) * (OC / 1024f);
        float coef = size / SM_count; if (coef < 0.35f) coef = 0.35f; else if (coef > 1f) coef = 1f;
        GZ2 = (int) (GZ2 * coef); if (GZ > GZ2) GZ = GZ2;

	//-----[Stage3]---------------------------------------------------------
	final int GZ3 = GK >> 8; if (GZ > GZ3) GZ = GZ3;//min: GK_slice = 512

	if (GZ < 2) GZ = 2; //padding to 2x
	else if (GZ < 32) GZ = (GZ + 3) >> 2 << 2;//padding to 4x
	else GZ = (GZ + 7) >> 3 << 3;//padding to 8x

	if (GZ > conv3D_dW_WinogradSHW_maxPart) GZ = conv3D_dW_WinogradSHW_maxPart;
	return GZ;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="GridZ -> Slices: WinogradV2SHW">
    protected int[] conv3D_dW_WinogradV2SHW_s4_slices(
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int ph, 
        int n, int r) {//F(n, r)
        final int GZ = conv3D_dW_WinogradV2SHW_64x64_GridZ(OH, OW, IH, IW, FH, FW, N, IC, OC, n, r);
        final int[] Slice = conv3D_dW_WinogradV2_SHW_slices(OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r, GZ);
        if (Slice == null) return Slice;
        int OH_slice = Slice[1], HZ = OH / OH_slice;
        int OW_slice = Slice[2], WZ = (OW + OW_slice - 1) / OW_slice;
        Slice[0] = HZ * WZ;//GZ = HZ * WZ
        return Slice;
    }
    
    protected int[] conv3D_dW_WinogradV2SHW_s8_slices(
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int ph,
        int n, int r) {//F(n, r)
        final int GZ = conv3D_dW_WinogradV2SHW_64x32_GridZ(OH, OW, IH, IW, FH, FW, N, IC, OC, n, r);
        final int[] Slice = conv3D_dW_WinogradV2_SHW_slices(OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r, GZ);
        if (Slice == null) return Slice;
        final int OH_slice = Slice[1], HZ = OH / OH_slice;
        final int OW_slice = Slice[2], WZ = (OW + OW_slice - 1) / OW_slice;
        Slice[0] = HZ * WZ;//GZ = HZ * WZ
        return Slice;
    }
    
    protected int[] conv3D_dW_WinogradV2SHW_s16_slices(
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int ph,
	int n, int r) {//F(n, r)
        boolean c64 = ((OC >= 64) && ((OC & 63) == (OC & 31))) || (OC >= 256);
        final int GZ = (c64 ?
                conv3D_dW_WinogradV2SHW_64x32_GridZ(OH, OW, IH, IW, FH, FW, N, IC, OC, n, r):
                conv3D_dW_WinogradV2SHW_32x32_GridZ(OH, OW, IH, IW, FH, FW, N, IC, OC, n, r));
        final int[] Slice = conv3D_dW_WinogradV2_SHW_slices(OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r, GZ);
        if (Slice == null) return Slice;
        final int OH_slice = Slice[1], HZ = OH / OH_slice;
        final int OW_slice = Slice[2], WZ = (OW + OW_slice - 1) / OW_slice;
        Slice[0] = HZ * WZ;//GZ = HZ * WZ
        return new int[] { Slice[0], Slice[1], Slice[2], (c64 ? 1 : 0) };
    }
    
    protected int[] conv3D_dW_WinogradV2_SHW_slices(
	int OH, int OW, int IH, int IW, int FH, int FW,
	int N, int IC, int OC, int ph,
	int n, int r,//F(n, r)
        int GZ) {
	final int HZ_max = OH / (ph + 1);//min(tOH_slice) = OH_slice - ph > 0, so: OH_slice > ph
	final int WZ_max = OW / r;
	final int GZ_max = HZ_max * WZ_max;

	if (GZ > (GZ_max << 1)) return null;//N is large, but OH, OW is small
	if (GZ > GZ_max) GZ = GZ_max;

	//------[Stage1]--------------------------------------------------------
	if (GZ == 1) {
            int OW_slice = (OW / r) * r;
            int OH_slice = OH;//HZ = 1
            return new int[] { 0, OH_slice, OW_slice };
	}

	//------[Stage2: GridZ > 1]---------------------------------------------
        if (((r & 1) == 0) && (OW % r == 0) && (GZ <= HZ_max)) {
            int countR2 = (OW / r) * (OH / r);
            if ((countR2 < 256) && (countR2 > 4)) {//Case1
                int OW_slice = OW;//WZ = 1
                int OH_slice = OH / GZ;
                return new int[] { 0, OH_slice, OW_slice };
            }
        }

	if (GZ >= WZ_max) {
            int OW_slice = r; 
            int WZ = OW / r; float fHZ = 1.0f * GZ / WZ; int HZ = (int) fHZ; 
            if (OW % OW_slice != 0) WZ++;
            if ((HZ < HZ_max) && (fHZ - HZ > 0.55f) && ((HZ + 1) * WZ <= conv3D_dW_WinogradSHW_maxPart)) HZ++;
            int OH_slice = OH / HZ;
            if ((OW % r != 0) && ((OH_slice >> 1) > ph)) {//better slices layout
                int g = OW / OW_slice;//most memory cost: 9/8 - > 5/4
                if ((g > 7) && ((g & 1) == 0)) {//g % 2 == 0
                    OW_slice = OW_slice << 1;
                    OH_slice = OH_slice >> 1;
                }
            }
            return new int[] { 0, OH_slice, OW_slice };
	}

	//------[Stage2: 1 < GridZ < WZ_max]----------------------------------
        //[1] WZ0 = ceil(OW / OW_slice), WZ = floor(OW / OW_slice)
	//[2] HZ = GridZ / WZ
	//[3] GridZ = HZ * WZ0
	//[4] OW_slice % r = 0
	//[5] GridZ < WZ_max
	//[6] GridZ0 = WZ0 * HZ
	//(1) to maximize speed: (OW % OW_slice) should be minimized
	//(2) to minimize memory: 
	//		(1.1) GridZ0 should be minimized, 
	//		(1.2) minimize HZ, maximize WZ, if OH_slice % r != 0
	//both (1) and (2) can be meet:
	//Case1: OW % r == 0; GridZ < HZ_max
	//Case2: OW_slice = (OW / GridZ) / r * r; OW % OW_slice < r
	//Case3: WZ_max % GridZ == 0; maximize WZ = GridZ
	//		=> WZ_max = k * GridZ
	//      => WZ_max * r = k * (GridZ * r), since: WZ_max * r + g = OW
	//      => WZ_max * r + g = GridZ * (k * r) + g = OW, where: g < r
	//      So: OW_slice = k * r = (WZ_max / GridZ) * r
	if ((OW % r == 0) && (GZ <= HZ_max)) {//Case1
            int OW_slice = OW;//WZ = 1
            int OH_slice = OH / GZ;
            return new int[] { 0, OH_slice, OW_slice };
	}

        {int OW_slice = (OW / GZ) / r * r; if (OW % OW_slice < r) {//Case2
            if ((OW % r != 0) && ((OH >> 1) > ph)) {//better slices layout
                int g = OW / OW_slice;//most memory cost: 9/8 - > 5/4
                if ((g > 7) && ((g & 1) == 0)) {//g % 2 == 0
                    OW_slice = OW_slice << 1;
                    int OH_slice = OH >> 1;//HZ = 2
                    return new int[] { 0, OH_slice, OW_slice };
                }
            }
            int OH_slice = OH;//HZ = 1
            return new int[] { 0, OH_slice, OW_slice };
	}}
	
	if (WZ_max % GZ == 0) {//Case3
            int OW_slice = (WZ_max / GZ) * r;//WZ = GridZ
            int OH_slice = OH;//HZ =1
            return new int[] { 0, OH_slice, OW_slice };
	}

	//------[Stage4: min factor]--------------------------------------------
        //WZ_max % GridZ != 0, and WZ is not a prime number
	//(1) E = OW / r * r = WZ_max * r = (k*GridZ + g) * r
	//    E = (GridZ*k) * r + g*r, where: GridZ, k, and g are multually prime
	//(2) OW_slice = x * r
	//(3) WZ = E / OW_slice =  E / (x * r) = WZ_max / x <= GridZ 
	//    So: x >= WZ_max / GridZ
	//(4) HZ = GridZ / WZ = GridZ / (WZ_max / x) = (GridZ * x) / WZ_max <= HZ_max
	//	  So: x <= (HZ_max * WZ_max) / GridZ
	//Steps:
	//  <1> s = WZ_max / GridZ
	//  <2> e = (HZ_max * WZ_max) / GridZ
	//  <3> minimize: x to maximize: WZ, meet: WZ_max % x == 0, s <= x <= e
	//[1] when: GridZ >= sqrtf(WZ_max): (HZ_max * WZ_max) / x > sqrtf(WZ_max)
	//    So: x <= HZ_max * sqrt(WZ_max)
	//[2] when: GridZ <= sqrtf(WZ_max): sqrtf(WZ_max) >= WZ_max / x
	//    So: x >= sqrt(WZ_max)
	if (!Num.is_prime(WZ_max)) {//WZ_max is not a prime number
            int s = WZ_max / GZ; if (s < 1) s = 1;
            int e = (HZ_max * WZ_max) / GZ;
            int x = -1; for (int i = s; i <= e; i++) if (WZ_max % i == 0) { x = i; break; }
            if (x != -1) {//meet the conditions
                int OW_slice = x * r;
                int WZ = OW / OW_slice; float fHZ = 1.0f * GZ / WZ; int HZ = (int) fHZ;
                if (OW % OW_slice != 0) WZ++;
                if ((HZ < HZ_max) && (fHZ - HZ > 0.55f) && ((HZ + 1) * WZ <= conv3D_dW_WinogradSHW_maxPart)) HZ++;
                int OH_slice = OH / HZ;
                return new int[] { 0, OH_slice, OW_slice };
            }
	}

	//------[Stage5: default]---------------------------------------------
	if (GZ < HZ_max) {
            int OW_slice = (OW / r) * r;
            int OH_slice = OH / GZ;
            return new int[] { 0, OH_slice, OW_slice };
	}
	
        //default method: 1 < GridZ < WZ_max
	int OW_slice = (OW / GZ) / r * r;
	int OH_slice = OH;
        return new int[] { 0, OH_slice, OW_slice };
    }
    //</editor-fold>
    
    @Override
    public Syncer conv3D_deltaW(
            long deltaW_address, int FH, int FW,
            long X_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {     
        final int GK = N * OH * OW;
        //<editor-fold defaultstate="collapsed" desc="Common Area: no need to split K">
        if(GK < conv3D_dW_sk_threshold) {
            final int length = Cuda_dconv3D_deltaW.GEMM_nstream(FH, FW, OC, IC);
            long[] streamArray = streamPool.getStreamArray(length);
            
            if((FH == 1) && (FW == 1) && (ph == 0) && (pw == 0) && (sh == 1) && (sw == 1))
                Cuda_dconv3D_deltaW.dconv3D_deltaW_W1(streamArray, length,
                        X_address, IH, IW,
                        deltaY_address,
                        deltaW_address, 
                        N, IC, OC);
            else /*Normal GEMM*/
                Cuda_dconv3D_deltaW.dconv3D_deltaW(streamArray, length,
                        X_address, IH, IW,
                        deltaY_address, OH, OW,
                        deltaW_address, FH, FW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            return new StreamArraySyncer(streamPool, streamArray);        
        } 
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="SplitK Area: improve parallelism">
        else { 
            //------[decide algorithm and GridZ]--------------------------------
            int algo = conv3D_deltaW_decide_algorithm(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            int GZ, Slice[] = null, n = 0, r = 0;
            for(;;) {
                if      ((algo / 10) == DECONV3D_DW_WINOGRAD_V2_SHW_S16) { n = algo - dc3dW_s16_nbase; r = dc3dW_s16_r[n - wgradSHW_s16_FW_min]; if ((Slice = conv3D_dW_WinogradV2SHW_s16_slices(OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r)) == null) algo = -1; else { algo /= 10; GZ = Slice[0]; break; } }
                if (algo == -1) algo = conv3D_deltaW_decide_algorithm_s8(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);//reselect s8 Algorithms
                
                if      ((algo / 10) == DECONV3D_DW_WINOGRAD_V2_SHW_S8)  { n = algo - dc3dW_s8_nbase;  r = dc3dW_s8_r [n - wgradSHW_s8_FW_min];  if ((Slice = conv3D_dW_WinogradV2SHW_s8_slices (OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r)) == null) algo = -1; else { algo /= 10; GZ = Slice[0]; break; } }
                else if ((algo / 10) == DECONV3D_DW_WINOGRAD_V2_SHW_S4)  { n = algo - dc3dW_s4_nbase;  r = dc3dW_s4_r [n - wgradSHW_s4_FW_min];  if ((Slice = conv3D_dW_WinogradV2SHW_s4_slices (OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r)) == null) algo = -1; else { algo /= 10; GZ = Slice[0]; break; } }
              
//                if      ((algo / 10) == DECONV3D_DW_WINOGRAD_V2_SHW_S8)  { n = algo - dc3dW_s8_nbase;  r = dc3dW_s8_r [n - wgradSHW_s8_FW_min];  if ((Slice = conv3D_dW_WinogradV2SHW_s8_slices (OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r)) == null) algo = -1; else { algo /= 10; GZ = Slice[0]; break; } }
//                else if ((algo / 10) == DECONV3D_DW_WINOGRAD_V2_SHW_S16) { n = algo - dc3dW_s16_nbase; r = dc3dW_s16_r[n - wgradSHW_s16_FW_min]; if ((Slice = conv3D_dW_WinogradV2SHW_s16_slices(OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r)) == null) algo = -1; else { algo /= 10; GZ = Slice[0]; break; } }
//                else if ((algo / 10) == DECONV3D_DW_WINOGRAD_V2_SHW_S4)  { n = algo - dc3dW_s4_nbase;  r = dc3dW_s4_r [n - wgradSHW_s4_FW_min];  if ((Slice = conv3D_dW_WinogradV2SHW_s4_slices (OH, OW, IH, IW, FH, FW, N, IC, OC, ph, n, r)) == null) algo = -1; else { algo /= 10; GZ = Slice[0]; break; } }
              
                if (algo == -1) algo = conv3D_deltaW_decide_algorithm_GEMM(OH, OW, IH, IW, FH, FW, IC, OC, sh, sw);//reselect Gemm Algorithms
                if (algo == DECONV3D_DW_GEMMSK_V2) { GZ = conv3D_dW_GemmSKV2_GridZ(OH, OW, IH, IW, FH, FW, N, IC, OC, GK); break; }
                /* (algo == DECONV3D_DW_GEMMSK)*/  { GZ = conv3D_dW_GemmSK_GridZ(OH, OW, IH, IW, FH, FW, N, IC, OC, GK); break; }
            }
            
            final int part = GZ - 1;
            final int sizeW = OC * FH * FW * IC;
            long block[] = null, deltaW_buf_address = 0L;
            if(part > 0) { block = core.malloc(sizeW * part); deltaW_buf_address = block[1]; }
            
            //------[decide stream size]----------------------------------------
            final int length;
            if      (algo == DECONV3D_DW_WINOGRAD_V2_SHW_S8)  length = Cuda_dconv3D_deltaW.Im2colWinogradV2_s8_nstream (OC, IC, OW, r);
            else if (algo == DECONV3D_DW_WINOGRAD_V2_SHW_S16) length = Cuda_dconv3D_deltaW.Im2colWinogradV2_s16_nstream(OC, IC, OW, r, Slice[3]);
            else if (algo == DECONV3D_DW_WINOGRAD_V2_SHW_S4)  length = Cuda_dconv3D_deltaW.Im2colWinogradV2_s4_nstream (OC, IC, OW, r);
            else if (algo == DECONV3D_DW_GEMMSK_V2)                  length = Cuda_dconv3D_deltaW.GEMMV2_nstream(OC, IC);
            else   /*(algo == DECONV3D_DW_GEMMSK)*/                  length = Cuda_dconv3D_deltaW.GEMM_nstream(FH, FW, OC, IC);
            
            //------[Stage1: find gradient of W]--------------------------------
            long[] streamArray = streamPool.getStreamArray(length);
            if (algo == DECONV3D_DW_GEMMSK_W1) 
                Cuda_dconv3D_deltaW.dconv3D_deltaW_GemmSK_W1(streamArray, length, GZ,
                        X_address, IH, IW,
                        deltaY_address,
                        deltaW_address,
                        deltaW_buf_address,
                        N, IC, OC);
            else if (algo == DECONV3D_DW_WINOGRAD_V2_SHW_S8 && 
                Cuda_dconv3D_deltaW.Im2col_WinogradV2_SHW_s8_texture(streamArray, length, GZ, Slice[1], Slice[2],
                        X_address, IH, IW,
                        deltaY_address, OH, OW,
                        deltaW_address,
                        deltaW_buf_address, FH, FW,
                        N, IC, OC,
                        ph, pw, n)); 
            else if (algo == DECONV3D_DW_WINOGRAD_V2_SHW_S16 &&
                Cuda_dconv3D_deltaW.Im2col_WinogradV2_SHW_s16_texture(streamArray, length, GZ, Slice[1], Slice[2], Slice[3], 
                        X_address, IH, IW,
                        deltaY_address, OH, OW, 
                        deltaW_address, 
                        deltaW_buf_address, FH, FW,
                        N, IC, OC, 
                        ph, pw, n));
            else if (algo == DECONV3D_DW_WINOGRAD_V2_SHW_S4 && 
                Cuda_dconv3D_deltaW.Im2col_WinogradV2_SHW_s4_texture(streamArray, length, GZ, Slice[1], Slice[2],
                        X_address, IH, IW,
                        deltaY_address, OH, OW,
                        deltaW_address,
                        deltaW_buf_address, FH, FW,
                        N, IC, OC,
                        ph, pw, n)); 
            else if (algo == DECONV3D_DW_GEMMSK_V2) 
                Cuda_dconv3D_deltaW.dconv3D_deltaW_GemmV2SK(streamArray, length, GZ,
                        X_address, IH, IW, 
                        deltaY_address, OH, OW,
                        deltaW_address, 
                        deltaW_buf_address, FH, FW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            else /*algo == DECONV3D_DW_GEMMSK*/
                Cuda_dconv3D_deltaW.dconv3D_deltaW_GemmSK(streamArray, length, GZ,
                        X_address, IH, IW,
                        deltaY_address, OH, OW,
                        deltaW_address,
                        deltaW_buf_address, FH, FW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            
            //------[Stage2: sum up deltaW from each part]----------------------
            if(part > 0) {
                long event = Cuda.newEvent_DisableTiming();
                Cuda.eventRecord(event, streamArray, streamArray.length);
             
                long stream = streamArray[0];
                Cuda.streamWaitEvent_default(stream, event);
                Cuda_dconv3D_deltaW.buf_summary(stream, 
                        deltaW_buf_address, deltaW_address, 
                        part, sizeW);
            
                Cuda.deleteEvent(event);
                return new SplitKSyncer(core, block, streamPool, streamArray);
            }
            
            return new StreamArraySyncer(streamPool, streamArray);        
        }
        //</editor-fold>
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: conv3D_deltaX">
    //<editor-fold defaultstate="collapsed" desc="Configurations">
    protected boolean conv3D_dX_s1_useTexture = true;
    public boolean conv3D_deltaX_s1_useTexture() { return conv3D_dX_s1_useTexture; }
    public CudaFloat32EngineBase conv3D_deltaX_s1_useTexture(boolean flag) { conv3D_dX_s1_useTexture = flag; return this; }
    
    protected boolean conv3D_dX_ks_useTexture = false;
    public boolean conv3D_deltaX_KernelSplit_useTexture() { return conv3D_dX_ks_useTexture; }
    public CudaFloat32EngineBase conv3D_deltaX_KernelSplit_useTexture(boolean flag) { conv3D_dX_ks_useTexture = flag; return this; } 
    
    protected float conv3D_dX_ZeroPaddingS1V2_Q  = 1.03923f;//sqrt(1.08f)
    protected float conv3D_dX_s1_V2_Q2 = 1.08f;
    public float conv3D_deltaX_ZeroPaddingS1V2_Q() { return conv3D_dX_ZeroPaddingS1V2_Q; }
    public CudaFloat32EngineBase conv3D_deltaX_ZeroPaddingS1V2_Q(float Q) {
        if(Q < 1.03f) throw new IllegalArgumentException(String.format("Q { got %f } must >= 1.03", Q));
        conv3D_dX_ZeroPaddingS1V2_Q  = Q;
        conv3D_dX_s1_V2_Q2 = Q * Q;
        return this;
    }
    
    protected float conv3D_dX_ks_V2_Q = 1.08f;
    public float conv3D_deltaX_KernelSplitV2_Q() { return conv3D_dX_ks_V2_Q; }
    public CudaFloat32EngineBase conv3D_deltaX_KernelSplitV2_Q(float Q) {
        if(Q < 1.05f) throw new IllegalArgumentException(String.format("Q { got %f }must >= 1.05", Q));
        conv3D_dX_ks_V2_Q = Q;
        return this;
    }
    
    protected boolean conv3D_dX_Im2colWinograd_s8 = true;
    public boolean conv3D_deltaX_Im2colWinograd_s8() { return conv3D_dX_Im2colWinograd_s8; }
    public CudaFloat32EngineBase conv3D_deltaX_Im2colWinograd_s8(boolean flag) { conv3D_dX_Im2colWinograd_s8 = flag; return this; }
    
    protected boolean conv3D_dX_Im2colWinograd_s16 = true;//0xG -> 16
    public boolean conv3D_deltaX_Im2colWinograd_s16() { return conv3D_dX_Im2colWinograd_s16; }
    public CudaFloat32EngineBase conv3D_deltaX_Im2colWinograd_s16(boolean flag) { conv3D_dX_Im2colWinograd_s16 = flag; return this; }
    //</editor-fold>
    
    public static final int DECONV3D_DX_ZERO_PADDING_S1       = 0;
    public static final int DECONV3D_DX_ZERO_PADDING_S1_V2    = 1;
    public static final int DECONV3D_DX_ZERO_PADDING_W1       = 2;
    public static final int DECONV3D_DX_IM2COL_WINOGRAD_S8    = 3;
    public static final int DECONV3D_DX_IM2COL_WINOGRAD_S16   = 4;
    public static final int DECONV3D_DX_CROSS_ADD             = 5; 
    public static final int DECONV3D_DX_KERNEL_SPLIT          = 6;
    public static final int DECONV3D_DX_KERNEL_SPLIT_IMSR     = 7;
    public static final int DECONV3D_DX_KERNEL_SPLIT_IMS2R    = 8;
    public static final int DECONV3D_DX_KERNEL_SPLIT_IMS2R_V2 = 9;
    //<editor-fold defaultstate="collapsed" desc="Decide Algorithm">
    public int conv3D_deltaX_decide_algorithm(
            int OH, int OW, int IH, int IW, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if((IC <= 8) || ((IC <= 16) && ((sh*sw >= 4) || (OC >= 64)))) return DECONV3D_DX_CROSS_ADD;
        else if((sh == 1) && (sw == 1)) {
            if((FH == 1) && (FW == 1) && (ph == 0) && (pw == 0)) return DECONV3D_DX_ZERO_PADDING_W1;
            
            //[speed of ZeroPaddingV2]------------------------------------------
            float ZeroPaddingV2 = 0;
            if((N > 63) && (IC > 63) && (Cuda_dconv3D_deltaX.GEMM_GM_slice(IC, N*IH*IW, FH*FW*OC) <= 0)) {
                float psu = Cuda_dconv3D_deltaX.psu_s1(IH, IW, FH, FW, OH, OW);
                ZeroPaddingV2 = (float) Math.sqrt(psu) / conv3D_dX_ZeroPaddingS1V2_Q;
            }
            
            //[speed of Im2colWinograd]-----------------------------------------
            boolean Im2col_Winograd_C = ((OC & 7) == 0) && (N * IH >= 32);
            
            float Im2col_Winograd_s8 = 0;
            if (conv3D_dX_Im2colWinograd_s8 && Im2col_Winograd_C &&
                    (FW + IW > 8) && (IC >= 64) &&
                    (FW >= wgrad_s8_FW_min && FW <= wgrad_s8_FW_max) && (FH >= 1 && FH <= 9))
                Im2col_Winograd_s8 = WGrad_s8_Derailleur[FW - wgrad_s8_FW_min].apply(IW);
            
            float Im2col_Winograd_s16 = 0;
            if (conv3D_dX_Im2colWinograd_s16 && Im2col_Winograd_C &&
                    (FW + IW > 16) && (IC >= 32) && 
                    (FW >= wgrad_s16_FW_min && FW <= wgrad_s16_FW_max) && (FH >= 1 && FH <= 9) && 
                    (pw >= ((FW - 1) >> 1)) && (OW - IW - pw + (FW >> 1 << 1)- ((FW - 1) >> 1) >= 0))
                Im2col_Winograd_s16 = Winograd_s16_Derailleur[FW - wgrad_s16_FW_min].apply(IW);
            
            //[decide algorithm]------------------------------------------------
            int Algo = DECONV3D_DX_ZERO_PADDING_S1; float speed = 1.0f;//default algorithm
            if (speed < Im2col_Winograd_s16) { Algo = DECONV3D_DX_IM2COL_WINOGRAD_S16; speed = Im2col_Winograd_s16; }
            if (speed < Im2col_Winograd_s8) { Algo = DECONV3D_DX_IM2COL_WINOGRAD_S8; speed = Im2col_Winograd_s8; }
            if (speed < ZeroPaddingV2) Algo = DECONV3D_DX_ZERO_PADDING_S1_V2;
            return Algo;
        }
        else {
            if((sh == 2) && (sw == 2) && (IH & 1) == 0 && (IW & 1) == 0) {
                boolean V2 = (N > 63) && (IC > 63) && //[FH = FW = 1], V2 = false
                        (Cuda_dconv3D_deltaX.psu_Ims2(IH, IW, FH, FW, OH, OW) > conv3D_dX_ks_V2_Q) && 
                        (Cuda_dconv3D_deltaX.GEMM_GM_slice(IC, N*IH*IW, FH*FW*OC) <= 0);
                return (V2 ? 
                        DECONV3D_DX_KERNEL_SPLIT_IMS2R_V2 : 
                        DECONV3D_DX_KERNEL_SPLIT_IMS2R);
            }
            return ((IH % sh == 0) && (IW % sw == 0) ? 
                    DECONV3D_DX_KERNEL_SPLIT_IMSR :
                    DECONV3D_DX_KERNEL_SPLIT);
        }
    }
    //</editor-fold>
    
    @Override
    public Syncer conv3D_deltaX(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw) 
    {   
        final int algo = conv3D_deltaX_decide_algorithm(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        //<editor-fold defaultstate="collapsed" desc="crossAdd kernel: when IC is small">
        if(algo == DECONV3D_DX_CROSS_ADD) {
            final int length = Cuda_dconv3D_deltaX.CrossAdd_nstream(OH, OW, N, OC);
            long[] streamArray = streamPool.getStreamArray(length);
            Cuda_dconv3D_deltaX.DCONV3D_deltaX_crossAdd(streamArray, length, 
                    deltaY_address, OH, OW,
                    W_address, FH, FW,
                    deltaX_address, IH, IW,
                    N, IC, OC,
                    sh, sw, ph, pw);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="dense kernel: sh = sw = 1">
        else if ((sh == 1) && (sw == 1)) {
            final int length;
            if      (algo == DECONV3D_DX_IM2COL_WINOGRAD_S8)  length = Cuda_dconv3D_deltaX.Im2colWinograd_s8_nstream(FW, IH, IW, N, IC);
            else if (algo == DECONV3D_DX_IM2COL_WINOGRAD_S16) length = Cuda_dconv3D_deltaX.Im2colWinograd_s16_nstream(FW, IH, IW, N, IC);
            else if (algo == DECONV3D_DX_ZERO_PADDING_S1_V2)  length = Cuda_dconv3D_deltaX.GEMMV2_nstream_s1(N, IC);
            else  /*(algo == DECONV3D_DX_ZERO_PADDING_S1)*/   length = Cuda_dconv3D_deltaX.GEMM_nstream_s1(IH, IW, N, IC);
            
            long[] streamArray = streamPool.getStreamArray(length);
            boolean useTexture = conv3D_dX_s1_useTexture && (IC > 15);
            
            if (algo == DECONV3D_DX_ZERO_PADDING_W1) 
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_W1(streamArray, length, 
                        deltaY_address, 
                        W_address, 
                        deltaX_address, IH, IW,
                        N, IC, OC);
            else if (algo == DECONV3D_DX_IM2COL_WINOGRAD_S8 && 
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_Im2col_Winograd_s8_texture(useTexture, streamArray, length,
                        deltaY_address, OH, OW, 
                        W_address, FH, FW,
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        ph, pw));
            else if (algo == DECONV3D_DX_IM2COL_WINOGRAD_S16 &&
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_Im2col_Winograd_s16_texture(useTexture, streamArray, length, 
                        deltaY_address, OH, OW, 
                        W_address, FH, FW, 
                        deltaX_address, IH, IW,
                        N, IC, OC, 
                        ph, pw));
            else if (algo == DECONV3D_DX_ZERO_PADDING_S1_V2) 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_V2_s1(useTexture, streamArray, length,
                        deltaY_address, OH, OW, 
                        W_address, FH, FW, 
                        deltaX_address, IH, IW, 
                        N, IC, OC, 
                        ph, pw);
            else if (useTexture) /*(algo == DECONV3D_DX_ZERO_PADDING_S1)*/
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_s1_texture(streamArray, length, 
                        deltaY_address, OH, OW, 
                        W_address, FH, FW, 
                        deltaX_address, IH, IW, 
                        N, IC, OC,
                        ph, pw);
            else /*(algo == DECONV3D_DX_ZERO_PADDING_S1)*/
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_s1(streamArray, length, 
                        deltaY_address, OH, OW, 
                        W_address, FH, FW, 
                        deltaX_address, IH, IW, 
                        N, IC, OC,
                        ph, pw);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="sparse kernel: sh * sw >= 2">
        else {
            //------[Stage1: remode W -> CW]-------------------------------------
            final int CFH = (FH + sh - 1) / sh;
            final int CFW = (FW + sw - 1) / sw;
            final int CW_size = (OC * CFH * CFW * IC * sh * sw);//CW[sh, sw, CFH, CFW, OC, IC]
            final long block[] = core.malloc(CW_size);
            final long CW_address = block[1];
            
            long stream = streamPool.getStream();
            Cuda_dconv3D_deltaX.ks_remodev2(stream, 
                    W_address, FH, FW,
                    CW_address, CFH, CFW,
                    OC, IC, sh, sw);
            
            long event1 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event1, stream);//streamArray wait stream
            
            //------[Stage2: deconv3D]------------------------------------------
            final int length = (algo == DECONV3D_DX_KERNEL_SPLIT_IMS2R_V2 ? 
                    Cuda_dconv3D_deltaX.GEMMV2_nstream_s1(N, IC): 
                    Cuda_dconv3D_deltaX.KernelSplit_nstream(IH, IW, N, IC, sh, sw));
            long[] streamArray = streamPool.getStreamArray(length);
            Cuda.streamsWaitEvent_default(streamArray, length, event1);
            
            boolean useTexture = conv3D_dX_ks_useTexture && (IC > 15);
            if (algo == DECONV3D_DX_KERNEL_SPLIT_IMS2R_V2) 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksV2_Ims2R(useTexture, streamArray, length,
                        deltaY_address, OH, OW,
                        CW_address, FH, FW,
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        ph, pw);
            else if (algo == DECONV3D_DX_KERNEL_SPLIT_IMS2R) {
                if(useTexture) 
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksIms2R_texture(streamArray, length,
                            deltaY_address, OH, OW,
                            CW_address, FH, FW,
                            deltaX_address, IH, IW,
                            N, IC, OC,
                            ph, pw);
                else 
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksIms2R(streamArray, length,
                            deltaY_address, OH, OW,
                            CW_address, FH, FW,
                            deltaX_address, IH, IW,
                            N, IC, OC,
                            ph, pw);
            }
            else if (algo == DECONV3D_DX_KERNEL_SPLIT_IMSR) {
                if (useTexture)
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksImsR_texture(streamArray, length,
                            deltaY_address, OH, OW,
                            CW_address, FH, FW,
                            deltaX_address, IH, IW,
                            N, IC, OC,
                            sh, sw, ph, pw);
                else 
                    Cuda_dconv3D_deltaX.DCONV3D_deltaX_ksImsR(streamArray, length,
                            deltaY_address, OH, OW,
                            CW_address, FH, FW,
                            deltaX_address, IH, IW,
                            N, IC, OC,
                            sh, sw, ph, pw);
            }
            else /*algo == DECONV3D_DX_KERNEL_SPLIT*/
                Cuda_dconv3D_deltaX.DCONV3D_deltaX_kernelSplit(streamArray, length,
                        deltaY_address, OH, OW,
                        CW_address, FH, FW,
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            
            Cuda.deleteEvent(event1);
            streamArray = Vector.append(stream, streamArray);
            return new StreamArrayBlockSyncer(streamPool, streamArray, core, block);     
        }
        //</editor-fold>
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="DepthWise Convolution3D">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer depthwise_conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {
        long stream = streamPool.getStream();
        Cuda_depthwise_conv3D.depthwise_conv3D(stream, 
                X_address, IH, IW, 
                W_address, FH, FW, 
                Y_address, OH, OW, 
                N, IC, OC, 
                sh, sw, ph, pw);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer depthwise_conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw, 
            long Bias_address, //stride = OC, lengthv = N*OH*OW*OC
            int lengthv, int width) 
    {
        long stream = streamPool.getStream();
        Cuda_depthwise_conv3D.depthwise_conv3D(stream, 
                X_address, IH, IW, 
                W_address, FH, FW, 
                Y_address, OH, OW, 
                N, IC, OC, 
                sh, sw, ph, pw);
        Cuda_function.linear_dual2D_row(stream, Y_address,
                Bias_address, OC,
                1.0f, 1.0f, 0.0f,
                Y_address,
                lengthv, width, OC);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaW">
    //<editor-fold defaultstate="collapsed" desc="Configurations: GemmSK"> 
    protected int dwconv3D_dW_GemmSK_maxPart = 32;
    public int depthwise_conv3D_deltaW_Gemm_SK_maxpart() { return dwconv3D_dW_GemmSK_maxPart; }
    public CudaFloat32EngineBase depthwise_conv3D_deltaW_Gemm_SK_maxpart(int maxPart) {
        if(maxPart < 8) throw new IllegalArgumentException(String.format(
                "maxPart { got %d } must >= 8", maxPart)); 
        dwconv3D_dW_GemmSK_maxPart = maxPart;
        return this;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="GridZ: GemmSK">
    protected final DWConvDW_GZ_Decider dwdconvdwGemmskGz = (LBX, LBK, FD, IH, IW, FH, FW, OH, OW, N, IC, OC) -> 
            depthwise_conv3D_dW_decide_GridZ(LBX, LBK, FD, IH, IW, FH, FW, OH, OW, N, IC, OC);
    
    protected final float[] dw_conv3D_expect_blockNum_perSM = {
         79.0f, 79.0f, 79.0f,//FD == 1, 2, 3
        105.1f,//FD ==  4
         65.6f,//FD ==  5
         78.8f,//FD ==  6
         91.9f,//FD ==  7
        105.1f,//FD ==  8
        118.2f,//FD ==  9
        131.3f,//FD == 10
        144.4f,//FD == 11
         78.8f,//FD == 12
         85.4f //FD == 13
    };
   
    protected int depthwise_conv3D_dW_minimize_RK(int LBK, int N_OH, int OW, int GZ) {
        final int GK = N_OH * OW;
        int NH_slice = ((N_OH / GZ) >> LBK << LBK);
        int GK_slice = NH_slice * OW;
        int threshold = (int) (GK_slice * 0.25f);
        int RK = GK - GZ * GK_slice; if (RK < threshold) return GZ;
        
        int start = 2, end = GZ;
        if      (GZ > 32) start = (int) (GZ * 0.84f);//5
        else if (GZ > 16) start = (int) (GZ * 0.75f);//4
        else if (GZ >  8) start = (int) (GZ * 0.65f);//3
        else if (GZ >  4) start = GZ - 2;
        
	for (int gz = start; gz < end; gz++) {
            int nh_slice = ((N_OH / gz) >> LBK << LBK);
            int rk = GK - gz * nh_slice * OW;
            if (rk < RK) { RK = rk; GZ = gz; if (RK < threshold) return GZ; }
	}
	return GZ;
    }
    
    protected int depthwise_conv3D_dW_decide_GridZ(int LBX, int LBK, int FD,
            int IH, int IW, int FH, int FW, int OH, int OW,//call back function
            int N, int IC, int OC)
    {
        final int N_OH = N * OH, BK = (1 << LBK);
        if (N_OH < BK) throw new IllegalArgumentException(String.format(
                "N { got %d } * OH { got %d } < BK { got %d  }", N, OH, BK));
        if ((FD > 13) || (FD <= 0)) throw new IllegalArgumentException(String.format(
                "FD { got %d } should in range of [1, 13]", FD));
        
        //-----[Stage0]---------------------------------------------------------
        final int b0 = Cuda_depthwise_conv3D.GEMM_nblock(OH, OW, N, OC);
        final int b1 = Cuda_depthwise_dconv3D_deltaX.GEMM_nblock(IH, IW, N, IC);
        final int b2 = Cuda_depthwise_dconv3D_deltaW.GEMM_nblock(FH, FW, OC, LBX, FD);
        int GZ = (int) ((b0 + b1) / (b2 * 1.45f));
        
        //-----[Stage1]---------------------------------------------------------
        final int GK = N * OH * OW;
	final int GZ1 = GK >> 7;//block.GK >= 128
        if (GZ > GZ1) GZ = GZ1;
        
        //-----[Stage2]---------------------------------------------------------
        final int SM_count = dev.multiProcessorCount();
        float expect_blockNum = SM_count * dw_conv3D_expect_blockNum_perSM[FD - 1];
        float coef =  GK / (128 * 1024.0f);//128 * 32 * 32 -> 1.0f
        if (coef < 0.35f) coef = 0.35f; else if (coef > 1.0f) coef = 1.0f;
        final int GZ2 = (int) (coef * expect_blockNum / b2); 
        if (GZ > GZ2) GZ = GZ2;
        
        //-----[Stage3]---------------------------------------------------------
	if (GZ < 2) GZ = 2; else GZ = (GZ + 3) >> 2 << 2;//padding GZ
        if (GZ > dwconv3D_dW_GemmSK_maxPart) GZ = dwconv3D_dW_GemmSK_maxPart;
        final int GZ3 = N_OH >> LBK; if (GZ > GZ3) GZ = GZ3;//(N * OH) / (GZ * BK) >= 1
        
        return depthwise_conv3D_dW_minimize_RK(LBK, N_OH, OW, GZ);
    }
    //</editor-fold>
    
    @Override
    public Syncer depthwise_conv3D_deltaW(
            long deltaW_address, int FH, int FW, 
            long      X_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {
        int GOC = OC / IC; if ((GOC != 1) && (GOC % 2 != 0)) throw new RuntimeException();
        long stream = streamPool.getStream();
        Cuda.memsetAsync(stream, deltaW_address, 0, (FH * FW * OC) << 2L);
        Cuda_depthwise_dconv3D_deltaW.depthwise_dconv3D_deltaW_GemmSK(stream, dwdconvdwGemmskGz,
                X_address, IH, IW, 
                deltaY_address, OH, OW,
                deltaW_address, FH, FW, 
                N, IC, OC,
                sh, sw, ph, pw);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-porpagation: deltaX">
    @Override
    public Syncer depthwise_conv3D_deltaX(
            long deltaX_address, int IH, int IW, 
            long deltaY_address, int OH, int OW, 
            long      W_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {
        if (sh == 1 && sw == 1) {
            long stream = streamPool.getStream();
            Cuda_depthwise_dconv3D_deltaX.depthwise_dconv3D_deltaX_s1(stream, 
                    deltaY_address, OH, OW,
                    W_address, FH, FW,
                    deltaX_address, IH, IW,
                    N, IC, OC, 
                    ph, pw);
            return new StreamSyncer(streamPool, stream);
        }
        else throw new RuntimeException();
    }
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Pooling 2D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    @Override
    public Syncer pool2D_max(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW, 
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.nstream(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_pool2D.pool2D_max(streamArray, length, 
                X_address, IH, IW, 
                FH, FW,
                Y_address, OH, OW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }

    @Override
    public Syncer pool2D_max_indexed(
            long Y_address, long Index_address, int OH, int OW,
            long X_address, int IH, int IW, 
            int FH, int FW, 
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.nstream(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_pool2D.pool2D_max_indexed(streamArray, length, 
                X_address, IH, IW,
                FH, FW, 
                Y_address, Index_address, OH, OW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer pool2D_avg(boolean ignore_padding,
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.nstream(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        if (ignore_padding)  
            Cuda_pool2D.pool2D_avg_ignore_padding(streamArray, length, 
                    X_address, IH, IW,
                    FH, FW, 
                    Y_address, OH, OW,
                    N, IC,
                    sh, sw, ph, pw);
        else /*ignore_padding = false*/
            Cuda_pool2D.pool2D_avg(streamArray, length,
                    X_address, IH, IW, 
                    FH, FW,
                    Y_address, OH, OW,
                    N, IC, 
                    sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    @Override
    public Syncer unpool2D_max(
            long deltaX_address, long X_address, int IH, int IW,
            long deltaY_address, long Y_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        int length = Cuda_upool2D.nstream(IH, IW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_upool2D.upool2D_max(streamArray, length,
                deltaY_address, Y_address, OH, OW, 
                FH, FW, 
                deltaX_address, X_address, IH, IW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer unpool2D_max_Indexed(
            long deltaX_address, int IH, int IW, 
            long deltaY_address, long Index_address, int OH, int OW, 
            int FH, int FW, 
            int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        //<editor-fold defaultstate="collapsed" desc="Xeno">
//        if(FH == sh && FW == sw && ph == 0 && pw == 0) {//dilated pool2D
//            long stream = streamPool.getStream();
//            int Xsize = N * IH * IW * IC;//X[N, IH, IW, IC]
//            
//            //stage1: deltaX = 0------------------------------------------------
//            long event = Cuda.newEvent_DisableTiming();
//            Cuda.memsetAsync(stream, deltaX_address, 0, Xsize << 2L);//sizeof(float) = 4
//           
//            //stage2: deltaX[index] = deltaY------------------------------------
//            int Ysize = N * OH * OW * IC;
//            Cuda.streamWaitEvent_default(stream, event);
//            Cuda_expk2.dstIndexedMemcpy(stream, 
//                    deltaY_address, Index_address, 
//                    deltaX_address, 
//                    Ysize, IC, IC);
//            
//            Cuda.deleteEvent(event);
//            return new StreamSyncer(streamPool, stream);
//        }
        //</editor-fold>
        int length = Cuda_upool2D.nstream(IH, IW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_upool2D.upool2D_max_Indexed(streamArray, length,
                deltaY_address, Index_address, OH, OW,
                FH, FW,
                deltaX_address, IH, IW,
                N, IC,
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer unpool2D_avg(boolean ignore_padding,
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        long[] streamArray;
        if(FH == sh && FW == sw) {//tiled average pool2D
            int length = Cuda_pool2D.nstream(OH, OW, N, IC);
            streamArray = streamPool.getStreamArray(length);
            if (ignore_padding)
                Cuda_upool2D.upool2D_avg_ignore_padding_tiled(streamArray, length, 
                        deltaY_address, OH, OW,
                        FH, FW,
                        deltaX_address, IH, IW,
                        N, IC,
                        sh, sw, ph, pw);
            else /*ignore_padding = false*/
                Cuda_upool2D.upool2D_avg_tiled(streamArray, length,
                        deltaY_address, OH, OW,
                        FH, FW,
                        deltaX_address, IH, IW,
                        N, IC,
                        sh, sw, ph, pw);
        }
        else {
            int length = Cuda_upool2D.nstream(IH, IW, N, IC);
            streamArray = streamPool.getStreamArray(length);          
            if (ignore_padding)
                Cuda_upool2D.upool2D_avg_ignore_padding(streamArray, length,
                        deltaY_address, OH, OW,
                        FH, FW,
                        deltaX_address, IH, IW,
                        N, IC,
                        sh, sw, ph, pw);
            else /*ignore_padding = false*/
                Cuda_upool2D.upool2D_avg(streamArray, length,
                        deltaY_address, OH, OW,
                        FH, FW,
                        deltaX_address, IH, IW,
                        N, IC,
                        sh, sw, ph, pw);
        }
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="equal, linear, quadratic, rpl, div, add_div"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs">
    @Override
    public Syncer equal_abs2D(long Y_address, 
            long X1_address, long X2_address,
            float min, float max, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.equal_abs2D(stream,
                X1_address, X2_address,
                min, max, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer equal_abs2D_int8(long Y_address, 
            long X1_address, long X2_address, 
            byte min, byte max,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.equal_abs2D_char(stream, 
                X1_address, X2_address, 
                min, max,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer equal_abs2D_int32(long Y_address, 
            long X1_address, long X2_address,
            int min, int max, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.equal_abs2D_int(stream, 
                X1_address, X2_address,
                min, max, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater">   
    @Override
    public Syncer linear_greater2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_greater2D(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear_greater2_2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_greater_dual2D(stream, 
                X1_address, X2_address,
                alpha, beta, gamma,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear_greater_switch2D(long Y_address,
            float alpha, long X_address, float beta, 
            float v1, float v2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_greater_switch2D(stream, 
                alpha, X_address, beta, 
                v1, v2, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear_bound_switch_mul2D(long Y_address,
            float alpha, long X1_address, float vmin, float vmax, 
            long X2_address, float v1, float v2, float v3,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_bound_switch_mul2D(stream, 
                alpha, X1_address, vmin, vmax, 
                X2_address, v1, v2, v3, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear">
    @Override
    public Syncer linear2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D(stream, 
                alpha, X_address, beta, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
   
    @Override
    public Syncer linear_2out2D(long Y1_address, long Y2_address, 
            long X_address,
            float alpha1, float beta1, 
            float alpha2, float beta2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual_out2D(stream, 
                X_address,
                alpha1, beta1, 
                alpha2, beta2,
                Y1_address, Y2_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int8 to float">
    @Override
    public Syncer linear2D_int8_to_dtype(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_char2float(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2D_dtype_to_int8(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_float2char(stream,
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int32 to float">
    @Override
    public Syncer linear2D_int32_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_int2float(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2D_dtype_to_int32(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_float2int(stream,
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear2">
    @Override
    public Syncer linear2_2D(long Y_address, 
            long X1_address, 
            long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override //Y = sum(alpha*Xs[i] + beta)
    public Syncer linear_summary2D(long Y_address,
            long[] Xs, float alpha, float beta, //Xs.length >= 2
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.linear_dual2D(stream,
                Xs[0], Xs[1], 
                alpha, alpha, 2*beta,
                Y_address,//Y = X0*alpha + X1*alpha + 2*beta
                lengthv, width, stride);
        
        for(int i=2; i<Xs.length; i++) 
            Cuda_function.linear_dual2D(stream, 
                    Y_address, Xs[i], 
                    1.0f, alpha, beta, 
                    Y_address, //Y = Y + Xs[i]*alpha + beta
                    lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_iteration2D(long Y_address, 
            long X1_address, long[] X2,
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.linear_dual2D(stream,
                X1_address, X2[0], 
                alpha, beta, gamma,
                Y_address,//Y = alpha*X1 + beta*X2 + gamma
                lengthv, width, stride);
        
        for(int i=1; i<X2.length; i++) 
            Cuda_function.linear_dual2D(stream, 
                    Y_address, X2[i], 
                    alpha, beta, gamma,
                    Y_address,//Y = alpha*Y1 + beta*X2 + gamma
                    lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_row">
    @Override
    public Syncer linear2_2D_row(long Y_address, 
            long X1_address,
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D_row(stream, 
                X1_address, X2_address, row_lengthv,
                alpha, beta, gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_center">
    @Override
    public Syncer linear2_2D_center(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int dim0, int dim1, int dim2,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D_center(stream, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                Y_address,
                dim0, dim1, dim2,
                width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_field">
    @Override
    public Syncer linear2_2D_field(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D_field(stream, 
                X1_address, X2_address, row_lengthv,
                alpha, beta, gamma, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);   
    }
    
    @Override
    public Syncer linear_greater_switch_mul2D(long Y_address, 
            float alpha, long X1_address, float beta, 
            long X2_address, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_greater_switch_mul2D(stream,
                alpha, X1_address, beta,
                X2_address, v1, v2, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);   
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="mul_linear2_2D">
    @Override
    public Syncer mul_linear2_2D(long Y_address, 
            long X_address, long X1_address, long X2_address, 
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.mul_linear_dual2D(stream, 
                X_address, X1_address, X2_address, 
                alpha, beta, gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic">
    @Override
    public Syncer quadratic2D( long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic2D(stream, 
                X_address, alpha, beta, gamma,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer quadratic2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic2D_deltaX(stream, 
                deltaX_address,
                deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2">
    @Override
    public Syncer quadratic2_2D(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D(stream, 
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer quadratic2_2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D_deltaX(stream, 
                deltaX1_address, deltaX2_address,
                deltaY_address, 
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override //Y = sum(alpha*Xs[i]^2 + beta*Xs[i] + gamma)
    public Syncer quadratic_summary2D(long Y_address,
            long[] Xs, float alpha, float beta, float gamma, //Xs.length >= 2
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D(stream,
                Xs[0], Xs[1], 
                alpha, 0, alpha, 
                beta, beta, 2*gamma,
                Y_address, //Y = alpha*X0^2 + alpha*X1^2 + beta*X1 + beta*X2 +2*gamma
                lengthv, width, stride);
        
        for(int i=2; i<Xs.length; i++) 
            Cuda_function.quadratic_dual2D(stream, 
                    Y_address, Xs[i],
                    0, 0, alpha, 
                    1.0f, beta, gamma,
                    Y_address, //Y = Y + alpha*Xs[i]^2 + beta*Xs[i] +gamma
                    lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer quadratic2_iteration2D(long Y_address, 
            long X1_address, long[] X2, 
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D(stream, 
                X1_address, X2[0],
                k11, k12, k22, 
                k1, k2, C, 
                Y_address,//Y = (X1, X2[0])
                lengthv, width, stride);
        
        for(int i=1; i<X2.length; i++) 
            Cuda_function.quadratic_dual2D(stream, 
                    Y_address, X2[i], 
                    k11, k12, k22, 
                    k1, k2, C, 
                    Y_address,//Y = (Y, X2[i])
                    lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_row">
    @Override
    public Syncer quadratic2_2D_row(long Y_address, 
            long X1_address, 
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D_row(stream, 
                X1_address, 
                X2_address, row_lengthv, 
                k11, k12, k22,
                k1, k2, C,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2_center">
    @Override
    public Syncer quadratic2_2D_center(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22, 
            float k1, float k2, 
            float C,
            int dim0, int dim1, int dim2, 
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic_dual2D_center(stream, 
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C, 
                Y_address, 
                dim0, dim1, dim2, 
                width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer quadratic2_2D_center_deltaX(
            long deltaX1_address,//result0 
            long deltaX2_address,//result1
            long deltaY_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2, 
            float C,
            int dim0, int dim1, int dim2,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;

        int N = dim1, M = dim0 * dim2;
        int nextN = Cuda_reduce.center_nextN(N, M);
        if(nextN != 1) {//V[HV: nextN, M: row_lengthv], [HV, dim0*dim2]
            block = core.malloc(nextN * M);
            V_address = block[1];
        }
        
        Cuda_reduce.center_quadratic_dual_deltaX(stream, 
                deltaY_address,
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C, 
                dim0, dim1, dim2, 
                deltaX1_address, 
                V_address, deltaX2_address, 
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_field">
    @Override
    public Syncer quadratic2_2D_field(long Y_address, 
            long X1_address, 
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic_dual2D_field(stream, 
                X1_address, 
                X2_address, row_lengthv,
                k11, k12, k22, 
                k1, k2, C,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: rpl">
    @Override
    public Syncer rpl2D(long Y_address, 
            float alpha, long X_address, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rpl2D(stream,
                alpha, X_address, beta, gamma,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer rpl2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rpl2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, alpha, gamma,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div">
    @Override
    public Syncer div2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D(stream,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer div2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, float alpha1, float beta1,
            long X2_address, float alpha2, float beta2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D_deltaX(stream,
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, alpha1, beta1, 
                X2_address, alpha2, beta2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div2D: row, field">
    @Override
    public Syncer div2D_row(long Y_address, 
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2, 
            float gamma, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D_row(stream,
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2, 
                gamma, row_lengthv,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer div2D_field(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2, 
            float gamma, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D_field(stream, 
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2,
                gamma, row_lengthv,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: div2D, (alpha*X1 + beta1) / (alpha2*X2 + beta2)">
  
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="add_div: row, field">
    @Override
    public Syncer linear2_div2D_row(long Y_address,
            long X1_address,
            long X2_address, long X3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.add_div2D_row(stream,
                X1_address,
                X2_address,
                X3_address, row_lengthv,
                alpha, beta, gamma, delta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2_div2D_field(long Y_address,
            long X1_address, 
            long X2_address, long X3_address, int row_lengthv, 
            float alpha, float beta, float gamma, float delta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.add_div2D_field(stream, 
                X1_address, 
                X2_address,
                X3_address, row_lengthv,
                alpha, beta, gamma, delta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="mul_squareDiv2D">
    @Override
    public Syncer mul_squareDiv2D(long Y_address, 
            float alpha1, long X1_address, float beta1, 
            float alpha2, long X2_address, float beta2, 
            float alpha3, long X3_address, float beta3, 
            float gamma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.mul_squareDiv2D(stream, 
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                alpha3, X3_address, beta3, 
                gamma, Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, sqrt">
    @Override
    public Syncer sign2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sign2D(stream,
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer ceil2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.ceil2D(stream,
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer floor2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.floor2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: abs(Absolute)">
    @Override
    public Syncer abs2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.abs2D(stream,
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer abs2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.abs2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    @Override
    public Syncer zero_nan2D(long Y_address, 
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.zero_nan2D(stream, 
                X_address, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sqrt2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqrt2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sqrt_quadratic2_2D(long Y_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqrt_quadratic_dual2D(stream, 
                X1_address, X2_address, 
                k11, k12, k22, 
                k1, k2, C,
                Y_address,
                lengthv, width, stride);
         return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="min, max, clip"> 
    //<editor-fold defaultstate="collapsed" desc="min, min_dual">
    @Override
    public Syncer min2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmin,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.min2D(stream, 
                alpha, X_address, beta, vmin,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer min2_2D(long Y_address, 
            float alpha1, long X1_address, float beta1, 
            float alpha2, long X2_address, float beta2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.min_dual2D(stream,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="max, max_dual">
    @Override
    public Syncer max2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmax,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.max2D(stream, 
                alpha, X_address, beta, vmax,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer max2_2D(long Y_address,
            float alpha1, long X1_address, float beta1, 
            float alpha2, long X2_address, float beta2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.max_dual2D(stream, 
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>

    @Override
    public Syncer clip2D(long Y_address, 
            float alpha, long X_address, float beta,
            float vmin, float vmax,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.clip2D(stream,
                alpha, X_address, beta,
                vmin, vmax,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    @Override
    public Syncer exp2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.exp2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    //<editor-fold defaultstate="collapsed" desc="BP: log">
    @Override
    public Syncer log2D(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.log2D(stream, 
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer log2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.log2D_deltaX(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address, alpha,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    @Override
    public Syncer relu2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.relu2D(stream,
                X_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer relu2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.relu2D_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer relu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.relu2D_deltaX_v2(stream, 
                deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    @Override
    public Syncer leakyRelu2D(long Y_address,
            long X_address, float k, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.leakyRelu2D(stream,
                X_address, k, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer leakyRelu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address, float k,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.leakyRelu2D_deltaX_v1(stream,
                deltaX_address,
                deltaY_address, 
                Y_address, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer leakyRelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address, float k,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.leakyRelu2D_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address,
                X_address, k,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    @Override
    public Syncer elu2D(long Y_address,
            long X_address, float alpha, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.elu2D(stream,
                X_address, alpha, k, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float k,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.elu2D_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address, 
                Y_address, alpha, k,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer elu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address, float alpha, float k, //V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.elu2D_deltaX_v2(stream,
                deltaX_address,
                deltaY_address,
                X_address, alpha, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    @Override
    public Syncer softPlus2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.softPlus2D(stream, 
                X_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer softPlus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.softPlus2D_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer softPlus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.softPlus2D_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                X_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: gelu">
    @Override
    public Syncer gelu2D(long Y_address, 
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.gelu2D(stream,
                X_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer gelu2D_deltaX(long deltaX_address, 
            long deltaY_address,
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.gelu2D_deltaX(stream, 
                deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_leakyRelu (relu)">
    @Override
    public Syncer linear2_relu2D(long Y_address, 
            long X1_address, long X2_address, 
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.linear_dual2D_with_relu(stream,
                X1_address, X2_address,
                alpha, beta, gamma, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_leakyRelu2D(long Y_address, 
            long X1_address, long X2_address, 
            float alpha, float beta, float gamma, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.linear_dual2D_with_leakyRelu(stream,
                X1_address, X2_address,
                alpha, beta, gamma, k, 
                Y_address,
                lengthv, width, stride); 
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2_leakyRelu2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.linear_dual2D_with_leakyRelu_deltaX_v1(stream,
                deltaX1_address, deltaX2_address,
                deltaY_address,
                Y_address, 
                alpha, beta, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2_leakyRelu2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.linear_dual2D_with_leakyRelu_deltaX_v2(stream,
                deltaX1_address, deltaX2_address,
                deltaY_address,
                X1_address, X2_address,
                alpha, beta, gamma, k,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_elu2D">
    @Override
    public Syncer linear2_elu2D(long Y_address, 
            long X1_address, long X2_address, 
            float alpha, float beta, float gamma, 
            float theta, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(theta, k);
        Cuda_function.linear_dual2D_with_function(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                func.type, func.params, func.params.length, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2_elu2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta,
            float theta, float k, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(theta, k);
        Cuda_function.linear_dual2D_with_function_deltaX_v1(stream,
                deltaX1_address, deltaX2_address,
                deltaY_address, 
                Y_address,
                alpha, beta, 
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2_elu2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address, 
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma,
            float theta, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(theta, k);
        Cuda_function.linear_dual2D_with_function_deltaX_v2(stream, 
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, X2_address,
                alpha, beta, gamma,
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_softplus">
    @Override
    public Syncer linear2_softplus2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.linear_dual2D_with_function(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                func.type, func.params, func.params.length, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_softplus2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.linear_dual2D_with_function_deltaX_v1(stream,
                deltaX1_address, deltaX2_address,
                deltaY_address, 
                Y_address,
                alpha, beta, 
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_softplus2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.linear_dual2D_with_function_deltaX_v2(stream, 
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, X2_address,
                alpha, beta, gamma,
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_gelu">
    @Override
    public Syncer linear2_gelu2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_function.linear_dual2D_with_function(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                func.type, func.params, func.params.length, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_gelu2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_function.linear_dual2D_with_function_deltaX_v2(stream, 
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, X2_address,
                alpha, beta, gamma,
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    @Override
    public Syncer sigmoid2D(long Y_address,
            long X_address,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid2D(stream, 
                X_address,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid2D_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sigmoid2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid2D_deltaX_v2(stream, 
                deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    @Override
    public Syncer tanh2D(long Y_address,
            long X_address,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.tanh2D(stream, 
                X_address,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer tanh2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tanh2D_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer tanh2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tanh2D_deltaX_v2(stream, 
                deltaX_address,
                deltaY_address, 
                X_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    @Override
    public Syncer softmax2D(long Y_address, 
            long X_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length :  field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV   = core.malloc(V_lengthv);
        long[] blockMax = core.malloc(field_lengthv);
        long expXm_max_rowSum = blockV[1], maxX = blockMax[1];//expXm_max = sumEachRow: exp(X - maxX)
      
        Cuda_reduce.row_max(stream,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
       
        Cuda_reduce.row_softmax(stream, //Y = exp(X - M), V = sumEachRow(Y) = sum(exp(X-M))
                X_address, maxX,
                Y_address, 
                field_length, row_lengthv, 
                expXm_max_rowSum,//result: expXm_max = sumEachRow: exp(X - maxX)
                width, stride, 1);
        
        Cuda_function.div2D_field(stream, //final value: Y -> Y / V = expX / sumOfEachRow(Y)
                1.0f, Y_address       , 0.0f, 
                1.0f, expXm_max_rowSum, 0.0f, 
                0.0f, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
       
        return new StreamBlock2Syncer(streamPool, stream, core, blockV, blockMax);
    }
    
    @Override
    public Syncer softmax2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length :  field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV = core.malloc(V_lengthv);
        long deltaY_Y_rowSum = blockV[1];//deltaY_Y_rowSum = sumEachRow: deltaY * Y
        
        //deltaY_Y_rowSum is the result and buffer for this reduction
        Cuda_reduce.row_quadratic_dual(stream, 
                deltaY_address, Y_address,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 
                field_length, row_lengthv,
                deltaY_Y_rowSum,//buffer
                deltaY_Y_rowSum,//result
                width, stride, 1);
        
        Cuda_function.softmax2D_deltaX(stream, 
                deltaX_address,
                deltaY_address, Y_address, 
                deltaY_Y_rowSum, row_lengthv, 
                lengthv, width, stride);
        
        return new StreamBlockSyncer(streamPool, stream, core, blockV);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: logsoftmax">  
    @Override
    public Syncer logsoftmax2D(long Y_address, 
            long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
       
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length :  field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV = core.malloc(V_lengthv);
        long[] blockMax = core.malloc(field_lengthv);
        long expXm_max_rowSum = blockV[1],  maxX = blockMax[1];//expXm_max_rowSum = sumEachRow: exp(X - maxX)
        
        Cuda_reduce.row_max(stream,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
         
        Cuda_reduce.row_softmaxCrossEntropy_stage1(stream, 
                X_address, maxX,
                field_length, row_lengthv, 
                expXm_max_rowSum,//expXm_max_rowSum = sumEachRow: exp(X - maxX)
                width, stride, 1);
         
        Cuda_function.logsoftmax2D(stream,
                X_address, maxX, 
                expXm_max_rowSum, 
                row_lengthv, 
                Y_address,
                lengthv, width, stride);
        
        return new StreamBlock2Syncer(streamPool, stream, core, blockV, blockMax);
    }

    @Override
    public Syncer logsoftmax2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length : field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV = core.malloc(V_lengthv);
        long deltaY_rowSum = blockV[1];//Y_rowSum = sumEachRow: Y[i]
        
        //Y_rowSum is the result and buffer for this reduction
        Cuda_reduce.row_linear(stream, 
                deltaY_address, 1.0f, 0.0f, 
                field_length, row_lengthv, 
                deltaY_rowSum, //buffer
                deltaY_rowSum, //result
                width, stride, 1);
        
        Cuda_function.logsoftmax2D_deltaX(stream,
                deltaX_address, 
                deltaY_address, 
                Y_address, 
                deltaY_rowSum, row_lengthv,
                lengthv, width, stride);
        
        return new StreamBlockSyncer(streamPool, stream, core, blockV);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_sigmoid">
    @Override
    public Syncer linear2_sigmoid2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.linear_dual2D_with_function(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                func.type, func.params, func.params.length, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_sigmoid2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.linear_dual2D_with_function_deltaX_v1(stream,
                deltaX1_address, deltaX2_address,
                deltaY_address, 
                Y_address,
                alpha, beta, 
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_sigmoid2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.linear_dual2D_with_function_deltaX_v2(stream, 
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, X2_address,
                alpha, beta, gamma,
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_tanh">
    @Override
    public Syncer linear2_tanh2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.linear_dual2D_with_function(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                func.type, func.params, func.params.length, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_tanh2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.linear_dual2D_with_function_deltaX_v1(stream,
                deltaX1_address, deltaX2_address,
                deltaY_address, Y_address,
                alpha, beta, 
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer linear2_tanh2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.linear_dual2D_with_function_deltaX_v2(stream, 
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, X2_address,
                alpha, beta, gamma,
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin">
    @Override
    public Syncer sin2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sin2D(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sin2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta, 
            int lengthv, int width, int stride) {
        long stream = streamPool.getStream();
        Cuda_function.sin2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address, 
                X_address, alpha, beta, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tan">
    @Override
    public Syncer tan2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tan2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer tan2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float alpha,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tan2D_deltaX(stream,
                deltaX_address, 
                deltaY_address,
                Y_address, alpha,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: csc">
    @Override
    public Syncer csc2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.csc2D(stream,
                alpha, X_address, beta, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer csc2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.csc2D_deltaX(stream,
                deltaX_address, deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin2D">
    @Override
    public Syncer arcsin2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arcsin2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer arcsin2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arcsin2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, alpha,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arctan2D">
    @Override
    public Syncer arctan2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arctan2D(stream, 
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer arctan2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arctan2D_deltaX(stream,
                deltaX_address,
                deltaY_address,
                Y_address, alpha, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin2D">
    @Override
    public Syncer halfSin2D(long dY_address, 
            float Amp, float alpha, long dX_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.halfSin2D(stream,
                Amp, alpha, dX_address, beta,
                dY_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer halfSin2D_deltaX(long d_deltaX_address,
            long d_deltaY_address, 
            long dY_address, float Amp, float alpha, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.halfSin2D_deltaX(stream,
                d_deltaX_address, 
                d_deltaY_address,
                dY_address, Amp, alpha, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss functions">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    @Override
    public Syncer L1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.L1_2D(stream, 
                Y_address, Yh_address,
                L_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer L1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.L1_2D_deltaYh(stream, 
                Y_address, Yh_address, 
                deltaYh_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    @Override
    public Syncer L2_2D(long L_address, 
            long Y_address, long Yh_address, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.L2_2D(stream,
                Y_address, Yh_address,
                L_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer L2_2D_deltaYh(long deltaYh_address, 
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.L2_2D_deltaYh(stream, 
                Y_address, Yh_address, 
                deltaYh_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: smoothL1">
    @Override
    public Syncer smoothL1_2D(long L_address,
            long Y_address, long Yh_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.smoothL1_2D(stream, 
                Y_address, Yh_address, 
                L_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer smoothL1_2D_deltaYh(long deltaYh_address, 
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.smoothL1_2D_deltaYh(stream,
                Y_address, Yh_address,
                deltaYh_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    @Override
    public Syncer binaryCrossEntropy2D(long L_address, 
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.binaryCrossEntropy2D(stream, 
                Y_address, Yh_address,
                alpha, beta,
                L_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer binaryCrossEntropy2D_deltaYh(long deltaYh_address, 
            long Y_address, long Yh_address, 
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.binaryCrossEntropy2D_deltaYh(stream,
                Y_address, Yh_address,
                alpha, beta,
                deltaYh_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    @Override
    public Syncer sigmoid_binaryCrossEntropy2D(long L_address, 
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid_binaryCrossEntropy2D(stream,
                Y_address, X_address,
                alpha, beta,
                L_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sigmoid_binaryCrossEntropy_deltaX(long deltaX_address,
            long Y_address, long X_address, 
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid_binaryCrossEntropy2D_deltaX(stream, 
                Y_address, X_address, 
                alpha, beta,
                deltaX_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    @Override
    public Syncer crossEntropy2D(long L_address, 
            long Y_address, long Yh_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.crossEntropy2D(stream, 
                Y_address, Yh_address, 
                L_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer crossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.crossEntropy2D_deltaYh(stream, 
                Y_address, Yh_address,
                deltaYh_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmaxCrossEntropy">
    //<editor-fold defaultstate="collapsed" desc="forward prop">
    @Override
    public Syncer softmax_crossEntropy2D(long L_address, 
            long Y_address, long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length : field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV   = core.malloc(V_lengthv);
        long[] blockMax = core.malloc(field_lengthv);
        long expXm_max_rowSum = blockV[1], maxX = blockMax[1];
        
        Cuda_reduce.row_max(stream,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
       
        Cuda_reduce.row_softmaxCrossEntropy_stage1(stream, 
                X_address, maxX,
                field_length, row_lengthv, 
                expXm_max_rowSum,
                width, stride, 1);
        
        //L = -Y * (X - maxX) + U + (Y - 1)*log(U - exp(X - M))
        Cuda_function.softmax_crossEntropy2D(stream,
                Y_address, X_address, 
                maxX, expXm_max_rowSum, row_lengthv,
                L_address, 
                lengthv, width, stride);
        
        return new StreamBlock2Syncer(streamPool, stream, core, blockV, blockMax);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward prop">
    @Override//Yh - Y = softmax(X) - Y
    public Syncer softmax_crossEntropy2D_deltaX(long deltaX_address, 
            long Y_address, long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream1 = streamPool.getStream();
        long stream2 = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length : field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV        = core.malloc(V_lengthv);
        long[] blockMax      = core.malloc(field_lengthv);
        long[] blockY_rowSum = core.malloc(V_lengthv);
        long expXm_max_rowSum = blockV[1];
        long maxX             = blockMax[1];
        long Y_rowSum         = blockY_rowSum[1];
        
        //Stage1: compute maxX, expXm_max_rowSum, Y_rowSum======================
        Cuda_reduce.row_max(stream1,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
       
        Cuda_reduce.row_softmaxCrossEntropy_stage1(stream1, 
                X_address, maxX,
                field_length, row_lengthv,//expXm_max_rowSum = sumEachRow: exp(X - maxX)
                expXm_max_rowSum,
                width, stride, 1);
        
        Cuda_reduce.row_linear(stream2, 
                Y_address, 1.0f, 0.0f, 
                field_length, row_lengthv,//Y_rowSum = sumEachRow: Y
                Y_rowSum, Y_rowSum,
                width, stride, 1);
        
        //Stage2: deltaX = YrowSum * [exp(X - maxX) / expXm_max_rowSum] - Y=====
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, stream1);
        Cuda.eventRecord(event, stream2);
        Cuda.streamWaitEvent_default(stream1, event);
        
        Cuda_function.softmax_crossEntropy2D_deltaX(stream1,
                Y_address, X_address,
                maxX, expXm_max_rowSum, 
                Y_rowSum, row_lengthv, 
                deltaX_address, //deltaX = YrowSum * [exp(X - maxX) / expXm_max_rowSum] - Y
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block3Syncer_1(streamPool, stream1, stream2,
                core, blockV, blockMax, blockY_rowSum);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="BP: affine">
    @Override
    public Syncer affine2D(long Y_address,
            long X_address,
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.affine2D_row(stream, 
                X_address, 
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer affine2D_deltaA_v1(long deltaA_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long block[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1];//V[HV: nextN, M: row_lengthv]
        }
        
        Cuda_reduce.field_affine_deltaA_v1(stream,
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer affine2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,//V1: holdY(), Y is not changed
            long Y_address,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address, 
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address,
            long X_address,//V2 holdX(), X is not changed
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_affine_deltaAB_v2(stream1, stream2, 
                deltaY_address, 
                X_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address, 
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_leakyRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer affine_relu2D(long Y_address, 
            long X_address, 
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.affine2D_row_with_relu(stream, 
                X_address,
                A_address, B_address, row_lengthv, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine_leakyRelu2D(long Y_address,
            long X_address,
            long A_address, long B_address,
            int row_lengthv, float k,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.affine2D_row_with_leakyRelu(stream, 
                X_address, 
                A_address, B_address, row_lengthv, 
                Y_address, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaX}">
    @Override
    public Syncer affine_leakyRelu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.affine2D_row_with_leakyRelu_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, k,
                A_address, row_lengthv,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine_leakyRelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float k,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.affine2D_row_with_leakyRelu_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, k,
                X_address, 
                A_address, 
                B_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaA, deletaB}">
    @Override
    public Syncer affine_leakyRelu2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_affine_with_leakyRelu_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address, k, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine_leakyRelu2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address, float k,
            long X_address,//V2 holdX(), X is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_affine_with_leakyRelu_deltaAB_v2(stream1, stream2, 
                deltaY_address, k,
                X_address, 
                A_address, B_address,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_elu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer affine_elu2D(long Y_address, 
            long X_address, 
            long A_address, long B_address, int row_lengthv,
            float alpha, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_function.affine2D_row_with_function(stream, 
                X_address, 
                A_address, B_address, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaX}">
    @Override
    public Syncer affine_elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_function.affine2D_row_with_function_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length, 
                A_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine_elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_function.affine2D_row_with_function_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                func.type, func.params, func.params.length, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaA, deletaB}">
    @Override
    public Syncer affine_elu2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine_elu2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address, float alpha, float k,
            long X_address,//V2 holdX(), X is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_reduce.field_affine_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                A_address, B_address,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer affine_softplus2D(long Y_address, 
            long X_address, 
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.affine2D_row_with_function(stream, 
                X_address, 
                A_address, B_address, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaX}">
    @Override
    public Syncer affine_softplus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.affine2D_row_with_function_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length, 
                A_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine_softplus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.affine2D_row_with_function_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                func.type, func.params, func.params.length, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaA, deletaB}">
    @Override
    public Syncer affine_softplus2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine_softplus2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address,
            long X_address,//V2 holdX(), X is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_reduce.field_affine_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                A_address, B_address,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer affine_gelu2D(long Y_address, 
            long X_address, 
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_function.affine2D_row_with_function(stream, 
                X_address, 
                A_address, B_address, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaX}">
    @Override
    public Syncer affine_gelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_function.affine2D_row_with_function_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                func.type, func.params, func.params.length, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaA, deletaB}">
    @Override
    public Syncer affine_gelu2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address,
            long X_address,//V2 holdX(), X is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_reduce.field_affine_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                A_address, B_address,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer affine_sigmoid2D(long Y_address, 
            long X_address, 
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.affine2D_row_with_function(stream, 
                X_address, 
                A_address, B_address, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaX}">
    @Override
    public Syncer affine_sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.affine2D_row_with_function_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length, 
                A_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine_sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.affine2D_row_with_function_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                func.type, func.params, func.params.length, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaA, deletaB}">
    @Override
    public Syncer affine_sigmoid2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine_sigmoid2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address,
            long X_address,//V2 holdX(), X is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_reduce.field_affine_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                A_address, B_address,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer affine_tanh2D(long Y_address, 
            long X_address, 
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.affine2D_row_with_function(stream, 
                X_address, 
                A_address, B_address, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaX}">
    @Override
    public Syncer affine_tanh2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.affine2D_row_with_function_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length, 
                A_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine_tanh2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.affine2D_row_with_function_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                func.type, func.params, func.params.length, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation {deltaA, deletaB}">
    @Override
    public Syncer affine_tanh2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine_tanh2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address,
            long X_address,//V2 holdX(), X is not changed
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_reduce.field_affine_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                A_address, B_address,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer sqBatchNorm2D(long Y_address, 
            long X_address, 
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqBatchNorm2D_row(stream, 
                X_address, 
                X_mean_address, 
                X_sqmean_address, eps, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sqBatchNorm2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqBatchNorm_affined2D_row(stream, 
                X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v2(stream1, stream2, 
                deltaY_address, 
                Y_address,//Y = X_norm
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.sqBatchNorm2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.sqBatchNorm2D_row_deltaX_v2(stream1,
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Override
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address, //V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is end
        
        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, 
                deltaXp1, 
                deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, 
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    @Override
    public Syncer sqBatchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mem stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1]; 
        }

        Cuda_reduce.field_sqBatchNorm_deltaA_v2(stream, 
                deltaY_address, 
                dX_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer sqBatchNorm2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length); deltaA_buf_address = blockA[1];
            blockB = core.malloc(mem_length); deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2,
                deltaY_address,
                dX_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB)); 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer sqBatchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }

        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2, 
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    
    @Override
    public Syncer sqBatchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
       
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm2D(long Y_address, 
            long X_address, 
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm2D_row(stream, 
                X_address, 
                X_mean_address, 
                X_var_address, eps, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer batchNorm2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm_affined2D_row(stream, 
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v2(stream1, stream2, 
                deltaY_address, 
                Y_address,//Y = X_norm
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA: deltaY * Y
                deltaXp1, deltaXp1,//deltaXp1 = deltaB: deltaY
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address, 
                X_var_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_deltaX_v2(stream1,
                deltaY_address, X_address, 
                X_mean_address, X_var_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Override
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA: deltaY * (Y - B) / A
                deltaXp1, deltaXp1,//deltaXp1 = deltaB: deltaY
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is end
        
        Cuda_function.batchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                X_var_address, eps,
                A_address, B_address, 
                deltaXp1, 
                deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA: deltaY * (X - X_mean) * X_rstd
                deltaXp1, deltaXp1,//deltaXp1 = deltaB: deltaY
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, 
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    @Override
    public Syncer batchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mem stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1]; 
        }

        Cuda_reduce.field_batchNorm_deltaA_v2(stream, 
                deltaY_address, 
                dX_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,//deltaY * (X - X_mean) * X_rstd
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer batchNorm2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length); deltaA_buf_address = blockA[1];
            blockB = core.malloc(mem_length); deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2,
                deltaY_address,
                dX_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB)); 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }

        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2, 
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    
    @Override
    public Syncer batchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
       
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address, X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_leakyRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation (relu)"> 
    @Override
    public Syncer batchNorm_relu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm2D_row_with_relu(stream,
                X_address, 
                X_mean_address,
                X_var_address, eps, row_lengthv, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_relu2D(long Y_address, 
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm_affined2D_row_with_relu(stream,
                X_address, 
                X_mean_address,
                X_var_address, eps, 
                A_address, B_address, row_lengthv,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward-propagation (leakyRelu)">
    @Override
    public Syncer batchNorm_leakyRelu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, float k, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm2D_row_with_leakyRelu(stream, 
                X_address,
                X_mean_address,
                X_var_address, eps, row_lengthv,
                Y_address, k,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_leakyRelu2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm_affined2D_row_with_leakyRelu(stream,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
        
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm_leakyRelu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_batchNorm_with_leakyRelu_deltaXp_v1(stream1, stream2, 
                deltaY_address, 
                Y_address, k, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2, 
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_leakyRelu_deltaX_v1(stream1, 
                deltaY_address,
                Y_address, k, 
                X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm_leakyRelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_batchNorm_with_leakyRelu_deltaXp_v2(stream1, stream2,
                deltaY_address, k, 
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_leakyRelu_deltaX_v2(stream1,
                deltaY_address, k,
                X_address, 
                X_mean_address, X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
      
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm_leakyRelu2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1 
            long deltaB_address,//result2
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }

        Cuda_reduce.field_affine_with_leakyRelu_deltaAB_v1(stream1, stream2,
                deltaY_address, Y_address, k,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_with_leakyRelu_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address, k, 
                X_var_address, eps, 
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer batchNorm_leakyRelu2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, float k, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        Cuda_reduce.field_batchNorm_with_leakyRelu_deltaAB_v2(stream1, stream2, 
                deltaY_address, k, 
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address,
                field_length, row_lengthv,
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_with_leakyRelu_deltaX_v2(stream1,
                deltaY_address, k, 
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_elu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm_elu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, float alpha, float k, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_function.batchNorm2D_row_with_function(stream, 
                X_address,
                X_mean_address,
                X_var_address, eps, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_elu2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, float alpha, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_function.batchNorm_affined2D_row_with_function(stream,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm_elu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v1(stream1, stream2, 
                deltaY_address, 
                Y_address,
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2, 
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address,
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm_elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v2(stream1, stream2,
                deltaY_address, 
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
      
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm_elu2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1 
            long deltaB_address,//result2
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address, Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer batchNorm_elu2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_reduce.field_batchNorm_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address,
                field_length, row_lengthv,
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm_softplus2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.batchNorm2D_row_with_function(stream, 
                X_address,
                X_mean_address,
                X_var_address, eps, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_softplus2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_function.batchNorm_affined2D_row_with_function(stream,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm_softplus2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v1(stream1, stream2, 
                deltaY_address, 
                Y_address,
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2, 
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address,
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm_softplus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v2(stream1, stream2,
                deltaY_address, 
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
      
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm_softplus2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1 
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address, Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer batchNorm_softplus2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_reduce.field_batchNorm_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address,
                field_length, row_lengthv,
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm_gelu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_function.batchNorm2D_row_with_function(stream, 
                X_address,
                X_mean_address,
                X_var_address, eps, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_gelu2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_function.batchNorm_affined2D_row_with_function(stream,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm_gelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v2(stream1, stream2,
                deltaY_address, 
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
      
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm_gelu2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_reduce.field_batchNorm_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address,
                field_length, row_lengthv,
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm_sigmoid2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.batchNorm2D_row_with_function(stream, 
                X_address,
                X_mean_address,
                X_var_address, eps, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_sigmoid2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_function.batchNorm_affined2D_row_with_function(stream,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm_sigmoid2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v1(stream1, stream2, 
                deltaY_address, 
                Y_address,
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2, 
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address,
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm_sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v2(stream1, stream2,
                deltaY_address, 
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
      
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm_sigmoid2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1 
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address, Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer batchNorm_sigmoid2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_reduce.field_batchNorm_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address,
                field_length, row_lengthv,
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm_tanh2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.batchNorm2D_row_with_function(stream, 
                X_address,
                X_mean_address,
                X_var_address, eps, row_lengthv,
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer batchNorm_tanh2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_function.batchNorm_affined2D_row_with_function(stream,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address,
                func.type, func.params, func.params.length,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm_tanh2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v1(stream1, stream2, 
                deltaY_address, 
                Y_address,
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2, 
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address,
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm_tanh2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_reduce.field_batchNorm_with_function_deltaXp_v2(stream1, stream2,
                deltaY_address, 
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp1,
                deltaXp2, deltaXp2,
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
      
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm_tanh2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1 
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_reduce.field_affine_with_function_deltaAB_v1(stream1, stream2,
                deltaY_address, Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                func.type, func.params, func.params.length,
                X_var_address, eps, 
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer batchNorm_tanh2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
        
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_reduce.field_batchNorm_with_function_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address,
                field_length, row_lengthv,
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1,
                func.type, func.params, func.params.length);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_with_function_deltaX_v2(stream1,
                deltaY_address,
                func.type, func.params, func.params.length,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    @Override
    public Syncer layerNorm2D(long Y_address, 
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.layerNorm2D_row(stream,
                X_address,
                X_mean_address, X_sqmean_address, eps, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.layerNorm_affined2D_row(stream,
                X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deltaX">
    @Override
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_deltaXp_v1(stream1, stream2,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv, 
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm2D_row_deltaX_v1(stream1,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps,
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_deltaXp_v2(stream1, stream2, 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv,
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm2D_row_deltaX_v2(stream1,
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): deltaX">
    @Override
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;//[nextM, SV]
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_affined_deltaXp_v1(stream1, stream2,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm_affined2D_row_deltaX_v1(stream1,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
         
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_affined_deltaXp_v2(stream1, stream2, 
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps,
                A_address, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address,
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): {deltaA, deltaB}">
    @Override
    public Syncer layerNorm2D_deltaA_v2(long deltaA_address, 
            long deltaY_address, long dX_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mem stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1];//V[HV: nextN, M: row_lengthv]
        }

        Cuda_reduce.field_layerNorm_deltaA_v2(stream,
                deltaY_address, dX_address,
                X_mean_address, 
                X_sqmean_address, eps,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer layerNorm2D_deltaAB_v2(long deltaA_address, long deltaB_address,
            long deltaY_address, long dX_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_layerNorm_deltaAB_v2(stream1, stream2,
                deltaY_address, dX_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot, pix2tensor">
    @Override
    public Syncer onehot2D_row_int8(long Y_address, 
            long X_address, 
            float alpha, float beta, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.onehot2D_row_char(stream, X_address, 
                alpha, beta, row_lengthv,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer onehot2D_row_int32(long Y_address, 
            long X_address, 
            float alpha, float beta, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.onehot2D_row_int(stream, X_address,
                alpha, beta, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer pix2tensor2D(long Y_address, 
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.pix2tensor2D(stream, X_address,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer tensor2pix2D(long Y_address, 
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tensor2pix2D(stream, X_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="repeat functions">
    @Override
    public Syncer repeat_linear2D_row(long Y_address, 
            long X_address, int row_lengthv, 
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.repeat_linear2D_row(stream, 
                X_address, row_lengthv,
                alpha, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer repeat_quadratic2D_row(long Y_address, 
            long X_address, int row_lengthv, 
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.repeat_quadratic2D_row(stream, 
                X_address, row_lengthv, 
                alpha, beta, gamma,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="SGD">
    @Override
    public Syncer sgd2D(long W_address, 
            long[] gradients, float lr, 
            int lengthv, int width, int stride)  
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.quadratic_dual2D(stream, 
                    W_address, gradient,
                    0, 0, 0, 
                    1.0f, -lr, 0,
                    W_address, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    @Override
    public Syncer sgdmn2D(long W_address, 
            long V_address, float momentum, float dampen, float nesterov, 
            long deltaW_address, float lr, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sgdmn2D(stream, W_address, 
                V_address, momentum, dampen, nesterov,
                deltaW_address, lr, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sgdmn2D(long W_address,
            long V_address, float momentum, float dampen, float nesterov, 
            long[] gradients, float lr, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.sgdmn2D(stream, W_address,
                    V_address, momentum, dampen, nesterov, 
                    gradient, lr, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SGDMN_decay">
    @Override
    public Syncer sgdmn2D_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sgdmn2D_decay(stream, W_address, 
                V_address, momentum, dampen, nesterov,
                deltaW_address, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sgdmn2D_decay(long W_address, 
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.sgdmn2D_decay(stream, W_address,
                    V_address, momentum, dampen, nesterov, 
                    gradient, lr, 
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Moumentum">
    @Override
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.momentum2D(stream,
                W_address, 
                V_address, a1, a2, 
                deltaW_address, lr_t, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long[] gradients, float lr_t,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.momentum2D(stream,
                    W_address,
                    V_address, a1, a2,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Moumentum_decay">
    @Override
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.momentum2D_decay(stream, 
                W_address, 
                V_address, a1, a2, 
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.momentum2D_decay(stream,
                    W_address,
                    V_address, a1, a2,
                    gradient, lr_t,
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    @Override
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rmsprop2D(stream, 
                W_address, 
                S_address, a1, a2, eps_t, 
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.rmsprop2D(stream,
                    W_address,
                    S_address, a1, a2, eps_t,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop_decay">
    @Override
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t, 
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rmsprop2D_decay(stream, 
                W_address, 
                S_address, a1, a2, eps_t, 
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t, 
            long[] gradients, float lr_t,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.rmsprop2D_decay(stream,
                    W_address,
                    S_address, a1, a2, eps_t,
                    gradient, lr_t, 
                    L1coef, L2coef, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="Adam">
    @Override
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        Cuda_function.adam2D(stream,
                W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                deltaW_address, lr_t, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adam2D_type2(long W_address, 
            long V_address, float a1, float a2, float Uv,
            long S_address, float b1, float b2, float eps, float Us,
            long deltaW_address, float lr, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.adam2D_type2(stream, W_address,
                V_address, a1, a2, Uv,
                S_address, b1, b2, eps, Us,
                deltaW_address, lr, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adam2D(stream,
                    W_address,
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam_decay">
    @Override
    public Syncer adam2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adam2D_decay(stream, 
                W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adam2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.adam2D_decay(stream, 
                W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                gradient, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam_AMSgrad">
    @Override
    public Syncer adam_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t,
            long Smax_address,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adam_amsgrad2D(stream,
                W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                Smax_address, 
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adam_amsgrad2D(long W_address,
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t,
            long Smax_address, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adam_amsgrad2D(stream,
                    W_address, 
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t, 
                    Smax_address, 
                    gradient, lr_t, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam_AMSgrad_decay">
    @Override
    public Syncer adam_amsgrad2D_decay(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adam_amsgrad2D_decay(stream,
                W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                Smax_address,
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adam_amsgrad2D_decay(long W_address,
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.adam_amsgrad2D_decay(stream,
                    W_address, 
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t,
                    Smax_address, 
                    gradient, lr_t,
                    L1coef, L2coef, 
                    lengthv, width, stride);
        } 
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    @Override
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adamax2D(stream,
                W_address,
                V_address, a1, a2,
                S_address, b1, eps,
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.adamax2D(stream,
                    W_address,
                    V_address, a1, a2,
                    S_address, b1, eps,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax_decay">
    @Override
    public Syncer adamax2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {        
        long stream = streamPool.getStream();
        Cuda_function.adamax2D_decay(stream,
                W_address, 
                V_address, a1, a2, 
                S_address, b1, eps,
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamax2D_decay(long W_address,
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps,
            long[] gradients, float lr_t,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamax2D_decay(stream,
                    W_address,
                    V_address, a1, a2,
                    S_address, b1, eps,
                    gradient, lr_t,
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    @Override
    public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adamW2D(stream, W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t,
                deltaW_address, lr_t, lr,
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamW2D(stream, W_address,
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t,
                    gradient, lr_t, lr,
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="AdamW_AMSgrad"> 
    @Override
    public Syncer adamW_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address, 
            long deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adamW_amsgrad2D(stream, W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                Smax_address, 
                deltaW_address, lr_t, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamW_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t,
            long Smax_address, 
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamW_amsgrad2D(stream, W_address, 
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t,
                    Smax_address, 
                    gradient, lr_t, lr, 
                    L1coef, L2coef, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    @Override
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        Cuda_function.adamod2D(stream, W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                G_address, c1, c2, 
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamod2D(stream, W_address, 
                    V_address, a1, a2, 
                    S_address, b1, b2, eps_t,
                    G_address, c1, c2,
                    gradient, lr_t, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod_decay">
     @Override
    public Syncer adamod2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        Cuda_function.adamod2D_decay(stream, W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                deltaW_address, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamod2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        for(long gradient : gradients) {
            Cuda_function.adamod2D_decay(stream, W_address,
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t, 
                    G_address, c1, c2, 
                    gradient, lr_t, 
                    L1coef, L2coef, 
                    lengthv, width, stride);
        } 
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    //<editor-fold defaultstate="collapsed" desc="bernouli">
    @Override
    public Syncer bernouli2D(long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.bernouli2D(stream,
                X_address, 
                seed,
                p, v1, v2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer bernouli_mul2D(long Y_address, long R_address, 
            long X_address,
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.bernouli_mul2D(stream,
                X_address, R_address,
                Y_address,
                seed,
                p, v1, v2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer leakyRelu_bernouli_mul2D(long Y_address, long R_address,
            long X_address, 
            float k, int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.leakyRelu_bernouli_mul2D(stream, 
                X_address, R_address,
                Y_address, k,
                seed,
                p, v1, v2,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer elu_bernouli_mul2D(long Y_address, long R_address,
            long X_address, 
            float alpha, float k, int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.elu(alpha, k);
        Cuda_random.function_bernouli_mul2D(stream, 
                X_address, R_address,
                Y_address, 
                seed, 
                p, v1, v2, 
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer softplus_bernouli_mul2D(long Y_address, long R_address,
            long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.softplus();
        Cuda_random.function_bernouli_mul2D(stream, 
                X_address, R_address,
                Y_address, 
                seed, 
                p, v1, v2, 
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer gelu_bernouli_mul2D(long Y_address, long R_address,
            long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.gelu();
        Cuda_random.function_bernouli_mul2D(stream, 
                X_address, R_address,
                Y_address, 
                seed, 
                p, v1, v2, 
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sigmoid_bernouli_mul2D(long Y_address, long R_address,
            long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.sigmoid();
        Cuda_random.function_bernouli_mul2D(stream, 
                X_address, R_address,
                Y_address, 
                seed, 
                p, v1, v2, 
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer tanh_bernouli_mul2D(long Y_address, long R_address,
            long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        FloatFuncConfig func = FloatFunc.tanh();
        Cuda_random.function_bernouli_mul2D(stream, 
                X_address, R_address,
                Y_address, 
                seed, 
                p, v1, v2, 
                func.type, func.params, func.params.length, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="uniform">
    @Override
    public Syncer uniform2D(long X_address,
            int seed, 
            float vmin, float vmax,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.uniform2D(stream,
                X_address,
                seed, 
                vmin, vmax,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sparse_uniform2D(long X_address, 
            int seed1, int seed2,
            float p, float vmin, float vmax,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.sparse_uniform2D(stream,
                X_address, 
                seed1, seed2, 
                p, vmin, vmax, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="gaussian">
    @Override
    public Syncer gaussian2D(long X_address, 
            int seed1, int seed2,
            float mu, float sigma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.gaussian2D(stream, 
                X_address, 
                seed1, seed2, 
                mu, sigma, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sparse_gaussian2D(long X_address,
            int seed1, int seed2, int seed3,
            float p, float mu, float sigma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.sparse_gaussian2D(stream, 
                X_address, 
                seed1, seed2, seed3, 
                p, mu, sigma, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
 
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    @Override
    public Result<Float> straight_linear(long X_address, 
            float alpha, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_linear(stream, 
                X_address, alpha, beta, lengthv,
                block[1],//V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }
    
    @Override
    public Result<Float> straight_quadratic(long X_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_quadratic(stream, 
                X_address, alpha, beta, gamma, lengthv, 
                block[1],//V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }
    
     @Override
    public Result<Float> straight_max(long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_max(stream,
                X_address, lengthv,
                block[1], //V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }

    @Override
    public Result<Float> straight_min(long X_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_min(stream,
                X_address, lengthv,
                block[1], //V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }
    
    @Override
    public IndexedResult<Float> straight_max_indexed(long X_address,
            int lengthv, int width, int stride) {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block1 = core.malloc(nextLengthV);
        long[] block2 = core.malloc_int32(nextLengthV);
        
        Cuda_reduce.straight_max_indexed(stream, 
                X_address, lengthv, 
                block1[1], //V_address = block1[1]
                block2[1], //Index_address = block2[1]
                width, stride, 1);
        
        return new FloatIndexedResult(streamPool, stream, core, block1, block2);
    }

    @Override
    public IndexedResult<Float> straight_min_indexed(long X_address,
            int lengthv, int width, int stride) {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block1 = core.malloc(nextLengthV);
        long[] block2 = core.malloc_int32(nextLengthV);
        
        Cuda_reduce.straight_min_indexed(stream, 
                X_address, lengthv,
                block1[1], //V_address = block1[1]
                block2[1], //Index_address = block2[1]
                width, stride, 1);
        
        return new FloatIndexedResult(streamPool, stream, core, block1, block2);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field reduce function">
    //<editor-fold defaultstate="collapsed" desc="field_linear">
    @Override
    public Syncer field_linear(long Y_address,
            long X_address,
            float alpha, float beta,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_linear(stream, 
                X_address, alpha, beta,
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,//V_address = block[1]
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer field_linear2(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
       
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_linear_dual(stream, 
                X1_address, X2_address, 
                alpha, beta, gamma,
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,//V_address = block[1]
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_quadratic">
    @Override
    public Syncer field_quadratic(long Y_address,
            long X_address,
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_quadratic(stream, 
                X_address, alpha, beta, gamma, 
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,//V_address = block[1]
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer field_quadratic2(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_quadratic_dual(stream,
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C, 
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,//V_address = block[1]
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_linear_quadratic">
    @Override
    public Syncer field_linear_quadratic(long Y1_address, long Y2_address,
            long X_address, 
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long V1_address = 0L; long block1[] = null;
        long V2_address = 0L; long block2[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length);
            block2 = core.malloc(mem_length);
            V1_address = block1[1];
            V2_address = block2[1];
        }
        
        Cuda_reduce.field_linear_quadratic(stream1, stream2,
                X_address,
                alpha1, beta1, 
                alpha2, beta2, gamma2,
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V1_address, Y1_address,//V1 = Y1.buf
                V2_address, Y2_address,//V2 = Y2.buf
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, block1, block2));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field_var_mean">
    private boolean field_var_safe = true;
    public boolean field_var_safe() { return field_var_safe; } 
    public CudaFloat32EngineBase field_var_safe(boolean flag) { field_var_safe = flag; return this; } 
    
    //<editor-fold defaultstate="collapsed" desc="field_var_mean_safe">
    protected Syncer field_var_mean_safe(boolean unbiased,
            long X_var_address, //result0
            long X_mean_address,//result1
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride)
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        //======[stage1: mean = sum(X) / field_length]==========================
        float alpha = 1.0f / field_length;
        Cuda_reduce.field_linear(stream, 
                X_address, alpha, 0.0f, 
                field_length, row_lengthv,
                V_address, X_mean_address, 
                width, stride, 1);
        
        //======[stage2: var = sum(X - mean)^2 / field_length]==================
        if(unbiased) alpha = (float) (1.0 / (field_length - 1.0));
        Cuda_reduce.field_linear2_square_row(stream, 
                X_address, X_mean_address, 
                alpha, 1.0f, -1.0f, 0.0f, //var = alpha*(X - mean)^2
                field_length, row_lengthv,
                V_address, X_var_address, 
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_var_mean_fast">
    protected Syncer field_var_mean_fast(boolean unbiased,
            long X_var_address, //result0
            long X_mean_address,//result1
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
                
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); V1_address = block1[1];
            block2 = core.malloc(mem_length); V2_address = block2[1];
        }
        
        //======[staget1: mean = E(X), sqmean = E(X^2)]=========================
        float alpha = 1.0f / field_length;
        Cuda_reduce.field_linear_quadratic(stream1, stream2, 
                X_address,
                alpha, 0.0f,               //mean   = sum_each_field: X   / field_length
                alpha, 0.0f, 0.0f,         //sqmean = sum_each_field: X^2 / field_length
                field_length, row_lengthv, //N = field_lenth, M = row_lengthv(mod stride)
                V1_address, X_mean_address,//V1 = mean.buf
                V2_address, X_var_address, //V2 = sqmean.buf
                width, stride, 1);
        
        //======[stage2 compute var = E(X^2) - E(X)^2]==========================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait for stream1 and stream2
        
        float k = 1.0f;
        if(unbiased) k = (float) (field_length / (field_length - 1.0));
        Cuda_function.quadratic_dual_clip2D(stream1, 
                X_mean_address, X_var_address, 
                -k, 0, 0, 0, k, 0,
                0.0f, Float.MAX_VALUE,//var belongs to [0, +inf]
                X_var_address, //var = -mean^2 + sqmean
                row_lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2):
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    
    @Override
    public Syncer field_var_mean(boolean unbiased,
            long X_var_address, //result0
            long X_mean_address,//resullt1 
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        if(field_var_safe) return field_var_mean_safe(unbiased,
                X_var_address, //result0
                X_mean_address,//result1
                X_address, 
                field_length, row_lengthv,
                width, stride);
        
        return field_var_mean_fast(unbiased,
                X_var_address, //result0
                X_mean_address,//result1
                X_address, 
                field_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_std_mean">
    private boolean field_std_safe = true;
    public boolean field_std_safe() { return field_std_safe; } 
    public CudaFloat32EngineBase field_std_safe(boolean flag) { field_std_safe = flag; return this;  } 
    
    //<editor-fold defaultstate="collapsed" desc="field_std_mean_safe">
    protected Syncer field_std_mean_safe(boolean unbiased,
            long X_std_address, //result0
            long X_mean_address,//result1
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride)
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        //======[stage1: mean = sum(X) / field_length]==========================
        float alpha = 1.0f / field_length;
        Cuda_reduce.field_linear(stream, 
                X_address, alpha, 0.0f, 
                field_length, row_lengthv,
                V_address, X_mean_address, 
                width, stride, 1);
        
        //======[stage2: var = sum(X - mean)^2 / field_length]==================
        if(unbiased) alpha = (float) (1.0 / (field_length - 1.0));
        Cuda_reduce.field_linear2_square_row(stream, 
                X_address, X_mean_address, 
                alpha, 1.0f, -1.0f, 0.0f, //var = alpha*(X - mean)^2
                field_length, row_lengthv,
                V_address, X_std_address, 
                width, stride, 1);
        
        //======[stage3: std = sqrt(var)]=======================================
        Cuda_function.sqrt2D(stream, 
                1.0f, X_std_address, 0.0f, 
                X_std_address,
                row_lengthv, width, stride);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_std_mean_fast">
    public Syncer field_std_mean_fast(boolean unbiased,
            long X_std_address, //result0 
            long X_mean_address,//result1 
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
                
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); V1_address = block1[1];
            block2 = core.malloc(mem_length); V2_address = block2[1];
        }

        //staget1: mean = E(X) and sqmean = E(X^2)]=============================
        float alpha = 1.0f / field_length;
        Cuda_reduce.field_linear_quadratic(stream1, stream2, 
                X_address,
                alpha, 0.0f,               //mean   = sum_each_field: X   / field_length
                alpha, 0.0f, 0.0f,         //sqmean = sum_each_field: X^2 / field_length
                field_length, row_lengthv, //N = field_lenth, M = row_lengthv(mod stride)
                V1_address, X_mean_address,//V1 = mean.buf
                V2_address, X_std_address, //V2 = sqmean.buf
                width, stride, 1);
        
        //=====[stage2: std = sqrt(E(X^2) - E(X)^2)]============================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait for stream1 and stream2
        
        float k = 1.0f;
        if(unbiased) k = (float) (field_length / (field_length - 1.0));
        Cuda_function.sqrt_positive_quadratic_dual2D(stream1, 
                X_mean_address, X_std_address, //std = sqrt(-mean^2 + sqmean)
                -k, 0, 0, 0, k, 0,
                X_std_address, 
                row_lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return (nextN == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2) :
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    
    @Override
    public Syncer field_std_mean(boolean unbiased,
            long X_std_address, //result0 
            long X_mean_address,//result1 
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        if(field_std_safe) return field_std_mean_safe(unbiased, 
                X_std_address, //result0
                X_mean_address,//result1
                X_address,
                field_length, row_lengthv, 
                width, stride);
        
        return field_std_mean_fast(unbiased,
                X_std_address, //result0
                X_mean_address,//result1
                X_address, 
                field_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field_max, min">
    @Override
    public Syncer field_max(long Y_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride)
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_max(stream, 
                X_address,
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride) 
                V_address, Y_address,
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer field_min(long Y_address,
            long X_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_min(stream, 
                X_address,
                field_length, row_lengthv, 
                V_address, Y_address,
                width, stride, 1);
       
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer field_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L; 
        long[] block1 = null, block2 = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;
            block1 = core.malloc(mem_length);//V[HV: nextN, M: row_lengthv]
            block2 = core.malloc_int32(mem_length);
            V_address      = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.field_max_indexed(stream,
                X_address, 
                field_length, row_lengthv, 
                V_address, VIndex_address,
                Y_address, Index_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }

    @Override
    public Syncer field_min_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L; 
        long[] block1 = null, block2 = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;
            block1 = core.malloc(mem_length);//V[HV: nextN, M: row_lengthv]
            block2 = core.malloc_int32(mem_length);
            V_address      = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.field_min_indexed(stream,
                X_address, 
                field_length, row_lengthv, 
                V_address, VIndex_address, 
                Y_address, Index_address,
                width, stride, 1);
     
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="center reduce function">
    @Override
    public Syncer center_linear(long Y_address, 
            long X_address, 
            float alpha, float beta,
            int dim0, int dim1, int dim2, 
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;

        int N = dim1, M = dim0 * dim2;
        int nextN = Cuda_reduce.center_nextN(N, M);
        if(nextN != 1) {//V[HV: nextN, M: row_lengthv], [HV, dim0*dim2]
            block = core.malloc(nextN * M);
            V_address = block[1];
        }
        
        Cuda_reduce.center_linear(stream, 
                X_address,
                alpha, beta, 
                dim0, dim1, dim2,
                V_address, Y_address,
                width, stride, 1);
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer center_quadratic(long Y_address,
            long X_address,
            float alpha, float beta, float gamma,
            int dim0, int dim1, int dim2,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;

        int N = dim1, M = dim0 * dim2;
        int nextN = Cuda_reduce.center_nextN(N, M);
        if(nextN != 1) {//V[HV: nextN, M: row_lengthv], [HV, dim0*dim2]
            block = core.malloc(nextN * M);
            V_address = block[1];
        }
        
        Cuda_reduce.center_quadratic(stream, 
                X_address, 
                alpha, beta, gamma, 
                dim0, dim1, dim2,
                V_address, Y_address,
                width, stride, 1);
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    //<editor-fold defaultstate="collapsed" desc="center_quadratic_dual">
    @Override
    public Syncer center_quadratic2(long Y_address,
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            float C, 
            int dim0, int dim1, int dim2,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;

        int N = dim1, M = dim0 * dim2;
        int nextN = Cuda_reduce.center_nextN(N, M);
        if(nextN != 1) {//V[HV: nextN, M: row_lengthv], [HV, dim0*dim2]
            block = core.malloc(nextN * M);
            V_address = block[1];
        }
        
        Cuda_reduce.center_quadratic_dual(stream,
                X1_address, X2_address, 
                k11, k12, k22, 
                k1, k2, C, 
                dim0, dim1, dim2, 
                V_address, Y_address, 
                width, stride, 1);
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row reduce function">
    //<editor-fold defaultstate="collapsed" desc="row_linear">
    @Override
    public Syncer row_linear(long Y_address,
            long X_address,
            float alpha, float beta,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_linear(stream,
                X_address, alpha, beta,
                field_length, row_lengthv, //N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer row_linear2(long Y_address,
            long X1_address, long X2_address, 
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }

        Cuda_reduce.row_linear_dual(stream, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_quadratic">
    @Override
    public Syncer row_quadratic(long Y_address, 
            long X_address,
            float alpha, float beta, float gamma, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_quadratic(stream, 
                X_address, alpha, beta, gamma,
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer row_quadratic2(long Y_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2, float C, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_quadratic_dual(stream, 
                X1_address, X2_address, 
                k11, k12, k22, 
                k1, k2, C,
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_linear_quadratic">
    @Override
    public Syncer row_linear_quadratic(long Y1_address, long Y2_address, 
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv); V1_address = block1[1];
            block2 = core.malloc(V_lengthv); V2_address = block2[1];
        }
         
        Cuda_reduce.row_linear_quadratic(stream1, stream2, 
                X_address, 
                alpha1, beta1,//Y1 = alpha1*X + beta1
                alpha2, beta2, gamma2,//Y2 = alpha2*X^2 + beta2*X + gamma2
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V1_address, Y1_address,
                V2_address, Y2_address, 
                width, stride, 1);
        
        return (nextM == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2) :
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, block1, block2));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row_var_mean">
    private boolean row_var_safe = true;
    public boolean row_var_safe() { return row_var_safe; } 
    public CudaFloat32EngineBase row_var_safe(boolean flag) { row_var_safe = flag; return this; } 
    
    //<editor-fold defaultstate="collapsed" desc="row_var_mean_safe">
    public Syncer row_var_mean_safe(boolean unbiased,
            long var_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            block = core.malloc(nextM * SV);//V[nextM, N = field_length]
            V_address = block[1];
        }
        
        //======[staget1: compute mean = sum(X) / row_length]===================
        float alpha = 1.0f / row_length;
        Cuda_reduce.row_linear(stream, 
                X_address, alpha, 0.0f, 
                field_length, row_lengthv, //N = field_lenth, M = row_lengthv(mod stride)
                V_address, mean_address,
                width, stride, 1);
        
        //======[stage2: compute var = sum(X - mean)^2 / row_length]============
        if(unbiased) alpha = (float) (1.0 / (row_length - 1.0));
        Cuda_reduce.row_linear2_square_row(stream, 
                X_address, mean_address, 
                alpha, 1.0f, -1.0f, 0.0f,//var = alpha * (X - mean)^2
                field_length, row_lengthv,
                V_address, var_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_var_mean_fast">
    public Syncer row_var_mean_fast(boolean unbiased,
            long var_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv); V1_address = block1[1];
            block2 = core.malloc(V_lengthv); V2_address = block2[1];
        }
        
        //======[staget1: mean = E(X) and squareMean = E(X^2)]==================
        float alpha = 1.0f / row_length;
        Cuda_reduce.row_linear_quadratic(stream1, stream2,
                X_address, 
                alpha, 0.0f,              //mean   = sum_each_row: X  /  row_length
                alpha, 0.0f, 0.0f,        //sqmean = sum_each_row: X^2 / row_length
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V1_address, mean_address, //V1 = mean.buf
                V2_address, var_address,  //V2 = sqmean.buf
                width, stride, 1);
        
        //======[stage2 var = E(X^2) - E(X)^2]==================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait for stream1 and stream2
        
        float k = 1.0f;
        if(unbiased) k = (float) (row_length / (row_length - 1.0));
        Cuda_function.quadratic_dual_clip2D(stream1, 
                mean_address, var_address, //-mean^2 + squareMean
                -k, 0, 0, 0, k, 0, 
                0.0f, Float.MAX_VALUE,//var >= 0
                var_address,
                field_lengthv, field_length, field_lengthv);
        
        Cuda.deleteEvent(event);
        return (nextM == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2) :
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    
    @Override
    public Syncer row_var_mean(boolean unbiased,
            long var_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        if(row_var_safe) return row_var_mean_safe(unbiased, 
                var_address, //result0
                mean_address,//result1
                X_address, 
                field_length, field_lengthv,
                row_length, row_lengthv, 
                width, stride);
        
        return row_var_mean_fast(unbiased,
                var_address, //result0
                mean_address,//result1
                X_address, 
                field_length, field_lengthv,
                row_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_std_mean">
    private boolean row_std_safe = true;
    public boolean row_std_safe() { return row_std_safe; } 
    public CudaFloat32EngineBase row_std_safe(boolean flag) { row_std_safe = flag; return this; } 
    
    //<editor-fold defaultstate="collapsed" desc="row_std_mean_safe">
    public Syncer row_std_mean_safe(boolean unbiased,
            long std_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            block = core.malloc(nextM * SV);//V[nextM, N = field_length]
            V_address = block[1];
        }
        
        //======[stage1: mean = sum(X) / row_length]============================
        float alpha = 1.0f / row_length;
        Cuda_reduce.row_linear(stream, 
                X_address, alpha, 0.0f, 
                field_length, row_lengthv, //N = field_lenth, M = row_lengthv(mod stride)
                V_address, mean_address,
                width, stride, 1);
        
        //======[stage2: var = sum(X - mean)^2 / row_length]====================
        if(unbiased) alpha = (float) (1.0 / (row_length - 1.0));
        Cuda_reduce.row_linear2_square_row(stream, 
                X_address, mean_address, 
                alpha, 1.0f, -1.0f, 0.0f,//var = alpha * (X - mean)^2
                field_length, row_lengthv,
                V_address, std_address,
                width, stride, 1);
        
        //======[stage3: std = sqrt(var)]=======================================
        Cuda_function.sqrt2D(stream, 
                1.0f, std_address, 0.0f, 
                std_address,
                field_lengthv, field_length, field_lengthv);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_std_mean_fast">
    public Syncer row_std_mean_fast(boolean unbiased,
            long std_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv); V1_address = block1[1];
            block2 = core.malloc(V_lengthv); V2_address = block2[1];
        }
        
        //======[staget1: mean = E(X) and sqmean = E(X^2)]======================
        float alpha = 1.0f / row_length;
        Cuda_reduce.row_linear_quadratic(stream1, stream2,
                X_address, 
                alpha, 0.0f,              //mean   = sum_each_row: X   / row_length
                alpha, 0.0f, 0.0f,        //sqmean = sum_each_row: X^2 / row_length
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V1_address, mean_address, //V1 = mean.buf
                V2_address, std_address,  //V2 = sqmean.buf
                width, stride, 1);
        
        //======[stage2 std = sqrt(E(X^2) - E(X)^2)]============================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait for stream1 and stream2
       
        float k = 1.0f;
        if(unbiased) k = (float) (row_length / (row_length - 1.0));
        Cuda_function.sqrt_positive_quadratic_dual2D(stream1,
                mean_address, std_address,//std = sqrt(-mean^2 + squareMean)
                -k, 0, 0, 0, k, 0, 
                std_address,
                field_lengthv, field_length, field_lengthv);
        
        Cuda.deleteEvent(event);
        return (nextM == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2):
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    
    @Override
    public Syncer row_std_mean(boolean unbiased,
            long std_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        if(row_std_safe) return row_std_mean_safe(unbiased,
                std_address, //result0
                mean_address,//result1
                X_address, 
                field_length, field_lengthv, 
                row_length, row_lengthv, 
                width, stride);
        
        return row_std_mean_fast(unbiased,
                std_address, //result0
                mean_address,//result1
                X_address, 
                field_length, field_lengthv,
                row_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row_max, max">
    @Override
    public Syncer row_max(long Y_address, 
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_max(stream, 
                X_address, 
                field_length, row_lengthv,
                V_address, Y_address, 
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer row_min(long Y_address, 
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_min(stream, 
                X_address, 
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, Y_address, 
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer row_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L;
        long[] block1 = null, block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc_int32(V_lengthv);
            V_address     = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.row_max_indexed(stream,
                X_address, 
                field_length, row_lengthv,//N = field_lenth, M = row_lengthv(mod stride)
                V_address, VIndex_address,
                Y_address, Index_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):   
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }

    @Override
    public Syncer row_min_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L;
        long[] block1 = null, block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc_int32(V_lengthv);
            V_address     = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.row_min_indexed(stream, 
                X_address, 
                field_length, row_lengthv, //N = field_lenth, M = row_lengthv(mod stride)
                V_address, VIndex_address, 
                Y_address, Index_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):   
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Image Functions<uint8>">
    //<editor-fold defaultstate="collapsed" desc="image: pixel to dtype">
    @Override
    public Syncer linear2D_pixel_to_dtype(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.linear2D_pixel2float(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2D_dtype_to_pixel(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.linear2D_float2pixel(stream,
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_dualLinear2_div2D(long Y_address,
            long X_address,
            long X1_address, long X2_address, 
            float alpha1, float beta1, float gamma1, 
            float alpha2, float beta2, float gamma2, float C, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_dualLinear2_div2D(stream, 
                X_address, 
                X1_address, X2_address, 
                alpha1, beta1, gamma1, 
                alpha2, beta2, gamma2, C, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_dualLinear2_normalize2D_row(long Y_address, 
            long X_address,
            long X1_address, long X2_address, int row_lengthv, 
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_dualLinear2_noramlize2D_row(stream, 
                X_address, 
                X1_address, X2_address, row_lengthv,
                alpha1, beta1, gamma1, 
                alpha2, beta2, gamma2, C, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_dualLinear2_normalize2D_center(long Y_address, 
            long X_address, 
            long X1_address, long X2_address, 
            float alpha1, float beta1, float gamma1, 
            float alpha2, float beta2, float gamma2, float C,
            int dim0, int dim1, int dim2,
            int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_dualLinear2_noramlize2D_center(stream, 
                X_address,
                X1_address, X2_address, 
                alpha1, beta1, gamma1, 
                alpha2, beta2, gamma2, C, 
                Y_address,
                dim0, dim1, dim2,
                width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_linear2_div2D_row(long Y_address,
            long X_address, 
            long X1_address, long X2_address, int row_lengthv,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_linear2_div2D_row(stream, 
                X_address,
                X1_address, X2_address, row_lengthv, 
                alpha1, beta1, gamma1,
                alpha2, beta2, C, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer img_linear2_div2D_field(long Y_address,
            long X_address, 
            long X1_address, long X2_address, int row_lengthv, 
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_linear2_div2D_field(stream,
                X_address, 
                X1_address, X2_address, row_lengthv, 
                alpha1, beta1, gamma1,
                alpha2, beta2, C, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: linear, linear_dual">
    @Override
    public Syncer img_linear2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_linear2D(stream, 
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_linear_dual2D_row(long Y_address,
            long X1_address,
            long X2_address, int row_lengthv, 
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_linear_dual2D_row(stream, 
                X1_address, 
                X2_address, row_lengthv,
                alpha, beta, gamma,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_linear_dual2D_field(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv, 
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_linear_dual2D_field(stream, 
                X1_address, 
                X2_address, row_lengthv,
                alpha, beta, gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: log, exp, quadratic, threshold">
    @Override
    public Syncer img_quadratic2D(long Y_address, 
            long X_address, float alpha, float beta, float gamma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_quadratic2D(stream, 
                X_address, alpha, beta, gamma,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_threshold2D(long Y_address, 
            long X_address, float alpha, float v, byte v1, byte v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_threshold2D(stream, 
                X_address, alpha, v, v1, v2, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_log2D(long Y_address, 
            float C, float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_log2D(stream,
                C, alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer img_exp2D(long Y_address,
            float alpha, long X_address, float beta, float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_exp2D(stream,
                alpha, X_address, beta, C, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }  
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: pad, trim, transpose">
    @Override
    public Syncer img_pad(
            long Y_address, int OH, int OW, int OC, 
            long X_address, int IH, int IW, int IC,
            int N, int ph0, int pw0, int pc0) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_pad(stream, 
                Y_address, OH, OW, OC,
                X_address, IH, IW, IC,
                N, ph0, pw0, pc0);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer img_trim(
            long Y_address, int OH, int OW, int OC, 
            long X_address, int IH, int IW, int IC, 
            int N, int ph0, int pw0, int pc0) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_trim(stream, 
                Y_address, OH, OW, OC, 
                X_address, IH, IW, IC, 
                N, ph0, pw0, pc0);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_transpose(
            long Y_address, int[] Ydim, 
            long X_address, int[] Xdim, 
            int dimIndex1, int dimIndex2, 
            int strideX, int strideY, int length) 
    {
        long stream = streamPool.getStream();
          if(Xdim.length == 4) {
            Cuda_image.img_transpose4D(stream, 
                    X_address, Y_address,
                    Xdim[1], Xdim[2], Xdim[3], 
                    Ydim[1], Ydim[2], Ydim[3], 
                    dimIndex1, dimIndex2,
                    strideX, strideY, length);
        }
        else if(Xdim.length == 3) {
            Cuda_image.img_transpose3D(stream,
                    X_address, Y_address, 
                    Xdim[1], Xdim[2], 
                    Ydim[1], Ydim[2], 
                    dimIndex1, dimIndex2,
                    strideX, strideY, length);
        }
        else {//X.dim = 2
            Cuda_image.img_transpose2D(stream,
                    X_address, Y_address,
                    Xdim[1], Ydim[1],
                    strideX, strideY,
                    length);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: resize, affine">
    @Override
    public Syncer img_resize(
            long X_address, int IH, int IW, 
            long Y_address, int OH, int OW, 
            int N, int C) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_resize(stream, 
                X_address, IH, IW, 
                Y_address, OH, OW, 
                N, C);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer img_affine(
            long X_address, int IH, int IW, 
            long Y_address, int OH, int OW, 
            float r00, float r01, float r02,
            float r10, float r11, float r12, 
            int N, int C)
    {
        long stream = streamPool.getStream();
        Cuda_image.img_affine(stream, 
                X_address, IH, IW, 
                Y_address, OH, OW,
                r00, r01, r02, 
                r10, r11, r12, 
                N, C);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: gappedMemcpy, extrac_3channels">
    @Override
    public Syncer img_gappedMemcpy2D(
            long X_address, int Xstart, int strideX, 
            long Y_address, int Ystart, int strideY, 
            int width, int length) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_gappedMemcpy2D(stream,
                X_address, Xstart, strideX, 
                Y_address, Ystart, strideY,
                width, length);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer extract_3channels(
            long X_address, int IC, 
            long Y_address, int c0, int c1, int c2, 
            int lengthv) 
    {
        long stream = streamPool.getStream();
        Cuda_image.img_extract_3channels(stream, 
                X_address, IC, 
                Y_address, c0, c1, c2, 
                lengthv);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
}
