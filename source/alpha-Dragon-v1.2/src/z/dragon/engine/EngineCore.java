/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.Objects;
import z.dragon.common.MemStatus;
import z.dragon.engine.memp.Mempool;
import z.dragon.engine.Result.IndexedResult;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 * 4D[N, IH, IW, IC]
 * dim: 0, 1, 2, 3.
 * @author Gilgamesh
 */
public class EngineCore implements MemStatus {
    public static final long MEM_1GB = (1L) << 30;
    public static final long MEM_1MB = (1L) << 20;
    public static final long NULL = 0L;
    
    public static final long L_sizeof_int32 = 2;
    public static final long L_sizeof_int8 = 0;
    public static final long sizeof_int32 = 4;
    public static final long sizeof_int8 = 1;
    
    //<editor-fold defaultstate="collapsed" desc="member params">
    protected EngineBase base;
    protected long L_sizeof_datatype;//sizeof_datatype = 1<<L_sizeof_datatype
    protected Mempool mempool;
    
    protected boolean check = true;
    
    protected ExRandom exr = new ExRandom();
    //</editor-fold>
    
    protected EngineCore() {}
    public EngineCore(Mempool memp, boolean check) {
        if(memp == null) throw new NullPointerException("Mempool is null");
        this.mempool = memp;
        this.check = check;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean check() { return check; }
    public void check(boolean flag) { this.check = flag; }
    
    public ExRandom random() { return exr; }
    public EngineCore random(ExRandom random) { 
        if(random == null) throw new NullPointerException("random is null");
        exr = random; 
        return this; 
    }
    
    public EngineBase engineBase() { return base; }
    public synchronized EngineCore engineBase(EngineBase base) {
        if(base == null) throw new NullPointerException("EngineBase is null");
        this.base = base; base.core = this;
        this.L_sizeof_datatype = base.L_sizeof_datatype;
        this.mempool.engineBase(base);
        return this;
    }
    
    public String dataType()        { return base.datatype; }
    public String dataType_int32() { return base.datatype_int32; }
    public String dataType_int8()  { return base.datatype_int8; }
    
    public long LsizeOf_dataType() { return L_sizeof_datatype; }
    public long sizeOf_dataType()  { return 1 << L_sizeof_datatype; }
    
    public Mempool mempool() { return this.mempool; }
    public synchronized EngineCore mempool(Mempool memp) {
        if(memp == null) throw new NullPointerException("Mempool is null");
        this.mempool = memp;
        if(base != null) this.mempool.engineBase(base);
        return this;
    }
    
    @Override public long max_mem_size() { return mempool.max_mem_size(); }
    @Override public long total_mem_size() { return mempool.total_mem_size(); }
    @Override public long used_mem_size() { return mempool.used_mem_size(); }

    public EngineCore max_mem_size(long maxMemSize) { mempool.max_mem_size(maxMemSize); return this; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append("{ ");
        sb.append("\ncheck = ").append(check);
        if(mempool != null) mempool.meta_data().forEach((String key, Object value)-> {
             sb.append("\n\t mempool.").append(key).append(" = ").append(value);
        });
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    
    public void clear() {
        mempool.clear();
    }
    
    @Override
    public void finalize() throws Throwable  {
        super.finalize();
        this.clear();
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 31 * hash + Objects.hashCode(this.base);
        hash = 31 * hash + Objects.hashCode(this.mempool);
        return hash;
    }
    
    @Override
    public boolean equals(Object o) {
        if(!(o instanceof EngineCore)) return false;
        EngineCore core = (EngineCore) o;
        return Objects.equals(core.engineBase(), base);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="extra: int32"> 
    //<editor-fold defaultstate="collapsed" desc="memcpy_int32">
    public Syncer memcpy_int32(long dst_address, int dst_pos, long src_address, int src_pos, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        dst_address += dst_pos << L_sizeof_int32;
        src_address += src_pos << L_sizeof_int32;
        return base.memcpy(dst_address, src_address, length << L_sizeof_int32);
    }
    
    public Syncer memcpy_int32(long dst_address, long src_address, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        return base.memcpy(dst_address, src_address, length << L_sizeof_int32);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="memset_int32"> 
    public Syncer memset_int32(long address, int value, int pos, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        address += pos << L_sizeof_int32;
        return base.memset(address, value, length << L_sizeof_int32);//length * 4
    }
    
    public Syncer memset_int32(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        return base.memset(address, value, length << L_sizeof_int32);//length * 4
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="get tensor value int32"> 
    public int[] get1D_int32(long address, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        int[] value = new int[length];
        base.get1D_int32(address, value, length);
        return value;
    }
    
    public int[] get2D_int32(long address, int height, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height < 0) throw new IllegalArgumentException(String.format(
                    "height { got %d } must be positive", height));
            if(width < 0) throw new IllegalArgumentException(String.format(
                    "width { got %d } must be positive", width));
        }
        int stride = ((width + 3) >> 2) << 2;
        int[] value = new int[height * width];
        base.get2D_int32(address, value, height, width, stride);
        return value;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set tensor value int32"> 
    public void set1D_int32(long address, int[] value) {
        if(address == NULL) throw new NullPointerException("Tensor.address is null");
        base.set1D_int32(address, value, value.length);
    }
    
    public void set2D_int32(long address, int[] value, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor is null");
            if(value.length < width) throw new IllegalArgumentException(String.format(
                    "value.length { got %d } < width { got %d }", 
                    value.length, width));
            if(width <= 0) throw new IllegalArgumentException(String.format(
                    "width { got %d } must positive", width));
            if(value.length % width !=0) throw new IllegalArgumentException(String.format(
                    "value.length { got %d } %% width { got %d } != 0",
                    value.length, width));
        }
        int height = value.length / width;
        int stride = ((width + 3) >> 2) << 2;
        base.set2D_int32(address, value, height, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="extra: int8">
    //<editor-fold defaultstate="collapsed" desc="memcpy_int8">
    public Syncer memcpy_int8(long dst_address, int dst_pos, long src_address, int src_pos, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        dst_address += dst_pos;
        src_address += src_pos;
        return base.memcpy(dst_address, src_address, length);
    }
    
    public Syncer memcpy_int8(long dst_address, long src_address, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        return base.memcpy(dst_address, src_address, length);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="memset_int8">
    public Syncer memset_int8(long address, int value, int pos, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                     "length { got %d } must be positive", length));
        }
        address += pos;
        return base.memset(address, value, length);//length * 1
    }
    
    public Syncer memset_int8(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must be positive", length));
        }
        return base.memset(address, value, length);//length * 1
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = fill(constant<int8>)">
    public Syncer set1D_int8(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must > 0", length));
        }
        return base.set1D_int8(address, value, length);
    }

    public Syncer set2D_int8(long address, int value, int height, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor is null");
            if(height <= 0) throw new IllegalArgumentException(String.format(
                    "height { got %d } must > 0", height));
            if(width <= 0) throw new IllegalArgumentException(String.format(
                    "height { got %d } must > 0", width));
        }
        int stride = ((width + 3) >> 2) << 2;
        return base.set2D_int8(address, value, height, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="get tensor value int8">
    public byte[] get1D_int8(long address, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length < 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } must > 0", length));
        }
        byte[] value = new byte[length];
        base.get1D_int8(address, value, length);
        return value;
    }
    
    public byte[] get2D_int8(long address, int height, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height < 0) throw new IllegalArgumentException(String.format(
                    "height { got %d } must > 0", height));
            if(width < 0) throw new IllegalArgumentException(String.format(
                    "width { got %d } must > 0", width));
        }
        int stride = ((width + 3) >> 2) << 2;
        byte[] value = new byte[height * width];
        base.get2D_int8(address, value, height, width, stride);
        return value;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set tensor value int8">
    public void set1D_int8(long address, byte[] value) {
        if(address == NULL) throw new NullPointerException("Tensor is null");
        base.set1D_int8(address, value, value.length);
    }
    
    public void set2D_int8(long address, byte[] value, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor is null");
            if(value.length < width) throw new IllegalArgumentException(String.format(
                    "value.length { got %d } < width { got %d }",  
                    value.length, width));
            if(width <= 0) throw new IllegalArgumentException(String.format(
                    "width { got %d } must > 0", width));
            if(value.length % width !=0) throw new IllegalArgumentException(String.format(
                    "value.length { got %d } %% width { got %d } != 0",
                    value.length, width));
        }
   
        int height = value.length / width;
        int stride = ((width + 3) >> 2) << 2;
        base.set2D_int8(address, value, height, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Pooling: create & delete"> 
    //malloc: return { mem_size, mem_address }
    public long[] malloc(int mem_length) { return mempool.malloc(check, mem_length, L_sizeof_datatype); }    
    public long[] malloc_int32(int mem_length) { return mempool.malloc(check, mem_length, 2); }//log2(32 / 8) = 2
    public long[] malloc_int8(int mem_length)  { return mempool.malloc(check, mem_length, 0); }//log2(8 / 8) = 0
   
    //padding to (1 << L_sizeof_datatype), As sizeof_datatype may not a power of 2
    public long[] malloc_dtype(int mem_length, long sizeof_dtype) {
        long mem_size = (mem_length * sizeof_dtype) +  + (1 << L_sizeof_datatype) - 1;
        mem_length = (int) (mem_size >> L_sizeof_datatype);
        return mempool.malloc(check, mem_length, L_sizeof_datatype);
    }
    
    public boolean free(long mem_size, long mem_address) {
        return mempool.free(check, mem_size, mem_address);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Operations">
    //<editor-fold defaultstate="collapsed" desc="memset">   
    public Syncer memset(long address, int value, int pos, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length < 0) throw new IllegalArgumentException("length must be non negative");
        }
        address += pos << L_sizeof_datatype;
        return base.memset(address, value, length << L_sizeof_datatype);
    }
    
    public Syncer memset(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length < 0) throw new IllegalArgumentException("length must be non negative");
        }
        return base.memset(address, value, length << L_sizeof_datatype);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="memcpy">
    public Syncer memcpy(long dst_address, int dst_pos, long src_address, int src_pos, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length < 0) throw new IllegalArgumentException("length must be positive");
        }
        dst_address += dst_pos << L_sizeof_datatype;
        src_address += src_pos << L_sizeof_datatype;
        return base.memcpy(dst_address, src_address, length << L_sizeof_datatype);
    }
    
    public Syncer memcpy(long dst_address, long src_address, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length < 0) throw new IllegalArgumentException("length must be positive");
        }
        return base.memcpy(dst_address, src_address, length << L_sizeof_datatype);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="tensor = fill(constant)">
    public Syncer set1D(long address, float value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must be postive");
        }
        return base.set1D(address, value, length);
    }

    public Syncer set2D(long address, float value, int height, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height <= 0) throw new IllegalArgumentException("height must positive");
            if(width <= 0) throw new IllegalArgumentException("height must positive");
        }
        int stride = ((width + 3) >> 2) << 2;
        return base.set2D(address, value, height, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="get tensor value">
    public float[] get1D(long address, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must be positive");
        }
        float[] value = new float[length];
        base.get1D(address, value, length);
        return value;
    }
    
    public float[] get2D(long address, int height, int width) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height <= 0) throw new IllegalArgumentException("mem_height <= 0 ");
            if(width <= 0) throw new IllegalArgumentException("mem_width <= 0");
        }
        int stride = ((width + 3) >> 2) << 2;
        float[] value = new float[height * width];
        base.get2D(address, value, height, width, stride);
        return value;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = value<float[]>">
    public void set1D(long address, float[] value) {
        if(address == NULL) throw new NullPointerException("Tensor.address is null");
        base.set1D(address, value, value.length);
    }
    
    public void set2D(long address, float[] value, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(value.length < width) throw new IllegalArgumentException();
            if(width <= 0) throw new IllegalArgumentException("mem_width <= 0");
            if(value.length % width !=0) throw new IllegalArgumentException();
        }
        
        int height = value.length / width;
        int stride = ((width + 3) >> 2) << 2;
        base.set2D(address, value, height, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = another tensor">
    public Syncer setFrom1Dto2D(long src_address, int src_length,
            long dst_address, int dst_height, int dst_width)
    {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(src_length <= 0) throw new IllegalArgumentException("src.length <= 0");
            if(dst_width <= 0) throw new IllegalArgumentException("dst.mem_width <= 0");
            if(dst_height <= 0) throw new IllegalArgumentException("dst.mem_height <= 0");
            if(src_length != dst_height * dst_width) throw new IllegalArgumentException("src.length != dst.length");
        }
        
        int dst_stride = ((dst_width + 3) >> 2) << 2;
        return base.setFrom1Dto2D(
                src_address, src_length, dst_address, 
                dst_height, dst_width, dst_stride);
    }
    
    public Syncer setFrom2Dto1D(long src_address, int src_height, int src_width,
            long dst_address, int dst_length)
    {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(dst_length <= 0) throw new IllegalArgumentException("dst.length <= 0");
            if(src_width <= 0) throw new IllegalArgumentException("src.mem_width <= 0");
            if(src_height <= 0) throw new IllegalArgumentException("src.mem_height <= 0");
            if(src_height * src_width != dst_length) throw new IllegalArgumentException("src.length != dst.length");
        }
        
        int src_stride = ((src_width + 3) >> 2) << 2;
        return base.setFrom2Dto1D(
                src_address, src_height, src_width, src_stride, 
                dst_address, dst_length);
    }
    
    public Syncer setFrom2Dto2D(long src_address, int src_height, int src_width,
            long dst_address, int dst_height, int dst_width)
    {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(src_width <= 0) throw new IllegalArgumentException("src.mem_width <= 0");
            if(src_height <= 0) throw new IllegalArgumentException("src.mem_height <= 0");
            if(dst_width <= 0) throw new IllegalArgumentException("dst.mem_width <= 0");
            if(dst_height <= 0) throw new IllegalArgumentException("dst.mem_height <= 0");
            if(src_height * src_width != dst_height * dst_width) throw new IllegalArgumentException("src.length != dst.length");
        }
        
        int src_stride = ((src_width + 3) >> 2) << 2;
        int dst_stride = ((dst_width + 3) >> 2) << 2;
        return base.setFrom2Dto2D(
                src_address, src_height, src_width, src_stride, 
                dst_address, dst_height, dst_width, dst_stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    //<editor-fold defaultstate="collapsed" desc="gappedMemcpy2D">
    public Syncer gappedMemcpy2D(
            long X_address, int Xstart, int strideX, 
            long Y_address, int Ystart, int strideY,
            int width, int length)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Xstart < 0) throw new IllegalArgumentException(String.format("Xstart { got %d } < 0", Xstart));
            if(Ystart < 0) throw new IllegalArgumentException(String.format("Ystart { got %d } < 0", Ystart));
            if(strideX < width) throw new IllegalArgumentException(String.format(
                    "strideX { got %d } < width { got %d }", strideX, width));
            if(strideY < width) throw new IllegalArgumentException(String.format(
                    "strideY { got %d } < width { got %d }", strideX, width));
            if(length < width) throw new IllegalArgumentException(String.format(
                    "length { got %d } < width { got %d }", length, width));
            if(length % width != 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } %% width { got %d } != 0", length, width));
            if(width < 0) throw new IllegalArgumentException(String.format(
                    "width { got %d } < 0", width));
        }
        return base.gappedMemcpy2D(
                X_address, Xstart, strideX,
                Y_address, Ystart, strideY, 
                width, length);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="indexedMemcpy">
    public Syncer srcIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index is null");
            func_param_check(lengthv, width, stride);
        }
        return base.srcIndexedMemcpy(Y_address, X_address, Index_address, 
                lengthv, width, stride);
    }
    
    public Syncer dstIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index is null");
            func_param_check(lengthv, width, stride);
        }
        return base.dstIndexedMemcpy(Y_address, X_address, Index_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="transpose(2D -> 4D)">
    public Syncer transpose(
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim, 
            int dimIndex1, int dimIndex2, 
            int widthX, int widthY, 
            int length) 
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Ydim.length < 2) throw new IllegalArgumentException(String.format(
                    "Y.ndim { got %d } must >= 2", Ydim.length));
            if(Xdim.length < 2) throw new IllegalArgumentException(String.format(
                    "X.ndim { got % d} must >= 2", Xdim.length));
            if(widthX <= 0) throw new IllegalArgumentException(String.format("widthX { got %d } <= 0", widthX));
            if(length < widthX) throw new IllegalArgumentException(String.format(
                    "length { got %d } < widthX { got %d }", length, widthX));
            if(length % widthX != 0) throw new IllegalArgumentException(String.format(
                    "lengthv { got %d } %% widthX { got %d } != 0", length, widthX));
            if(widthY <= 0) throw new IllegalArgumentException(String.format("widthY { got %d } <= 0", widthY));
            if(length < widthY) throw new IllegalArgumentException(String.format(
                    "length { got %d } < widthY { got %d }", length, widthY));
            if(length % widthY != 0) throw new IllegalArgumentException(String.format(
                    "lengthv { got %d } %% widthY { got %d } != 0", length, widthY));
        }
        return base.transpose(
                Y_address, Ydim,
                X_address, Xdim, 
                dimIndex1, dimIndex2, 
                (widthX + 3) >> 2 << 2,
                (widthY + 3) >> 2 << 2,
                length);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="rot180">
    public Syncer rot180(long Y_address,
            long X_address,
            int N, int IH, int IW, int IC)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(N <= 0) throw new IllegalArgumentException(String.format("N { got %d } must > 0", N));
            if(IH <= 0) throw new IllegalArgumentException(String.format("IH { got %d } must > 0", IH));
            if(IW <= 0) throw new IllegalArgumentException(String.format("IW { got %d } must > 0", IW));
            if(IC <= 0) throw new IllegalArgumentException(String.format("IC { got %d } must > 0", IC));
        }
        IC = ((IC + 3) >> 2) << 2;
        int length = N * IH * IW * IC;
        return base.rot180(Y_address, X_address, IH, IW, IC, length);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pad(2D -> 4D)">
    public Syncer pad(//X.ndim = Y.ndim = p0.length
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim,
            int[] p0)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Ydim.length != p0.length) throw new IllegalArgumentException(String.format(
                    "Y.ndim { got %d } != p0.length { got %d }", Ydim.length, p0.length));
            if(Xdim.length != p0.length) throw new IllegalArgumentException(String.format(
                    "X.ndim { got %d } != p0.length { got %d }", Xdim.length, p0.length));
            for(int i=0; i<Ydim.length; i++) if(Xdim[i] + p0[i] > Ydim[i])
                throw new IllegalArgumentException(String.format(
                        "X.dim[%d] { got %d } + p0[%d] { got %d } > Y.dim[%d] { got %d }",
                        i, Xdim[i], i, p0[i], i, Ydim[i]));
        }
        
        int n = Ydim.length - 1;//width -> stride
        Ydim = Vector.arrayCopy(Ydim); Ydim[n] = (Ydim[n] + 3) >> 2 << 2;
        Xdim = Vector.arrayCopy(Xdim); Xdim[n] = (Xdim[n] + 3) >> 2 << 2;
        return base.pad(
                Y_address, Ydim, 
                X_address, Xdim, 
                p0);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="trim(2D -> 4D)">
    public Syncer trim(//X.ndim = Y.ndim = p0.length
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim,
            int[] p0)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Ydim.length != p0.length) throw new IllegalArgumentException(String.format(
                    "Y.ndim { got %d } != p0.length { got %d }", Ydim.length, p0.length));
            if(Xdim.length != p0.length) throw new IllegalArgumentException(String.format(
                    "X.ndim { got %d } != p0.length { got %d }", Xdim.length, p0.length));
            for(int i=0; i<Ydim.length; i++) if(Xdim[i] - p0[i] < Ydim[i])
                throw new IllegalArgumentException(String.format(
                        "X[%d].dim { got %d } - p0[%d] { got %d } < Y.dim[%d] { got %d }",
                        i, Xdim[i], i, p0[i], i, Ydim[i]));
        }
        
        int n = Ydim.length - 1;//width -> stride
        Ydim = Vector.arrayCopy(Ydim); Ydim[n] = (Ydim[n] + 3) >> 2 << 2;
        Xdim = Vector.arrayCopy(Xdim); Xdim[n] = (Xdim[n] + 3) >> 2 << 2;
        return base.trim(
                Y_address, Ydim, 
                X_address, Xdim, 
                p0);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected void matMul_check(long C_address, long A_address, long B_address,
            int N, int M, int K)
    {
        if(C_address == NULL) throw new NullPointerException("Tensor C is null");
        if(B_address == NULL) throw new NullPointerException("Tensor B is null");
        if(A_address == NULL) throw new NullPointerException("Tensor A is null");
        
        if(N <= 0) throw new IllegalArgumentException(String.format("N { got %d } must > 0", N));
        if(M <= 0) throw new IllegalArgumentException(String.format("M { got %d } must > 0", M));
        if(K <= 0) throw new IllegalArgumentException(String.format("K { got %d } must > 0", K));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Normal matMul">
    public Syncer matMul(long C_address,
            long A_address, long B_address,
            int N, int M, int K) 
    {
        if(check) matMul_check(C_address, A_address, B_address, N, M, K);
        return base.matMul(C_address, A_address, B_address, 
                ((N + 3) >> 2) << 2, 
                ((M + 3) >> 2) << 2, 
                ((K + 3) >> 2) << 2);
    }
    
    public Syncer matMul_biased(long C_address, 
            long A_address, long B_address,
            int N, int M, int K,
            long Bias_address,//lengthv = C.lengthv = N*M, stride = C.mem_stride = M
            int lengthv)
    {
        int M_4x = ((M + 3) >> 2) << 2;
        if(check){
            matMul_check(C_address, A_address, B_address, N, M, K);
            //lengthv = Y.lengthv = N * M_4x
            //row_lengthv = M_4x
            //[width, stride] = M, M_4x
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");
            func_param_check_row(lengthv, M_4x, M, M_4x);
        }
        return base.matMul_biased(C_address, A_address, B_address,
                ((N + 3) >> 2) << 2, 
                M_4x, //((M + 3) >> 2) << 2
                ((K + 3) >> 2) << 2,
                Bias_address,
                lengthv, M);
    }
    
    //transpose B
    public Syncer matMulT1(long C_address,
            long A_address, long B_address,
            int N, int M, int K)
    {
        if(check) matMul_check(C_address, A_address, B_address, N, M, K);
        return base.matMulT1(C_address, A_address, B_address,
                ((N + 3) >> 2) << 2, 
                ((M + 3) >> 2) << 2, 
                ((K + 3) >> 2) << 2);
    }
    
    //transpose A
    public Syncer matMulT2(long C_address, long 
            A_address, long B_address,
            int N, int M, int K)
    {
        if(check) matMul_check(C_address, A_address, B_address, N, M, K);
        return base.matMulT2(C_address, A_address, B_address,
                ((N + 3) >> 2) << 2, 
                ((M + 3) >> 2) << 2, 
                ((K + 3) >> 2) << 2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Batch matMul">
    public Syncer batchMatMul(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int K) 
    {
        if(check) {
            if(Batch <= 0) throw new IllegalArgumentException("Batch MatMul: Batch must be positive");
            matMul_check(C_address, A_address, B_address, N, M, K);
        }
        //A[Batch, N, AK] * B[Batch, BK, M] = C[Batch, N, M]
        return base.batchMatMul(C_address, A_address, B_address,
                ((Batch + 3) >> 2) << 2,//Batch % 4 == 0
                N, ((M + 3) >> 2) << 2, //N % 4 != 0, M % 4 == 0
                K, ((K + 3) >> 2) << 2);//K = BK % 4 != 0, AK % 4 == 0
    }
    
    public Syncer batchMatMulT1(long C_address, 
            long A_address, long B_address,//A.transpose(1, 2)
            int Batch, int N, int M, int K)
    {
        if(check) {
            if(Batch <= 0) throw new IllegalArgumentException("Batch MatMul: Batch must be positive");
            matMul_check(C_address, A_address, B_address, N, M, K);
        }
        //(A[Batch, K, AN])^T * B[Batch, K, M] = C[Batch, CN, M]
        return base.batchMatMulT1(C_address, A_address, B_address,
                ((Batch + 3) >> 2) << 2,//Batch % 4 == 0
                N, ((N + 3) >> 2) << 2, //N = CN % 4 != 0, AN % 4 == 0
                ((M + 3) >> 2) << 2, K);//memAligned, M % 4 == 9, K % 4 ! = 0
    }
    
    public Syncer batchMatMulT2(long C_address, 
            long A_address, long B_address,//B.transpose(1, 2)
            int Batch, int N, int M, int K)
    {
        if(check) {
            if(Batch <= 0) throw new IllegalArgumentException("Batch MatMul: Batch must be positive");
            matMul_check(C_address, A_address, B_address, N, M, K);
        }
        //A[Batch, N, K] * (B[Batch, BM, K])^T = C[Batch, N, CM]
        return base.batchMatMulT2(C_address, A_address, B_address,
                ((Batch + 3) >> 2) << 2, N, //Batch % 4 == 0, N % 4 != 0
                ((M + 3) >> 2) << 2, M,//M = CM % 4 == 0, BM % 4 != 0
                ((K + 3) >> 2) << 2); //K % 4 == 0
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convlution 3D (NHWC)">
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected void conv3D_param_check(
            int OH, int OW, int IH, int IW, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw)
    {
        if(OH <= 0) throw new IllegalArgumentException(String.format("OH { got %d } must > 0", OH));
        if(OW <= 0) throw new IllegalArgumentException(String.format("OW { got %d } must > 0", OW));
        if(N  <= 0) throw new IllegalArgumentException(String.format("N  { got %d } must > 0", N));
        if(IC <= 0) throw new IllegalArgumentException(String.format("IC { got %d } must > 0", IC));
        if(OC <= 0) throw new IllegalArgumentException(String.format("OC { got %d } must > 0", OC));
        if(sh <= 0) throw new IllegalArgumentException(String.format("sh { got %d } must > 0", sh));
        if(sw <= 0) throw new IllegalArgumentException(String.format("sw { got %d } must > 0", sw));
        if(ph < 0) throw new IllegalArgumentException(String.format("ph { got %d } must >= 0", ph));
        if(pw < 0) throw new IllegalArgumentException(String.format("pw { got %d } must >= 0", pw));
        
        if(FH <= ph) throw new IllegalArgumentException(String.format("FH { got %d } <= ph { got %d }", FH, ph));
        if(FW <= pw) throw new IllegalArgumentException(String.format("FW { got %d } <= pw { got %d }", FW, pw));
        if(FH < sh) throw new IllegalArgumentException(String.format("FH { got %d } < sh { got %d }", FH, sh));
        if(FW < sw) throw new IllegalArgumentException(String.format("FW { got %d } < sw { got %d }", FW, sw));
        
        if(FH > IH + (ph << 1)) throw new IllegalArgumentException(String.format(
                "FH { got %d } > IH { got %d } + 2 * ph { %d }", FH, IH, ph));
        if(FW > IW + (pw << 1)) throw new IllegalArgumentException(String.format(
                "FW { got %d } > IW { got %d } + 2 * pw { %d }", FW, IW, pw));
        
        if(IH - FH + (ph << 1) < (OH - 1)*sh) throw new IllegalArgumentException(String.format(
                "IH { got %d } - FH { got %d } + 2 * ph { got %d } >= OH { got %d } - 1) * sh { got %d }",
                IH, FH, ph, OH, sh));
        if(IW - FW + (pw << 1) < (OW - 1)*sw) throw new IllegalArgumentException(String.format(
                "IW { got %d } - FW { got %d } + 2 * pw { got %d } >= OW { got %d } - 1) * sw { got %d }", 
                IW, FW, pw, OW, sw));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW,
            long W_address, int FH, int FW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IH, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OH, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FH, FW, IC]
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        }
        return base.conv3D(
                Y_address, OH, OW, 
                X_address, IH, IW,
                W_address, FH, FW, 
                ((N  + 3) >> 2) << 2,
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
   
    public Syncer conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW,         
            int N, int IC, int OC,//row_lengthv = OC_4x, [width, stride] = OC, OC_4x
            int sh, int sw, int ph, int pw,
            long Bias_address, int lengthv)//lengthv = Y.lengthv = N * OH * OW * OC_4x
    {
        int OC_4x = ((OC + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IH, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OH, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FH, FW, IC]
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");//Bias[OC]
            func_param_check_row(lengthv, OC_4x, OC, OC_4x);
        }
        return base.conv3D_biased(
                Y_address, OH, OW,
                X_address, IH, IW,
                W_address, FH, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                OC_4x, //((OC + 3) >> 2) << 2
                sh, sw, ph, pw, 
                Bias_address,
                lengthv, OC);
    }
    
    public Syncer deconv3D_biased(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,//row_lengthv = OC_4x, [width, stride] = OC, OC_4x
            int sh, int sw, int ph ,int pw, 
            long Bias_address, int lengthv)//lengthv = Y.lengthv = N * OH * OW * OC_4x
    {
        int OC_4x = ((OC + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            conv3D_param_check(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");//Bias[OC]
            func_param_check_row(lengthv, OC_4x, OC, OC_4x);
        }
        return base.deconv3D_biased(
                Y_address, OH, OW, 
                X_address, IH, IW, 
                W_address, FH, FW, 
                ((N  + 3) >> 2) << 2,  
                ((IC + 3) >> 2) << 2,
                OC_4x, //((OC + 3) >> 2) << 2
                sh, sw, ph, pw, 
                Bias_address, 
                lengthv, OC);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    public Syncer conv3D_deltaW(
            long deltaW_address, int FH, int FW, 
            long X_address,      int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        }
        return base.conv3D_deltaW(
                deltaW_address, FH, FW, 
                X_address,      IH, IW, 
                deltaY_address, OH, OW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
    
    public Syncer conv3D_deltaX(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            long W_address,      int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph ,int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        }
        return base.conv3D_deltaX(
                deltaX_address, IH, IW, 
                deltaY_address, OH, OW, 
                W_address,      FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Convlution 2D (NWC)">
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected void conv2D_param_check(
            int OW, int IW, int FW, 
            int N, int IC, int OC, 
            int sw, int pw)
    {
        if(OW <= 0) throw new IllegalArgumentException(String.format("OW { got %d } must > 0", OW));
        if(N  <= 0) throw new IllegalArgumentException(String.format("N  { got %d } must > 0", N));
        if(IC <= 0) throw new IllegalArgumentException(String.format("IC { got %d } must > 0", IC));
        if(OC <= 0) throw new IllegalArgumentException(String.format("OC { got %d } must > 0", OC));
        if(sw <= 0) throw new IllegalArgumentException(String.format("sw { got %d } must > 0", sw));
        if(pw < 0) throw new IllegalArgumentException(String.format("pw { got %d } must >= 0", pw));
        
        if(FW <= pw) throw new IllegalArgumentException(String.format("FW { got %d } <= pw { got %d }", FW, pw));
        if(FW < sw) throw new IllegalArgumentException(String.format("FW { got %d } < sw { got %d }", FW, sw));
        
        if(FW > IW + (pw << 1)) throw new IllegalArgumentException(String.format(
                "FW { got %d } > IW { got %d } + 2 * pw { %d }", FW, IW, pw));
        if(IW - FW + (pw << 1) < (OW - 1)*sw) throw new IllegalArgumentException(String.format(
                "IW { got %d } - FW { got %d } + 2 * pw { got %d } >= OW { got %d } - 1) * sw { got %d }", 
                IW, FW, pw, OW, sw));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer conv2D(
            long Y_address, int OW, 
            long X_address, int IW,
            long W_address, int FW,
            int N, int IC, int OC, 
            int sw, int pw)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FW, IC]
            conv2D_param_check(OW, IW, FW, N, IC, OC, sw, pw);
        }
        return base.conv3D(
                Y_address, 1, OW, 
                X_address, 1, IW,
                W_address, 1, FW, 
                ((N  + 3) >> 2) << 2,
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                1, sw, 0, pw);
    }
   
    public Syncer conv2D_biased(
            long Y_address, int OW, 
            long X_address, int IW, 
            long W_address, int FW,         
            int N, int IC, int OC,//row_lengthv = OC_4x, [width, stride] = OC, OC_4x
            int sw, int pw,
            long Bias_address, int lengthv)//lengthv = Y.lengthv = N * OH * OW * OC_4x
    {
        int OC_4x = ((OC + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FW, IC]
            conv2D_param_check(OW, IW, FW, N, IC, OC, sw, pw);
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");//Bias[OC]
            func_param_check_row(lengthv, OC_4x, OC, OC_4x);
        }
        return base.conv3D_biased(
                Y_address, 1, OW,
                X_address, 1, IW,
                W_address, 1, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                OC_4x,//((OC + 3) >> 2) << 2
                1, sw, 0, pw, 
                Bias_address,
                lengthv, OC);
    }
    
    public Syncer deconv2D_biased(
            long Y_address, int OW,
            long X_address, int IW,
            long W_address, int FW,
            int N, int IC, int OC,//row_lengthv = OC_4x, [width, stride] = OC, OC_4x
            int sw, int pw, 
            long Bias_address, int lengthv)//lengthv = Y.lengthv = N * OH * OW * OC_4x
    {
        int OC_4x = ((OC + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            conv2D_param_check(IW, OW, FW, N, IC, OC, sw, pw);
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");//Bias[OC]
            func_param_check_row(lengthv, OC_4x, OC, OC_4x);
        }
        return base.deconv3D_biased(
                Y_address, 1, OW, 
                X_address, 1, IW, 
                W_address, 1, FW, 
                ((N  + 3) >> 2) << 2,  
                ((IC + 3) >> 2) << 2,
                OC_4x,//((OC + 3) >> 2) << 2
                1, sw, 0, pw, 
                Bias_address, 
                lengthv, OC);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    public Syncer conv2D_deltaW(
            long deltaW_address, int FW, 
            long X_address,      int IW,
            long deltaY_address, int OW,
            int N, int IC, int OC,
            int sw, int pw)
    {
        if(check) {
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            conv2D_param_check(OW, IW, FW, N, IC, OC, sw, pw);
        }
        return base.conv3D_deltaW(
                deltaW_address, 1, FW, 
                X_address,      1, IW, 
                deltaY_address, 1, OW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                1, sw, 0, pw);
    }
    
    public Syncer conv2D_deltaX(
            long deltaX_address, int IW,
            long deltaY_address, int OW,
            long W_address,      int FW,
            int N, int IC, int OC,
            int sw, int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            conv2D_param_check(OW, IW, FW, N, IC, OC, sw, pw);
        }
        return base.conv3D_deltaX(
                deltaX_address, 1, IW, 
                deltaY_address, 1, OW, 
                W_address,      1, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                1, sw, 0, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="DepthWiseConvlution 3D (NHWC)">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer depthwise_conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW,
            long W_address, int FH, int FW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IH, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OH, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[FH, FW, OC: M * IC]
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            if(OC % IC != 0) throw new IllegalArgumentException(String.format("OC { got %d } %% IC { got %d } ! = 0", OC, IC));
        }
        
        return base.depthwise_conv3D(Y_address, OH, OW, 
                X_address, IH, IW,
                W_address, FH, FW,
                ((N  + 3) >> 2) << 2,
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer depthwise_conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW,         
            int N, int IC, int OC,//row_lengthv = OC_4x, [width, stride] = OC, OC_4x
            int sh, int sw, int ph, int pw,
            long Bias_address, int lengthv)//lengthv = Y.lengthv = N * OH * OW * OC_4x
    {
        int OC_4x = ((OC + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IH, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OH, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FH, FW, IC]
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            if(OC % IC != 0) throw new IllegalArgumentException(String.format("OC { got %d } %% IC { got %d } ! = 0", OC, IC));
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");//Bias[OC]
            func_param_check_row(lengthv, OC_4x, OC, OC_4x);
        }
        return base.depthwise_conv3D_biased(Y_address, OH, OW,
                X_address, IH, IW,
                W_address, FH, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                OC_4x, //((OC + 3) >> 2) << 2
                sh, sw, ph, pw, 
                Bias_address,
                lengthv, OC);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    public Syncer depthwise_conv3D_deltaW(
            long deltaW_address, int FH, int FW, 
            long X_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            if(OC % IC != 0) throw new IllegalArgumentException(String.format("OC { got %d } %% IC { got %d } ! = 0", OC, IC));
        }
        return base.depthwise_conv3D_deltaW(
                deltaW_address, FH, FW, 
                X_address, IH, IW, 
                deltaY_address, OH, OW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
    
    public Syncer depthwise_conv3D_deltaX(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph ,int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            conv3D_param_check(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            if(OC % IC != 0) throw new IllegalArgumentException(String.format("OC { got %d } %% IC { got %d } ! = 0", OC, IC));
        }
        return base.depthwise_conv3D_deltaX(
                deltaX_address, IH, IW, 
                deltaY_address, OH, OW, 
                W_address, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Pooling 2D (NHWC)"> 
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected void pool2D_param_check(
            int OH, int OW, int IH, int IW, int FH, int FW, 
            int N, int IC, int sh, int sw, int ph, int pw)
    {
        if(OH <= 0) throw new IllegalArgumentException(String.format("OH { got %d } must > 0", OH));
        if(OW <= 0) throw new IllegalArgumentException(String.format("OW { got %d } must > 0", OW));
        if(N <= 0) throw new IllegalArgumentException(String.format("N (batch_size) must > 0", N));
        if(IC <= 0) throw new IllegalArgumentException(String.format("IC (input_channel) { got %d } must > 0", IC));
        if(sh <= 0) throw new IllegalArgumentException(String.format("sh (stride_height) { got %d } must > 0", sh));
        if(sw <= 0) throw new IllegalArgumentException(String.format("sw (srtide_width) { got %d } must > 0", sw));
        if(ph < 0) throw new IllegalArgumentException(String.format("ph (padding_height) { got %d } must > 0", ph));
        if(pw < 0) throw new IllegalArgumentException(String.format("pw (padding_width) { got %d }  must > 0", pw));
        
        if(FH * FW < 2) throw new IllegalArgumentException(String.format(
                "FH { got %d }* FW { got %d } < 2", FH, FW));
        if(FH <= ph) throw new IllegalArgumentException(String.format(
                "FH { got %d } <= ph { got %d }", FH, ph));
        if(FW <= pw) throw new IllegalArgumentException(String.format(
                "FW { got %d } <= pw { got %d }", FW, pw));
        if(FH < sh) throw new IllegalArgumentException(String.format(
                "FH { got %d } < sh { got %d }", FH, sh));
        if(FW < sw) throw new IllegalArgumentException(String.format(
                "FW { got %d } < sw { got %d }", FW, sw));
        
        if(FH > IH + (ph << 1)) throw new IllegalArgumentException(String.format(
                "FH { got %d } > IH { got %d } + 2 * ph { %d }", FH, IH, ph));
        if(FW > IW + (pw << 1)) throw new IllegalArgumentException(String.format(
                "FW { got %d } > IW { got %d } + 2 * pw { %d }", FW, IW , pw));
        
        if(IH - FH + (ph << 1) < (OH - 1)*sh) throw new IllegalArgumentException(String.format(
                "IH { got %d } - FH { got %d } + 2 * ph { got %d } >= (OH { got %d } - 1) * sh { got %d }",
                IH, FH, ph, OH, sh));
        if(IW - FW + (pw << 1) < (OW - 1)*sw) throw new IllegalArgumentException(String.format(
                "IW { got %d } - FW { got %d } + 2 * pw { got %d } >= (OW { got %d } - 1) * sw { got %d }", 
                IW, FW, pw, OW, sw));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    //<editor-fold defaultstate="collapsed" desc="pool2D_max">
    public Syncer pool2D_max(//ndim = 3
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, 1, IC, sh, sw, ph, pw);
        }
        return base.pool2D_max(
                Y_address, OH, OW, 
                X_address, 
                ((IH + 3) >> 2) << 2, 
                IW, FH, FW, 1,//N = 1
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer pool2D_max(//ndim = 4
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        return base.pool2D_max(
                Y_address, OH, OW, 
                X_address, IH, IW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="pool2D_max_indexed">
    public Syncer pool2D_max_indexed(
            long Y_address, long Index_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Index is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        return base.pool2D_max_indexed(
                Y_address, Index_address, OH, OW, 
                X_address, IH, IW, FH, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="pool2D_avg">
    public Syncer pool2D_avg(boolean ignore_padding,//ndim = 3
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, 1, IC, sh, sw, ph, pw);
        }
        return base.pool2D_avg(ignore_padding,
                Y_address, OH, OW, 
                X_address, 
                ((IH + 3) >> 2) << 2, 
                IW, FH, FW, 1,//N = 1
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer pool2D_avg(boolean ignore_padding,//ndim = 4
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        return base.pool2D_avg(ignore_padding,
                Y_address, OH, OW, 
                X_address, IH, IW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    public Syncer unpool2D_max(
            long deltaY_address, long Y_address, int OH, int OW, 
            long deltaX_address, long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        return base.unpool2D_max(
                deltaX_address, X_address, IH, IW, 
                deltaY_address, Y_address, OH, OW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer unpool2D_max_Indexed(
            long deltaX_address, int IH, int IW,
            long deltaY_address, long Index_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor<int32> Index is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        return base.unpool2D_max_Indexed(
                deltaX_address, IH, IW, 
                deltaY_address, Index_address, OH, OW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer unpool2D_avg(boolean ignore_padding,
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            pool2D_param_check(OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        return base.unpool2D_avg(ignore_padding,
                deltaX_address, IH, IW, 
                deltaY_address, OH, OW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Pooling 1D (NWC)"> 
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected void pool1D_param_check(
            int OW, int IW, int FW, 
            int N, int IC, int sw, int pw)
    {
        if(OW <= 0) throw new IllegalArgumentException(String.format("OW { got %d } must > 0", OW));
        if(N <= 0) throw new IllegalArgumentException(String.format("N (batch_size) must > 0", N));
        if(IC <= 0) throw new IllegalArgumentException(String.format("IC (input_channel) { got %d } must > 0", IC));
        if(sw <= 0) throw new IllegalArgumentException(String.format("sw (srtide_width) { got %d } must > 0", sw));
        if(pw < 0) throw new IllegalArgumentException(String.format("pw (padding_width) { got %d }  must > 0", pw));
        
        if(FW < 2) throw new IllegalArgumentException(String.format("FW { got %d } < 2", FW));
        if(FW <= pw) throw new IllegalArgumentException(String.format("FW { got %d } <= pw { got %d }", FW, pw));
        if(FW < sw) throw new IllegalArgumentException(String.format("FW { got %d } < sw { got %d }", FW, sw));
        
        if(FW > IW + (pw << 1)) throw new IllegalArgumentException(String.format(
                "FW { got %d } > IW { got %d } + 2 * pw { %d }", FW, IW , pw));
        if(IW - FW + (pw << 1) < (OW - 1)*sw) throw new IllegalArgumentException(String.format(
                "IW { got %d } - FW { got %d } + 2 * pw { got %d } >= (OW { got %d } - 1) * sw { got %d }", 
                IW, FW, pw, OW, sw));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    //<editor-fold defaultstate="collapsed" desc="pool1D_max">
    public Syncer pool1D_max(//ndim = 3
            long Y_address, int OW,
            long X_address, int IW,
            int FW, int N, int IC,
            int sw, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool1D_param_check(OW, IW, FW, N, IC, sw, pw);
        }
        return base.pool2D_max(
                Y_address, 1, OW, 
                X_address, 1, IW, 1, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                1, sw, 0, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="pool1D_max_indexed">
    public Syncer pool1D_max_indexed(
            long Y_address, long Index_address, int OW,
            long X_address, int IW,
            int FW, int N, int IC,
            int sw, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Index is null");
            pool1D_param_check(OW, IW, FW, N, IC, sw,pw);
        }
        return base.pool2D_max_indexed(
                Y_address, Index_address, 1, OW, 
                X_address, 1, IW, 1, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                1, sw, 0, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="pool1D_avg">
    public Syncer pool1D_avg(boolean ignore_padding,//ndim = 4
            long Y_address, int OW,
            long X_address, int IW,
            int FW, int N, int IC,
            int sw, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool1D_param_check(OW, IW, FW, N, IC, sw, pw);
        }
        return base.pool2D_avg(ignore_padding,
                Y_address, 1, OW, 
                X_address, 1, IW, 1, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                1, sw, 0, pw);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    public Syncer unpool1D_max(
            long deltaY_address, long Y_address, int OW, 
            long deltaX_address, long X_address, int IW,
            int FW, int N, int IC,
            int sw, int pw)
    {
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            pool1D_param_check(OW, IW, FW, N, IC, sw, pw);
        }
        
        return base.unpool2D_max(
                deltaX_address, X_address, 1, IW, 
                deltaY_address, Y_address, 1, OW, 1, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                1, sw, 0, pw);
    }
    
    public Syncer unpool1D_max_Indexed(
            long deltaX_address, int IW,
            long deltaY_address, long Index_address, int OW, 
            int FW, int N, int IC,
            int sw, int pw)
    {
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor<int32> Index is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            pool1D_param_check(OW, IW, FW, N, IC, sw, pw);
        }
        
        return base.unpool2D_max_Indexed(
                deltaX_address, 1, IW, 
                deltaY_address, Index_address, 1, OW, 1, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                1, sw, 0, pw);
    }
    
    public Syncer unpool1D_avg(boolean ignore_padding,
            long deltaX_address, int IW,
            long deltaY_address, int OW, 
            int FW, int N, int IC,
            int sw, int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            pool1D_param_check(OW, IW, FW, N, IC, sw, pw);
        }
        
        return base.unpool2D_avg(ignore_padding,
                deltaX_address, 1, IW, 
                deltaY_address, 1, OW, 1, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                1, sw, 0, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected void func_param_check(int lengthv, int width, int stride) {
        if(lengthv < width) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } < width { got %d }", lengthv, width));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } <= 0", width));
        if(lengthv % stride != 0) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } %% stride { got %d } != 0", lengthv, width));
    }
    
    protected void func_param_check_row(int lengthv, int row_lengthv, int width, int stride) {
        if(lengthv < row_lengthv) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } < row_lengthv { %d }", lengthv, row_lengthv));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } <= 0", width));
        if(lengthv % row_lengthv != 0) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } %% row_lengthv { %d } != 0", lengthv, row_lengthv));
        if(row_lengthv % stride != 0) throw new IllegalArgumentException(String.format(
                "row_lengthv { %d } %% stride { %d } != 0", row_lengthv, stride));
    }
    
    protected void func_param_check_center(int dim0, int dim1, int dim2, int width) {
        if(dim0 < 0) throw new RuntimeException(String.format("dim0 { got %d } must > 0", dim0));
        if(dim1 < 0) throw new RuntimeException(String.format("dim1 { got %d } must > 0", dim1));
        if(dim2 < width) throw new RuntimeException(String.format("dim2 { got %d } < width { got %d }", dim2, width));
        if(dim2 % width != 0) throw new RuntimeException(String.format("dim2 { got %d } %% width { got %d } != 0", dim2, width));
        if(width < 0) throw new RuntimeException(String.format("width { got %d } must > 0", width));
    }
    
    protected void func_param_check_field(int lengthv, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        if(lengthv < row_lengthv) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } < row_lengthv { %d }", lengthv, row_lengthv));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width{ %d } <= 0", width));
        if(lengthv % field_length != 0) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } %% field_length { %d } != 0", lengthv, field_length));
        if(row_lengthv % stride != 0) throw new IllegalArgumentException(String.format(
                "row_lengthv { got %d } %% stride { got %d } != 0", row_lengthv, stride));
    }
    
    final void func_param_check_softmax(int lengthv, int row_length, int row_lengthv, int width) {
        if(lengthv < row_length) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } < row_length { got %d }", lengthv, row_length));
        if(row_length < width) throw new IllegalArgumentException(String.format(
                "row_length { got %d } < width { got %d }", row_length, width));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } <= 0", width));
        if(row_length % width != 0) throw new IllegalArgumentException(String.format(
                "row_length { got %d } %% width { got %d } != 0", row_length, width));
        if(lengthv % row_lengthv != 0) throw new IllegalArgumentException(String.format(
                "lengthv { got %d } %% row_lengthv { got %d } != 0", lengthv, row_lengthv));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="greater, equal, linear, rpl, div, quadratic"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs">
    public Syncer equal_abs2D(long Y_address, 
            long X1_address, long X2_address,
            float min, float max,
            int lengthv, int width)
    {
        if(min > max) { float t = min; min = max; max = t; }
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(min < 0) throw new IllegalArgumentException(String.format("min { got %f } must >= 0", min));
            if(max < 0) throw new IllegalArgumentException(String.format("max { got %f } must >= 0", max));
            func_param_check(lengthv, width, stride);
        }
        return base.equal_abs2D(Y_address,
                X1_address, X2_address,
                min, max, 
                lengthv, width, stride);
    }
    
    public Syncer equal_abs2D_int8(long Y_address, 
            long X1_address, long X2_address,
            byte min, byte max,
            int lengthv, int width)
    {
        if(min > max) { byte t = min; min = max; max = t; }
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(min < 0) throw new IllegalArgumentException(String.format("min { got %d } must >= 0", min));
            if(max < 0) throw new IllegalArgumentException(String.format("max { got %d } must >= 0", max));
            func_param_check(lengthv, width, stride);
        }
        return base.equal_abs2D_int8(Y_address,
                X1_address, X2_address, 
                min, max, 
                lengthv, width, stride);
    }
    
    public Syncer equal_abs2D_int32(long Y_address, 
            long X1_address, long X2_address,
            int min, int max,
            int lengthv, int width)
    {
        if(min > max) { int t = min; min = max; max = t; }
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(min < 0) throw new IllegalArgumentException(String.format("min { got %d } must >= 0", min));
            if(max < 0) throw new IllegalArgumentException(String.format("max { got %d } must >= 0", max));
            func_param_check(lengthv, width, stride);
        }
        return base.equal_abs2D_int32(Y_address,
                X1_address, X2_address,
                min, max, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater">   
    public Syncer linear_greater2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_greater2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }

    public Syncer linear_greater2_2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_greater2_2D(Y_address, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater_switch">   
    public Syncer linear_greater_switch2D(long Y_address,
            float alpha, long X_address, float beta,
            float v1, float v2,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_greater_switch2D(Y_address, 
                alpha, X_address, beta,
                v1, v2,
                lengthv, width, stride);
    }
    
    public Syncer linear_greater_switch_mul2D(long Y_address,
            float alpha, long X1_address, float beta,
            long X2_address, float v1, float v2,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_greater_switch_mul2D(Y_address, 
                alpha, X1_address, beta, 
                X2_address, v1, v2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_bound_switch">   
    public Syncer linear_bound_switch_mul2D(long Y_address,
            float alpha, long X1_address, float vmin, float vmax,
            long X2_address, float v1, float v2, float v3,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        if(vmin > vmax) { float t = vmin; vmin = vmax; vmax = t; }  
        return base.linear_bound_switch_mul2D(Y_address, 
                alpha, X1_address, vmin, vmax,
                X2_address, v1, v2, v3,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear">
    public Syncer linear2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer linear_2out2D(long Y1_address, long Y2_address,
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y1_address == NULL) throw new NullPointerException("Tensor Y1 is null");
            if(Y2_address == NULL) throw new NullPointerException("Tensor Y2 is null");
            if(X_address  == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_2out2D(Y1_address, Y2_address,
                X_address, 
                alpha1, beta1, 
                alpha2, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int8 to dtype">
    public Syncer linear2D_int8_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_int8_to_dtype(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
     
    public Syncer linear2D_dtype_to_int8(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_dtype_to_int8(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int32 to dtype">
    public Syncer linear2D_int32_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_int32_to_dtype(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
     
    public Syncer linear2D_dtype_to_int32(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_dtype_to_int32(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="linear2">
    public Syncer linear2_2D(long Y_address,
            long X1_address, 
            long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_2D(Y_address, 
                X1_address,
                X2_address,
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear_summary2D(long Y_address,
            float alpha, float beta, long[] Xs,//X2.length >= 2
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor W is null");
            Vector.requireNonNull(Xs, "Tensor Xs");
            if(Xs.length < 2) throw new IllegalArgumentException("At least 2 Tensors to find their summary");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_summary2D(Y_address, 
                Xs, alpha, beta,
                lengthv, width, stride);
    }
    
    public Syncer linear2_iteration2D(long Y_address,
            long X1_address, long[] X2,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor W is null");  
            Vector.requireNonNull(X2, "Tensor X2");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_iteration2D(Y_address, 
                X1_address, X2, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_row">
    public Syncer linear2_2D_row(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.linear2_2D_row(Y_address, 
                X1_address,
                X2_address, row_lengthv, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_center">
    public Syncer linear2_2D_center(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int dim0, int dim1, int dim2,
            int width)
    {
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_center(dim0, dim1, dim2, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        dim2 = (dim2 / width) * stride;
        return base.linear2_2D_center(Y_address, 
                X1_address, X2_address,
                alpha, beta, gamma,
                dim0, dim1, dim2,
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_field">
    public Syncer linear2_2D_field(long Y_address,
            long X1_address, 
            long X2_address, int field_length,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_field(lengthv,  field_length, row_lengthv, width, stride); 
        }
        return base.linear2_2D_field(Y_address,
                X1_address,
                X2_address, row_lengthv,
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="mul_linear_dual2D">
    public Syncer mul_linear2_2D(long Y_address,
            long X_address, long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.mul_linear2_2D(Y_address, 
                X_address, X1_address, X2_address, 
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic">
    public Syncer quadratic2D(long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2D(Y_address,
                X_address, alpha, beta, gamma, 
                lengthv, width, stride);
    }

    public Syncer quadratic2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2D_deltaX(deltaX_address,
                deltaY_address,
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2">
    public Syncer quadratic2_2D(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2_2D(Y_address, 
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, C, 
                lengthv, width, stride);
    }

    public Syncer quadratic2_2D_deltaX(
            long deltaX1_address,//result0
            long deltaX2_address,//result1
            long deltaY_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(deltaY_address  == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2_2D_deltaX(deltaX1_address, deltaX2_address,
                deltaY_address, 
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, 
                lengthv, width, stride);
    }
    
    public Syncer quadratic_summary2D(long Y_address,
            float alpha, float beta, float gamma, long[] Xs,//X2.length >= 2
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor W is null");
            Vector.requireNonNull(Xs, "Tensor Xs");
            if(Xs.length < 2) throw new IllegalArgumentException("At least 2 Tensors to find their summary");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic_summary2D(Y_address, 
                Xs, alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer quadratic2_iteration2D(long Y_address,
            long X1_addresss, long[] X2,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor W is null");  
            Vector.requireNonNull(X2, "Tensor X2");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2_iteration2D(Y_address,
                X1_addresss, X2,
                k11, k12, k22,
                k1, k2, C,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_row">
    public Syncer quadratic2_2D_row(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.quadratic2_2D_row(Y_address, 
                X1_address,
                X2_address, row_lengthv,
                k11, k12, k22, 
                k1, k2, C, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2_center">
    public Syncer quadratic2_2D_center(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int dim0, int dim1, int dim2,
            int width)
    {
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_center(dim0, dim1, dim2, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        dim2 = (dim2 / width) * stride;
        return base.quadratic2_2D_center(Y_address,
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C,
                dim0, dim1, dim2,
                width, stride);
    }
    
    public Syncer quadratic2_2D_center_deltaX(
            long deltaX1_address,//result0
            long deltaX2_address,//result1
            long deltaY_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int dim0, int dim1, int dim2,
            int width)
    {
        if(check) {
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_center(dim0, dim1, dim2, width);
        } 
        int stride = ((width + 3) >> 2) << 2;
        dim2 = (dim2 / width) * stride;
        return base.quadratic2_2D_center_deltaX(
                deltaX1_address,//result0
                deltaX2_address,//result1
                deltaY_address,
                X1_address, X2_address, 
                k11, k12, k22,
                k1, k2, C,
                dim0, dim1, dim2,
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_field">
    public Syncer quadratic2_2D_field(long Y_address,
            long X1_address, 
            long X2_address, int field_length,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.quadratic2_2D_field(Y_address,
                X1_address, 
                X2_address, row_lengthv,
                k11, k12, k22,
                k1, k2, C,
                lengthv, width, stride);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: rpl">
    public Syncer rpl2D(long Y_address,
            float alpha, long X_address, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.rpl2D(Y_address, 
                alpha, X_address, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer rpl2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float alpha, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check(lengthv, width, stride);
        }
        return base.rpl2D_deltaX(deltaX_address,
                deltaY_address, 
                Y_address, alpha, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div">
    public Syncer div2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(String.format(
                    "(alpha2 { got %f } * X2 + beta2 { got %f } ) identically equals to zero", alpha2, beta2));
            func_param_check(lengthv, width, stride);
        }
        return base.div2D(Y_address,
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2, 
                gamma,
                lengthv, width, stride);
    }
    
    public Syncer div2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, float alpha1, float beta1,
            long X2_address, float alpha2, float beta2, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(deltaY_address  == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(String.format(
                    "(alpha2 { got %f } * X2 + beta2 { got %f } ) identically equals to zero", alpha2, beta2));
            func_param_check(lengthv, width, stride);
        }
        return base.div2D_deltaX(deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, alpha1, beta1,
                X2_address, alpha2, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div2D: row, field">
    public Syncer div2D_row(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(String.format(
                    "(alpha2 { got %f } * X2 + beta2 { got %f } ) identically equals to zero", alpha2, beta2));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.div2D_row(Y_address, 
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2,
                gamma, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer div2D_field(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(String.format(
                    "(alpha2 { got %f } * X2 + beta2 { got %f } ) identically equals to zero", alpha2, beta2));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.div2D_field(Y_address,
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2, 
                gamma, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="add_div: row, field">
    public Syncer linear2_div2D_row(long Y_address,
            long X1_address, 
            long X2_address,
            long X3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(X3_address == NULL) throw new NullPointerException("Tensor X3 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.linear2_div2D_row(Y_address, 
                X1_address,
                X2_address, X3_address, row_lengthv, 
                alpha, beta, gamma, delta, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_div2D_field(long Y_address,
            long X1_address,
            long X2_address,
            long X3_address, int field_length,
            float alpha, float beta, float gamma, float delta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(X3_address == NULL) throw new NullPointerException("Tensor X3 is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.linear2_div2D_field(Y_address, 
                X1_address, 
                X2_address, X3_address, row_lengthv, 
                alpha, beta, gamma, delta, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="mul_squareDiv2D">
    public Syncer mul_squareDiv2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float alpha3, long X3_address, float beta3, 
            float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha3 == 0 && beta3 == 0) throw new IllegalArgumentException(
                    "(alpha3*X3 + beta3) identically equals to zero");
            func_param_check(lengthv, width, stride);
        }
        return base.mul_squareDiv2D(Y_address, 
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                alpha3, X3_address, beta3, 
                gamma, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, sqrt"> 
    public Syncer sign2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sign2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer ceil2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.ceil2D(Y_address,
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
   
    public Syncer floor2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.floor2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: abs">
    public Syncer abs2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.abs2D(Y_address, 
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer abs2D_deltaX(long deltaX_address,
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.abs2D_deltaX(deltaX_address, 
                deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    public Syncer zero_nan2D(long Y_address, 
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.zero_nan2D(Y_address,
                X_address,
                lengthv, width, stride);
    }
    
    public Syncer sqrt2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sqrt2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
   public Syncer sqrt_quadratic2_2D(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
         if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sqrt_quadratic2_2D(Y_address, 
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="minValue, max, clip">
    //<editor-fold defaultstate="collapsed" desc="minValue, min2">
    public Syncer min2D(long Y_address,
            float alpha, long X_address, float beta, 
            float vmin,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.min2D(Y_address, 
                alpha, X_address, beta, vmin, 
                lengthv, width, stride);
    }
    
    public Syncer min2_2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.min2_2D(Y_address,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="max, max2">
    public Syncer max2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.max2D(Y_address, 
                alpha, X_address, beta, vmax,
                lengthv, width, stride);
    }
    
    public Syncer max2_2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.max2_2D(Y_address,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="clip">
    public Syncer clip2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmin, float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        if(vmin > vmax) { float t = vmin; vmin = vmax; vmax = t; }  
        return base.clip2D(Y_address, 
                alpha, X_address, beta, vmin, vmax,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    //<editor-fold defaultstate="collapsed" desc="exp">
    public Syncer exp2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.exp2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: log">
    public Syncer log2D(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.log2D(Y_address, 
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer log2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.log2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address, alpha, 
                lengthv, width, stride);
    }
    //</editor-fold>          
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    public Syncer relu2D(long Y_address,
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.relu2D(Y_address,
                X_address, 
                lengthv, width, stride);
    }
    
    public Syncer relu2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.relu2D_deltaX_v1(deltaX_address, 
                deltaY_address, Y_address, 
                lengthv, width, stride);
    }
    
     public Syncer relu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.relu2D_deltaX_v2(deltaX_address, 
                deltaY_address, X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    public Syncer leakyRelu2D(long Y_address,
            long X_address, float k, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu2D(Y_address,
                X_address, k, 
                lengthv, width, stride);
    }
    
    public Syncer leakyRelu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, k, 
                lengthv, width, stride);
    }
    
    public Syncer leakyRelu2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu2D_deltaX_v2(deltaX_address, 
                deltaY_address,
                X_address, k,
                lengthv, width, stride);
    }        
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    public Syncer elu2D(long Y_address,
            long X_address, float alpha, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if(k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check(lengthv, width, stride);
        }
        return base.elu2D(Y_address, 
                X_address, alpha, k, 
                lengthv, width, stride);
    }
    
    public Syncer elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float k,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if(k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check(lengthv, width, stride);
        }
        return base.elu2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, alpha, k, 
                lengthv, width, stride);
    }
    
    public Syncer elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float k,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if(k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check(lengthv, width, stride);
        }
        return base.elu2D_deltaX_v2(deltaX_address,
                deltaY_address, 
                X_address, alpha, k, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    public Syncer softPlus2D(long Y_address,
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.softPlus2D(Y_address, X_address, 
                lengthv, width, stride);
    }
    
    public Syncer softPlus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.softPlus2D_deltaX_v1(deltaX_address, 
                deltaY_address, 
                Y_address, 
                lengthv, width, stride);
    }
    
    public Syncer softPlus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.softPlus2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: gelu">
    public Syncer gelu2D(long Y_address,
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.gelu2D(Y_address,
                X_address, 
                lengthv, width, stride);
    }
    
    public Syncer gelu2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.gelu2D_deltaX(deltaX_address, 
                deltaY_address, Y_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_leakyRelu2D (relu)">
    public Syncer linear2_relu2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_relu2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_leakyRelu2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_leakyRelu2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, k, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_leakyRelu2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_leakyRelu2D_deltaX_v1(
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                Y_address, 
                alpha, beta, k, 
                lengthv, width, stride);
    }
     
    public Syncer linear2_leakyRelu2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_leakyRelu2D_deltaX_v2(
                deltaX1_address, deltaX2_address, 
                deltaY_address,
                X1_address, X2_address, 
                alpha, beta, gamma, k, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_elu2D">
    public Syncer linear2_elu2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            float theta, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if (Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if (X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if (X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if (theta < 0) throw new IllegalArgumentException("Elu: alpha must >= 0");
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_elu2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                theta, k, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_elu2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta, 
            float theta, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if (deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if (deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (theta < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_elu2D_deltaX_v1(
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                Y_address, 
                alpha, beta, 
                theta, k, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_elu2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma, 
            float theta, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if (deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if (deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if (X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if (X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if (theta < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope { got %f } must be non-negative", k));
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_elu2D_deltaX_v2(
                deltaX1_address, deltaX2_address, 
                deltaY_address,
                X1_address, X2_address, 
                alpha, beta, gamma,
                theta, k, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_softplus2D">
    public Syncer linear2_softplus2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_softplus2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_softplus2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_softplus2D_deltaX_v1(
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                Y_address, 
                alpha, beta,
                lengthv, width, stride);
    }
     
    public Syncer linear2_softplus2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_softplus2D_deltaX_v2(
                deltaX1_address, deltaX2_address, 
                deltaY_address,
                X1_address, X2_address, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_gelu2D">
    public Syncer linear2_gelu2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_gelu2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_gelu2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_gelu2D_deltaX_v2(
                deltaX1_address, deltaX2_address, 
                deltaY_address,
                X1_address, X2_address, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    public Syncer sigmoid2D(long Y_address,
            long X_address,
           int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid2D(Y_address, 
                X_address, 
                lengthv, width, stride);
    }
    
    public Syncer sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid2D_deltaX_v1(deltaX_address,
                deltaY_address, 
                Y_address,
                lengthv, width, stride);
    }
    
    public Syncer sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid2D_deltaX_v2(deltaX_address, 
                deltaY_address,
                X_address,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    public Syncer tanh2D(long Y_address,
            long X_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tanh2D(Y_address, X_address,
                lengthv, width, stride);
    }
    
    public Syncer tanh2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tanh2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, 
                lengthv, width, stride);
    }
    
    public Syncer tanh2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tanh2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    public Syncer softmax2D(long Y_address, 
            long X_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax2D(Y_address,
                X_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer softmax2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: logsoftmax">  
    public Syncer logsoftmax2D(long Y_address, 
            long X_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.logsoftmax2D(Y_address,
                X_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer logsoftmax2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.logsoftmax2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_sigmoid2D">
    public Syncer linear2_sigmoid2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_sigmoid2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_sigmoid2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_sigmoid2D_deltaX_v1(
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                Y_address, 
                alpha, beta,
                lengthv, width, stride);
    }
     
    public Syncer linear2_sigmoid2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_sigmoid2D_deltaX_v2(
                deltaX1_address, deltaX2_address, 
                deltaY_address,
                X1_address, X2_address, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_tanh2D">
    public Syncer linear2_tanh2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_tanh2D(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear2_tanh2D_deltaX_v1(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_tanh2D_deltaX_v1(
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                Y_address, 
                alpha, beta,
                lengthv, width, stride);
    }
     
    public Syncer linear2_tanh2D_deltaX_v2(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,//V2: holdX(), {X1, X2} are not changed
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2_tanh2D_deltaX_v2(
                deltaX1_address, deltaX2_address, 
                deltaY_address,
                X1_address, X2_address, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin">
    public Syncer sin2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sin2D(Y_address,
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer sin2D_deltaX(long deltaX_address, 
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sin2D_deltaX(deltaX_address,
                deltaY_address,
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tan">
    public Syncer tan2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tan2D(Y_address, 
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer tan2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tan2D_deltaX(deltaX_address, 
                deltaY_address, 
                Y_address, alpha, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: csc">
    public Syncer csc2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.csc2D(Y_address,
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer csc2D_deltaX(long deltaX_address, 
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.csc2D_deltaX(deltaX_address,
                deltaY_address,
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin2D">
    public Syncer arcsin2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arcsin2D(Y_address,
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer arcsin2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arcsin2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address, alpha, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arctan2D">        
    public Syncer arctan2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arctan2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer arctan2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arctan2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address, alpha,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin">
    public Syncer halfSin2D(long Y_address,
            float Amp, float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.halfSin2D(Y_address, 
                Amp, alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer halfSin2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float Amp, float alpha,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.halfSin2D_deltaX(deltaX_address, 
                deltaY_address, 
                Y_address, Amp, alpha,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    public Syncer L1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L1_2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer L1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y  is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L1_2D_deltaYh(deltaYh_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    public Syncer L2_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L2_2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer L2_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L2_2D_deltaYh(deltaYh_address,
                Y_address, Yh_address,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: SmoothL1">
    public Syncer smoothL1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.smoothL1_2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer smoothL1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.smoothL1_2D_deltaYh(deltaYh_address,
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    public Syncer binaryCrossEntropy2D(long L_address,
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.binaryCrossEntropy2D(L_address, 
                Y_address, Yh_address, 
                alpha, beta,
                lengthv, width, stride);
    }
     
    public Syncer binaryCrossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.binaryCrossEntropy2D_deltaYh(deltaYh_address,
                Y_address, Yh_address,
                alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    public Syncer sigmoid_binaryCrossEntropy2D(long L_address,
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid_binaryCrossEntropy2D(L_address,
                Y_address, X_address, 
                alpha, beta,
                lengthv, width, stride);
    }
    
    public Syncer sigmoid_binaryCrossEntropy2D_deltaX(long deltaX_address,
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X(Yh = sigmoid(X)) is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid_binaryCrossEntropy_deltaX(deltaX_address,
                Y_address, X_address, 
                alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    public Syncer crossEntropy2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.crossEntropy2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer crossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.crossEntropy2D_deltaYh(deltaYh_address,
                Y_address, Yh_address,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax_crossEntropy">
    public Syncer softmax_crossEntropy2D(long L_address,
            long Y_address, long X_address, int row_length, //lengthv = field_length * row_lengthv
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;

        if(check) {
            if(L_address == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax_crossEntropy2D(L_address,
                Y_address, X_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer softmax_crossEntropy2D_deltaX(long deltaX_address,
            long Y_address, long X_address, int row_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X(Yh = softmax(X)) is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax_crossEntropy2D_deltaX(deltaX_address,
                Y_address, X_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="BP: affine">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine2D(long Y_address,
            long X_address,
            long A_address, long B_address, int row_lengthv, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine2D(Y_address, 
                X_address, 
                A_address, B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public Syncer affine2D_deltaA_v1(long deltaA_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine2D_deltaA_v1(deltaA_address, 
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer affine2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//tesult1
                deltaY_address,
                Y_address,
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer affine2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long X_address,//(V2: X for Affine || V1: Y for Norm)
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, 
                X_address,
                field_length, row_lengthv, 
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_leakyRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine_relu2D(long Y_address,
            long X_address,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_relu2D(Y_address,
                X_address, 
                A_address, B_address, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer affine_leakyRelu2D(long Y_address,
            long X_address,
            long A_address, long B_address, 
            int row_lengthv, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_leakyRelu2D(Y_address,
                X_address, 
                A_address, B_address, 
                row_lengthv, k, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Syncer affine_leakyRelu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(k <= 0) throw new IllegalArgumentException(String.format("k { got %f } must > 0", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_leakyRelu2D_deltaX_v1(deltaX_address,
                deltaY_address, k,
                Y_address,
                A_address, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer affine_leakyRelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float k,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_leakyRelu2D_deltaX_v2(deltaX_address,
                deltaY_address, k, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    public Syncer affine_leakyRelu2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(k <= 0) throw new IllegalArgumentException(String.format("k { got %f } must > 0", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_leakyRelu2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, k,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer affine_leakyRelu2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, float k,
            long X_address,//V2: holdX(), X is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_leakyRelu2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, k,
                X_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_elu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine_elu2D(long Y_address,
            long X_address,
            long A_address, long B_address, int row_lengthv,
            float alpha, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_elu2D(Y_address,
                X_address, 
                A_address, B_address, row_lengthv,
                alpha, k,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Syncer affine_elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_elu2D_deltaX_v1(deltaX_address,
                deltaY_address, alpha, k,
                Y_address,
                A_address, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer affine_elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_elu2D_deltaX_v2(deltaX_address,
                deltaY_address, alpha, k,
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    public Syncer affine_elu2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_elu2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, alpha, k,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer affine_elu2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_elu2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, alpha, k,
                X_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine_softplus2D(long Y_address,
            long X_address,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_softplus2D(Y_address,
                X_address, 
                A_address, B_address, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Syncer affine_softplus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_softplus2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address,
                A_address, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer affine_softplus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_softplus2D_deltaX_v2(deltaX_address,
                deltaY_address, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    public Syncer affine_softplus2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_softplus2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer affine_softplus2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_softplus2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                X_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine_gelu2D(long Y_address,
            long X_address,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_gelu2D(Y_address,
                X_address, 
                A_address, B_address, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Syncer affine_gelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_gelu2D_deltaX_v2(deltaX_address,
                deltaY_address, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    public Syncer affine_gelu2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_gelu2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                X_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine_sigmoid2D(long Y_address,
            long X_address,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_sigmoid2D(Y_address,
                X_address, 
                A_address, B_address, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Syncer affine_sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_sigmoid2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address,
                A_address, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer affine_sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_sigmoid2D_deltaX_v2(deltaX_address,
                deltaY_address, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    public Syncer affine_sigmoid2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_sigmoid2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer affine_sigmoid2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_sigmoid2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                X_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine_tanh2D(long Y_address,
            long X_address,
            long A_address, long B_address, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_tanh2D(Y_address,
                X_address, 
                A_address, B_address, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Syncer affine_tanh2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_tanh2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address,
                A_address, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer affine_tanh2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, 
            long B_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.affine_tanh2D_deltaX_v2(deltaX_address,
                deltaY_address, 
                X_address,
                A_address,
                B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    public Syncer affine_tanh2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_tanh2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer affine_tanh2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine_tanh2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address,
                X_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer sqBatchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                row_lengthv, lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, 
                row_lengthv, lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_sqmean_address, eps, 
                A_address,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>  
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public Syncer sqBatchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long X_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaA_v2(deltaA_address, 
                deltaY_address,
                X_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                width, stride);
    }
    
   public Syncer sqBatchNorm2D_deltaAB_v2(
           long deltaA_address,//result0
           long deltaB_address,//result1
           long deltaY_address, long X_address, //V2: holdX(), X is not changed
           long X_mean_address, long X_sqmean_address, float eps,
           int row_lengthv, int lengthv, int width)
   {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_squmean is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
       
        return base.sqBatchNorm2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, 
                X_address,
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv,
                width, stride);
   }
   //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    public Syncer sqBatchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        
        return base.sqBatchNorm2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, 
                Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_gradients_v2(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps, 
                A_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv, lengthv, width, stride);
    }
    
    public Syncer batchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                row_lengthv, lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address, 
                Y_address, 
                X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>  
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public Syncer batchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaA_v2(deltaA_address, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                width, stride);
    }
    
   public Syncer batchNorm2D_deltaAB_v2(
           long deltaA_address,//result0
           long deltaB_address,//result1
           long deltaY_address,
           long X_address,//V2: holdX(), X is not changed
           long X_mean_address, long X_var_address, float eps,
           int row_lengthv, int lengthv, int width)
   {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                width, stride);
   }
   //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must >= 0", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, 
                Y_address, 
                X_var_address, eps, 
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must >= 0", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_gradients_v2(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2 
                deltaY_address, X_address,
                X_mean_address, X_var_address, eps, 
                A_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_leakyRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation (relu)">
    public Syncer batchNorm_relu2D(long Y_address,
            long X_address,
            long X_mean_address,
            long X_var_address, float eps, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_relu2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_relu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, int row_lengthv, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_relu2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward-propagation (leakyRelu)">
    public Syncer batchNorm_leakyRelu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_leakyRelu2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv, k, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_leakyRelu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_leakyRelu2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                row_lengthv, k,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm_leakyRelu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(k <= 0) throw new IllegalArgumentException(String.format("k { got %f } must > 0", k));
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_leakyRelu2D_deltaX_v1(deltaX_address,
                deltaY_address, k, 
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_leakyRelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_leakyRelu2D_deltaX_v2(deltaX_address,
                deltaY_address, k, 
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm_leakyRelu2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if (check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            if (k <= 0) throw new IllegalArgumentException(String.format("k { got %f } must > 0", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_leakyRelu2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, k,
                Y_address, 
                X_var_address, eps,
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_leakyRelu2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_leakyRelu2D_gradients_v2(
                deltaX_address,//result0 
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, k, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_elu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm_elu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, float alpha, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_elu2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv, alpha, k, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_elu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, float alpha, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_elu2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                row_lengthv, alpha, k,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm_elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(k <= 0) throw new IllegalArgumentException(String.format("k { got %f } must > 0", k));
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_elu2D_deltaX_v1(deltaX_address,
                deltaY_address, alpha, k, 
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must be non-negative", k));
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_elu2D_deltaX_v2(deltaX_address,
                deltaY_address, alpha, k, 
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm_elu2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, float alpha, float k,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_elu2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, alpha, k,
                Y_address, 
                X_var_address, eps,
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_elu2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, float alpha, float k,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (alpha < 0) throw new IllegalArgumentException(String.format("Elu: alpha {got %f} must >=0", alpha));
            if (k < 0) throw new IllegalArgumentException(String.format("Elu: negative_slope(got %f) must >= 0", k));
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_elu2D_gradients_v2(
                deltaX_address,//result0 
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, alpha, k, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm_softplus2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_softplus2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_softplus2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_softplus2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm_softplus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_softplus2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_softplus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_softplus2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm_softplus2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_softplus2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_softplus2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_softplus2D_gradients_v2(
                deltaX_address,//result0 
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm_gelu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_gelu2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_gelu2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_gelu2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm_gelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_gelu2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm_gelu2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_gelu2D_gradients_v2(
                deltaX_address,//result0 
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm_sigmoid2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_sigmoid2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_sigmoid2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_sigmoid2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm_sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_sigmoid2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_sigmoid2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm_sigmoid2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_sigmoid2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_sigmoid2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_sigmoid2D_gradients_v2(
                deltaX_address,//result0 
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm_tanh2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_tanh2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_tanh2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException(String.format("eps { got %f } must be non-negative", eps));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm_tanh2D(Y_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm_tanh2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_tanh2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_tanh2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if (check) {
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor X is null");
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_tanh2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm_tanh2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if (check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            if (B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_tanh2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm_tanh2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if (deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if (deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if (deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if (deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if (X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if (X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if (X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if (A_address == NULL) throw new NullPointerException("Tensor A is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm_tanh2D_gradients_v2(
                deltaX_address,//result0 
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D(Y_address,
                X_address, 
                X_mean_address,
                X_sqmean_address, eps, row_lengthv, 
                lengthv, width, stride);
    }
     
    public Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D(Y_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deltaX">
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, 
            long X_sqmean_address, float eps, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v1(deltaX_address,
                deltaY_address, 
                Y_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, 
            long X_sqmean_address, float eps, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address,
                X_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): deltaX">
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, 
            long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v2(deltaX_address,
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer layerNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long dX_address, //V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaA_v2(deltaA_address, 
                deltaY_address, 
                dX_address, 
                X_mean_address, 
                X_sqmean_address, eps, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): {deltaA, deltaB}">
    public Syncer layerNorm2D_deltaAB_v2(long deltaA_address, long deltaB_address,
            long deltaY_address, long X_address, //V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmsean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaAB_v2(deltaA_address, deltaB_address, 
                deltaY_address, X_address,
                X_mean_address,
                X_sqmean_address, eps,
                field_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot, pix2tensor">
    public Syncer onehot2D_row_int8(long Y_address,
            long X_address, 
            float alpha, float beta, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.onehot2D_row_int8(Y_address, 
                X_address, alpha, beta, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer onehot2D_row_int32(long Y_address,
            long X_address, 
            float alpha, float beta, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.onehot2D_row_int32(Y_address,
                X_address, alpha, beta, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer pix2tensor2D(long Y_address, 
            long X_address, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.pix2tensor2D(Y_address,
                X_address, 
                lengthv, width, stride);
    }
    
    public Syncer tensor2pix2D(long Y_address, 
            long X_address, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.tensor2pix2D(Y_address,
                X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="SGD">
    public Syncer sgd(long W_address,//Y += alpha * Xs[i] + beta, [from 1 to X.lenght]
            long[] gradients, float lr, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            Vector.requireNonNull(gradients, "W.gradients");
            func_param_check(lengthv, width, stride);
        }
        return base.sgd2D(W_address, gradients, lr,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    public Syncer sgdmn(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(momentum < 0) throw new IllegalArgumentException(String.format("momentum { got % f } must >= 0", momentum));
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D(W_address,
                V_address, momentum, dampen, nesterov, 
                deltaW_address, lr, 
                lengthv, width, stride);
    }
    
   public Syncer sgdmn(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr,
            int lengthv, int width)
    {
         int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(momentum < 0) throw new IllegalArgumentException(String.format("momentum must >= 0", momentum));
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D(W_address,
                V_address, momentum, dampen, nesterov, 
                gradients, lr, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SGDMN_decay">
    public Syncer sgdmn_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(momentum < 0) throw new IllegalArgumentException(String.format("momentum { got % f } must >= 0", momentum));
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D_decay(W_address,
                V_address, momentum, dampen, nesterov, 
                deltaW_address, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer sgdmn_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
         int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(momentum < 0) throw new IllegalArgumentException(String.format("momentum must >= 0", momentum));
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D_decay(W_address,
                V_address, momentum, dampen, nesterov, 
                gradients, lr,
                L1coef, L2coef,
                lengthv, width, stride);
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="Momentum">
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D(W_address, 
                V_address, a1, a2,
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long[] gradients, float lr_t,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D(W_address, 
                V_address, a1, a2,
                gradients, lr_t,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Momentum_decay">
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D_decay(W_address,
                V_address, a1, a2,
                deltaW_address, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long[] gradients, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D_decay(W_address, 
                V_address, a1, a2,
                gradients, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D(W_address,
                S_address, a1, a2, eps_t,
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long[] gradients , float lr_t, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D(W_address,
                S_address, a1, a2, eps_t,
                gradients, lr_t,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop_decay">
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D_decay(W_address,
                S_address, a1, a2, eps_t,
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long[] gradients , float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D_decay(W_address,
                S_address, a1, a2, eps_t,
                gradients, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adam">
    //<editor-fold defaultstate="collapsed" desc="adam2D_type2">
    public Syncer adam2D_type2(long W_address,
            long V_address, float a1, float a2, float Uv, 
            long S_address, float b1, float b2, float eps, float Us,
            long deltaW_address, float lr,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            if(a1 + a2 < 0.99999f || a1 + a2 > 1.00001f) throw new IllegalArgumentException("a1 + a2 != 1");
            if(b1 + b2 < 0.99999f || b1 + b2 > 1.00001f) throw new IllegalArgumentException("b1 + b2 != 1");
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D_type2(W_address, 
                V_address, a1, a2, Uv,
                S_address, b1, b2, eps, Us, 
                deltaW_address, lr,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                gradients, lr_t,
                lengthv, width, stride);
    }
       
   
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam_decay">
    public Syncer adam2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D_decay(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer adam2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D_decay(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                gradients, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
     }
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="Adam_AMSgrad">
    public Syncer adam_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address,
            long deltaW_address, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(Smax_address == NULL) throw new NullPointerException("Tensor Smax is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam_amsgrad2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                Smax_address, 
                deltaW_address, lr_t, 
                lengthv, width, stride);
    }
    
    public Syncer adam_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address,
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(Smax_address == NULL) throw new NullPointerException("Tensor Smax is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam_amsgrad2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                Smax_address,
                gradients, lr_t,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam_AMSgrad_decay">
    public Syncer adam_amsgrad2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(Smax_address == NULL) throw new NullPointerException("Tensor Smax is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam_amsgrad2D_decay(W_address,
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                Smax_address, deltaW_address, lr_t, 
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer adam_amsgrad2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(Smax_address == NULL) throw new NullPointerException("Tensor Smax is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adam_amsgrad2D_decay(W_address,
                V_address, a1, a2,
                S_address, b1, b2, eps_t,
                Smax_address, 
                gradients, lr_t, L1coef, L2coef, 
                lengthv, width, stride);
     }
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got %f } must >= 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got %f } must >= 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got %f } must >= 0", b1));
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D(W_address,
                V_address, a1, a2, 
                S_address, b1, eps,
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got %f} must >= 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got %f} must >= 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got %f} must >= 0", b1));
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D(W_address,
                V_address, a1, a2,
                S_address, b1, eps,
                gradients, lr_t, 
                lengthv, width, stride);
     }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax_decay">
    public Syncer adamax2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got %f } must >= 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got %f } must >= 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got %f } must >= 0", b1));
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D_decay(W_address,
                V_address, a1, a2, 
                S_address, b1, eps,
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamax2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got %f } must >= 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got %f } must >= 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got %f } must >= 0", b1));
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D_decay(W_address, 
                V_address, a1, a2,
                S_address, b1, eps,
                gradients, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
     }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="AdamW">
     public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamW2D(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t, lr,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamW2D(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                gradients, lr_t, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
     }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="AdamW_AMSgrad">
    public Syncer adamW_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address,
            long deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(Smax_address == NULL) throw new NullPointerException("Tensor Smax is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamW_amsgrad2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                Smax_address, 
                deltaW_address, lr_t, lr,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamW_amsgrad2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long Smax_address,
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(Smax_address == NULL) throw new NullPointerException("Tensor Smax is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamW_amsgrad2D(W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                Smax_address,
                gradients, lr_t, lr,
                L1coef, L2coef,
                lengthv, width, stride);
     }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(G_address == NULL) throw new NullPointerException("Tensor G is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            if(c1 <= 0) throw new IllegalArgumentException(String.format("c1 { got % f } must > 0", c1));
            if(c2 <= 0) throw new IllegalArgumentException(String.format("c2 { got % f } must > 0", c2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamod2D(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                deltaW_address, lr_t, 
                lengthv, width, stride);
    }
    
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            if(c1 <= 0) throw new IllegalArgumentException(String.format("c1 { got % f } must > 0", c1));
            if(c2 <= 0) throw new IllegalArgumentException(String.format("c2 { got % f } must > 0", c2));
             func_param_check(lengthv, width, stride);
        }

        return base.adamod2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                G_address, c1, c2,
                gradients, lr_t, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod_decay">
    public Syncer adamod2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(G_address == NULL) throw new NullPointerException("Tensor G is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", a1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", a2));
            if(c1 <= 0) throw new IllegalArgumentException(String.format("c1 { got % f } must > 0", c1));
            if(c2 <= 0) throw new IllegalArgumentException(String.format("c2 { got % f } must > 0", c2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamod2D_decay(W_address,
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                deltaW_address, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamod2D_decay(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException(String.format("a1 { got % f } must > 0", a1));
            if(a2 <= 0) throw new IllegalArgumentException(String.format("a2 { got % f } must > 0", a2));
            if(b1 <= 0) throw new IllegalArgumentException(String.format("b1 { got % f } must > 0", b1));
            if(b2 <= 0) throw new IllegalArgumentException(String.format("b2 { got % f } must > 0", b2));
            if(c1 <= 0) throw new IllegalArgumentException(String.format("c1 { got % f } must > 0", c1));
            if(c2 <= 0) throw new IllegalArgumentException(String.format("c2 { got % f } must > 0", c2));
            func_param_check(lengthv, width, stride);
        }
        return base.adamod2D_decay(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                gradients, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    protected int next_seed() { return exr.nextInt(); } 
    public EngineCore set_seed(long seed) { exr.setSeed(seed); return this; }
    
    public Syncer bernouli2D(long X_address, 
            float p, float v1, float v2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(p<0 || p>1) throw new IllegalArgumentException(String.format("p { got %f } must belong to [0,1]", p));
            func_param_check(lengthv, width, stride);
        }
        return base.bernouli2D(X_address,
                next_seed(),
                p, v1, v2, 
                lengthv, width, stride);
    }
    
    public Syncer bernouli2D_mul(long Y_address, long R_address,
            long X_address, 
            float p, float v1, float v2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(R_address == NULL) throw new NullPointerException("Tensor R is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(p<0 || p>1) throw new IllegalArgumentException(String.format("p { got %f } must belong to [0,1]", p));
            func_param_check(lengthv, width, stride);
        }
        return base.bernouli_mul2D(Y_address, R_address,
                X_address, 
                next_seed(), 
                p, v1, v2,
                lengthv, width, stride);
    }
    
    public Syncer leakyRelu_bernouli2D_mul(long Y_address, long R_address,
            long X_address, 
            float k, float p, float v1, float v2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(R_address == NULL) throw new NullPointerException("Tensor R is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(k < 0) throw new IllegalArgumentException(String.format("k { got %f } must >= 0", k));
            if(p<0 || p>1) throw new IllegalArgumentException(String.format("p { got %f } must belong to [0,1]", p));
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu_bernouli_mul2D(Y_address, R_address, 
                X_address, 
                k, next_seed(), 
                p, v1, v2, 
                lengthv, width, stride);
    }
    
    public Syncer uniform2D(long X_address, 
            float vmin, float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        if(vmin < vmax) { float t = vmin; vmin = vmax; vmax = t; }
        return base.uniform2D(X_address, 
                next_seed(), 
                vmin, vmax, 
                lengthv, width, stride);
    }
     
    public Syncer sparse_uniform2D(long X_address, 
            float p, float vmin, float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(p<0 || p>1) throw new IllegalArgumentException(String.format("p { got %f } must belong to [0,1]", p));
            func_param_check(lengthv, width, stride);
        }
        if(vmin < vmax) { float t = vmin; vmin = vmax; vmax = t; }
        return base.sparse_uniform2D(X_address,
                next_seed(), next_seed(),
                p, vmin, vmax, 
                lengthv, width, stride);
    }
    
    public Syncer gaussian2D(long X_address,
            float mu, float sigma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(sigma < 0) throw new IllegalArgumentException(String.format(
                    "sigma { got %f } (the standard deviation) must be positive", sigma));
            func_param_check(lengthv, width, stride);
        }
        return base.gaussian2D(X_address, 
                next_seed(), next_seed(), 
                mu, sigma, 
                lengthv, width, stride);
    }
    
    public Syncer sparse_gaussian2D(long X_address,
            float p, float mu, float sigma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(sigma < 0) throw new IllegalArgumentException(String.format(
                    "sigma { got %f } (the standard deviation) must> 0", sigma));
            if(p<0 || p>1) throw new IllegalArgumentException(String.format("p { got %f } must belong to [0, 1]", p));
            func_param_check(lengthv, width, stride);
        }
        return base.sparse_gaussian2D(X_address, 
                next_seed(), next_seed(), next_seed(), 
                p, mu, sigma,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="param_check">
    protected void field_reduce_param_check(int length, int row_length, int width) {
        if(length < width) throw new IllegalArgumentException(String.format(
                "length { got %d } < width { got %d }", length, width));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        if(length % row_length != 0) throw new IllegalArgumentException(String.format(
                "length { got %d } %% row_length { got %d }", length, row_length));
        if(row_length % width != 0) throw new IllegalArgumentException(String.format(
                "row_length { got %d } %% width { got %d } ! = 0", row_length, width)); 
    }
    
    protected void center_reduce_param_check(int dim0, int dim1, int dim2, int width) {
        if(width <= 0) throw new IllegalArgumentException(String.format("width { got %d } must > 0", width));
        if(dim0 <= 0) throw new IllegalArgumentException(String.format("dim0 { got %d } must > 0", dim0));
        if(dim1 <= 0) throw new IllegalArgumentException(String.format("dim1 { got %d } must > 0", dim1));
        if(dim2 < width) throw new IllegalArgumentException(String.format(
                "dim2 { got %d } < width { got %d }", dim2, width));
        if(dim2 % width != 0) throw new IllegalArgumentException(String.format(
                "dim2 { got %d } %% width { got %d } ! = 0", dim2, width)); 
    }
    
    protected void row_reduce_param_check(int field_length, int row_length, int width){
        if(field_length <= 0) throw new IllegalArgumentException(String.format(
                "field_length { got %d } <= 0", field_length));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        if(row_length % width != 0) throw new IllegalArgumentException(String.format(
                "row_length { got %d } %% width { got %d } ! = 0", row_length, width)); 
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    public Result<Float> straight_linear(long X_address,
            float alpha, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_linear(X_address, alpha, beta,  
                lengthv, width, stride);
    }
    
    public Result<Float> straight_quadratic(long X_address,
            float alpha, float beta, float gamma,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_quadratic(X_address, alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Result<Float> straight_max(long X_address,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_max(X_address, 
                lengthv, width, stride);
    }
    
    public Result<Float> straight_min(long X_address,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_min(X_address, 
                lengthv, width, stride);
    }
    
    public IndexedResult<Float> straight_max_indexed(long X_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_max_indexed(X_address,
                lengthv, width, stride);
    }
     
    public IndexedResult<Float> straight_min_indexed(long X_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_min_indexed(X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field reduce function">
    //<editor-fold defaultstate="collapsed" desc="field linear">
    public Syncer field_linear(long Y_address, 
            long X_address, float alpha, float beta, 
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_linear(Y_address,
                X_address, alpha, beta,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_linear2(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int length, int row_length, int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        
        return base.field_linear2(Y_address,
                X1_address, X2_address,
                alpha, beta, gamma, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field quadratic">
    public Syncer field_quadratic(long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_quadratic(Y_address,
                X_address, alpha, beta, gamma, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_quadratic2(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int length, int row_length, int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_quadratic2(Y_address, 
                X1_address, X2_address, 
                k11, k12, k22,
                k1, k2, C, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_linear_quadratic & var & std">
    public Syncer field_linear_quadratic(long Y1_address, long Y2_address,
            long X_address, 
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int length, int row_length, int width) 
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y1_address == NULL) throw new NullPointerException("Tensor Y1 is null");
            if(Y2_address == NULL) throw new NullPointerException("Tensor Y2 is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_linear_quadratic(Y1_address, Y2_address, 
                X_address, 
                alpha1, beta1, 
                alpha2, beta2, gamma2, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_var_mean(boolean unbiased,
            long var_address, //result0
            long mean_address,//result1
            long X_address,
            int length, int row_length, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(var_address == NULL) throw new NullPointerException("Tensor var is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(unbiased && field_length < 2) throw new IllegalArgumentException(String.format(
                    "to find the unbiased stddev, field_length { got %d }must >= 2", field_length));
            field_reduce_param_check(length, row_length, width);
        }
        return base.field_var_mean(unbiased,
                var_address, //result0
                mean_address,//result1
                X_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer field_std_mean(boolean unbiased,
            long std_address, //result0
            long mean_address,//result1
            long X_address,
            int length, int row_length, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(std_address == NULL) throw new NullPointerException("Tensor std is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(unbiased && field_length < 2) throw new IllegalArgumentException(String.format(
                    "to find the unbiased variance, field_length { got %d } must >= 2", field_length));
            field_reduce_param_check(length, row_length, width);
        }
        return base.field_std_mean(unbiased,
                std_address, //result0
                mean_address,//result1
                X_address,
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field max, minValue">
    public Syncer field_max(long Y_address,
            long X_address, 
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_max(Y_address,
                X_address, field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer field_min(long Y_address,
            long X_address, 
            int length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_min(Y_address,
                X_address, field_length, row_lengthv, 
                width, stride);
    }
    
     public Syncer field_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_max_indexed(Y_address, Index_address,
                X_address, field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_min_indexed(long Y_address, long Index_address,
            long X_address, 
            int length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_min_indexed(Y_address, Index_address,
                X_address, field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="center reduce function">
    public Syncer center_linear(long Y_address, 
            long X_address,
            float alpha, float beta,
            int dim0, int dim1, int dim2,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            center_reduce_param_check(dim0, dim1, dim2, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        dim2 = (dim2 / width) * stride;
        return base.center_linear(Y_address, 
                X_address, 
                alpha, beta, 
                dim0, dim1, dim2,
                width, stride);
    }
    
    public Syncer center_quadratic(long Y_address, 
            long X_address,
            float alpha, float beta, float gamma,
            int dim0, int dim1, int dim2,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            center_reduce_param_check(dim0, dim1, dim2, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        dim2 = (dim2 / width) * stride;
        return base.center_quadratic(Y_address,
                X_address, 
                alpha, beta, gamma, 
                dim0, dim1, dim2, 
                width, stride);
    }
    
    //<editor-fold defaultstate="collapsed" desc="center_quadratic2">
    public Syncer center_quadratic2(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int dim0, int dim1, int dim2,
            int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            center_reduce_param_check(dim0, dim1, dim2, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        dim2 = (dim2 / width) * stride;
        return base.center_quadratic2(Y_address,
                X1_address, X2_address, 
                k11, k12, k22,
                k1, k2, C, 
                dim0, dim1, dim2, 
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row reduce function">
    //<editor-fold defaultstate="collapsed" desc="row linear">
    public Syncer row_linear(long Y_address, 
            long X_address, float alpha, float beta, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_linear(Y_address, 
                X_address, alpha, beta,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_linear2(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        
        return base.row_linear2(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                field_length, 
                row_lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row quadratic">
    public Syncer row_quadratic(long Y_address, 
            long X_address, float alpha, float beta, float gamma, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_quadratic(Y_address, 
                X_address, alpha, beta, gamma,
                field_length, row_lengthv,
                width, stride);
    }

    public Syncer row_quadratic2(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22, 
            float k1, float k2, float C, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_quadratic2(Y_address,
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, C, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear_quadratic & var & std">
    public Syncer row_linear_quadratic(long Y1_address, long Y2_address,
            long X_address, 
            float alpha1, float beta1, 
            float alpha2, float beta2, float gamma2,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y1_address == NULL) throw new NullPointerException("Tensor Y1 is null");
            if(Y2_address == NULL) throw new NullPointerException("Tensor Y2 is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_linear_quadratic(Y1_address, Y2_address,
                X_address, 
                alpha1, beta1,
                alpha2, beta2, gamma2, 
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer row_var_mean(boolean unbiased,
            long var_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(var_address == NULL) throw new NullPointerException("Tensor var is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(unbiased && row_length < 2) throw new IllegalArgumentException(String.format(
                    "to find the unbiased variance, row_length { got %d } must >= 2", row_length));
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_lengthv  = ((field_length + 3) >> 2) << 2;
        return base.row_var_mean(unbiased,
                var_address, //result0
                mean_address,//result1
                X_address,
                field_length, field_lengthv, 
                row_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer row_std_mean(boolean unbiased,
            long std_address, //result0
            long mean_address,//result1
            long X_address, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(std_address == NULL) throw new NullPointerException("Tensor std is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(unbiased && row_length < 2) throw new IllegalArgumentException(String.format(
                    "to find the unbiased variance, row_length { got %d } must >= 2", row_length));
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        return base.row_std_mean(unbiased,
                std_address, //result0
                mean_address,//result1
                X_address,
                field_length, field_lengthv,
                row_length, row_lengthv,
                width, stride);
    }
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="row max, minValue">
    public Syncer row_max(long Y_address, 
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_max(Y_address,
                X_address,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_min(long Y_address, 
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_min(Y_address,
                X_address,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_max_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_max_indexed(Y_address, Index_address,
                X_address, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_min_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_min_indexed(Y_address, Index_address,
                X_address, 
                field_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Image Function<pixel: unit8>">
    //<editor-fold defaultstate="collapsed" desc="auxilary: check_image_size">
    protected void check_image_size(int IH, int IW, int OH, int OW, int N, int C) {
        if(IH <= 0) throw new IllegalArgumentException(String.format("IH { got %d } must > 0", IH));
        if(IW <= 0) throw new IllegalArgumentException(String.format("IW { got %d } must > 0", IW));
        if(OH <= 0) throw new IllegalArgumentException(String.format("OH { got %d } must > 0", OH));
        if(OW <= 0) throw new IllegalArgumentException(String.format("OW { got %d } must > 0", OW));
        if(N <= 0) throw new IllegalArgumentException(String.format("N { got %d } must > 0", IH));
        if(C <= 0) throw new IllegalArgumentException(String.format("C { got %d } must > 0", C));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: pixel(uint8) to dtype">
    //<editor-fold defaultstate="collapsed" desc="linear: pixel to dtype">
    public Syncer linear2D_pixel_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_pixel_to_dtype(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
     
    public Syncer linear2D_dtype_to_pixel(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_dtype_to_pixel(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: dualLinear2_div2D">
    public Syncer img_dualLinear2_div2D(long Y_address, 
            long X_address, 
            long X1_address, long X2_address,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.img_dualLinear2_div2D(Y_address, 
                X_address,
                X1_address, X2_address, 
                alpha1, beta1, gamma1,
                alpha2, beta2, gamma2, C, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: dualLinear2_normalize2D: row, center">
    public Syncer img_dualLinear2_normalize2D_row(long Y_address, 
            long X_address,
            long X1_address, long X2_address, int row_lengthv, 
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.img_dualLinear2_normalize2D_row(Y_address, 
                X_address,
                X1_address, X2_address, row_lengthv, 
                alpha1, beta1, gamma1, 
                alpha2, beta2, gamma2, C,
                lengthv, width, stride);
    }
    
    public Syncer img_dualLinear2_normalize2D_center(long Y_address, 
            long X_address,
            long X1_address, long X2_address,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C,
            int dim0, int dim1, int dim2,
            int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_center(dim0, dim1, dim2, width);
        }
        return base.img_dualLinear2_normalize2D_center(Y_address, 
                X_address, 
                X1_address, X2_address, 
                alpha1, beta1, gamma1,
                alpha2, beta2, gamma2, C,
                dim0, dim1, dim2,
                width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: linear2_div2D_row, field">
    public Syncer img_linear2_div2D_row(long Y_address,
            long X_address, 
            long X1_address, long X2_address, int row_lengthv,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(String.format(
                    "(alpha2 { got %f } * X2 + beta2 { got %f }) identically equals to zero", alpha2, beta2));
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.img_linear2_div2D_row(Y_address, 
                X_address, 
                X1_address, X2_address, row_lengthv,
                alpha1, beta1, gamma1,
                alpha2, beta2, C,
                lengthv, width, stride);
    }
    
    public Syncer img_linear2_div2D_field(long Y_address,
            long X_address,
            long X1_address, long X2_address, int field_length,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(String.format(
                    "(alpha2 { got %f } * X2 + beta2 { got %f }) identically equals to zero", alpha2, beta2));
            func_param_check_field(lengthv,  field_length, row_lengthv, width, stride); 
        }
        return base.img_linear2_div2D_field(Y_address,
                X_address, 
                X1_address, X2_address, row_lengthv,
                alpha1, beta1, gamma1, 
                alpha2, beta2, C,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: elementwise functions">
    public Syncer img_linear2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.img_linear2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    //<editor-fold defaultstate="collapsed" desc="image: linear_dual2D_row, field">
    public Syncer img_linear_dual2D_row(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.img_linear_dual2D_row(Y_address, 
                X1_address,
                X2_address, row_lengthv, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    
    public Syncer img_linear_dual2D_field(long Y_address,
            long X1_address, 
            long X2_address, int field_length,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_field(lengthv,  field_length, row_lengthv, width, stride); 
        }
        return base.img_linear_dual2D_field(Y_address, 
                X1_address,
                X2_address, row_lengthv,
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    public Syncer img_quadratic2D(long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.img_quadratic2D(Y_address, 
                X_address, alpha, beta, gamma,
                lengthv, width, stride);
    }
    
    public Syncer img_threshold2D(long Y_address, 
            long X_address, float alpha, float v, byte v1, byte v2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.img_threshold2D(Y_address, 
                X_address, alpha, v, v1, v2,
                lengthv, width, stride);
    }
    
    public Syncer img_log2D(long Y_address, 
            float C, float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.img_log2D(Y_address, 
                C, alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer img_exp2D(long Y_address, 
            float alpha, long X_address, float beta, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.img_exp2D(Y_address,
                alpha, X_address, beta, C,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: tensor trick">
    //<editor-fold defaultstate="collapsed" desc="image: pad, trim, transpose">
    public Syncer img_pad(
            long Y_address, int OH, int OW, int OC,
            long X_address, int IH, int IW, int IC,
            int N, int ph0, int pw0, int pc0) 
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(IH + ph0 > OH) throw new IllegalArgumentException(String.format(
                    "IH { got %d } + ph0 { got %d } > OH { got %d }", IH, ph0, OH));
            if(IW + pw0 > OW) throw new IllegalArgumentException(String.format(
                    "IW { got %d } + pw0 { got %d } > OW { got %d }", IW, pw0, OW));
            if(IC + pc0 > OC) throw new IllegalArgumentException(String.format(
                    "IC { got %d } + pc0 { got %d } > OC { got %d }", IC, pc0, OC));
        }
        return base.img_pad(
                Y_address, OH, OW, (OC + 3) >> 2 << 2, 
                X_address, IH, IW, (IC + 3) >> 2 << 2, 
                N, ph0, pw0, pc0);
    }
    
    public Syncer img_trim(//X.ndim = Y.ndim = p0.length
            long Y_address, int OH, int OW, int OC, 
            long X_address, int IH, int IW, int IC,
            int N, int ph0, int pw0, int pc0)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(IH - ph0 < OH) throw new IllegalArgumentException(String.format(
                    "IH { got %d } - ph0 { got %d } < OH { got %d }", IH, ph0, OH));
            if(IW - pw0 < OW) throw new IllegalArgumentException(String.format(
                    "IW { got %d } - pw0 { got %d } < OW { got %d }", IW, pw0, OW));
            if(IC - pc0 < OC) throw new IllegalArgumentException(String.format(
                    "IC { got %d } - pc0 { got %d } < OC { got %d }", IC, pc0, OC));
        }
        return base.img_trim(
                Y_address, OH, OW, (OC + 3) >> 2 << 2, 
                X_address, IH, IW, (IC + 3) >> 2 << 2,
                N, ph0, pw0, pc0);
    }
    
    public Syncer img_transpose(
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim, 
            int dimIndex1, int dimIndex2, 
            int widthX, int widthY, 
            int length) 
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Ydim.length < 2) throw new IllegalArgumentException(String.format(
                    "Y.ndim { got %d } must >= 2", Ydim.length));
            if(Xdim.length < 2) throw new IllegalArgumentException(String.format(
                    "X.ndim { got %d } must >= 2", Xdim.length));
            if(length < widthX) throw new IllegalArgumentException(String.format(
                    "length { got %d } < X.width { got %d }", length, widthX));
            if(length < widthY) throw new IllegalArgumentException(String.format(
                    "length { got %d } < Y.width { got %d }", length, widthX));
            if(widthX <= 0) throw new IllegalArgumentException(String.format(
                    "X.width { got %d } must > 0", widthX));
            if(widthY <= 0) throw new IllegalArgumentException(String.format(
                    "Y.width { got %d } must > 0", widthY));
            if(length % widthX != 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } %% X.width { got %d } != 0", length, widthX));
            if(length % widthY != 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } %% Y.width { got %d } != 0", length, widthX));
        }
        return base.img_transpose(
                Y_address, Ydim,
                X_address, Xdim, 
                dimIndex1, dimIndex2, 
                (widthX + 3) >> 2 << 2,
                (widthY + 3) >> 2 << 2,
                length);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: resize, affine">
    public Syncer img_resize(
            long X_address, int IH, int IW,
            long Y_address, int OH, int OW, 
            int N, int C)
    {
        if(check) { 
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            check_image_size(IH, IW, OH, OW, N, C); 
        }
        return base.img_resize(
                X_address, IH, IW, 
                Y_address, OH, OW, 
                N, (C + 3) >> 2 << 2);
    }
    
    public Syncer img_affine(
            long X_address, int IH, int IW,
            long Y_address, int OH, int OW, 
            float r00, float r01, float r02, 
            float r10, float r11, float r12,
            int N, int C)
    {
        if(check) { 
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            check_image_size(IH, IW, OH, OW, N, C); 
        }
        return base.img_affine(
                X_address, IH, IW, 
                Y_address, OH, OW, 
                r00, r01, r02, 
                r10, r11, r12,
                N, (C + 3) >> 2 << 2);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: gappedMemcpy2D, extract_3channels">
    public Syncer img_gappedMemcpy2D(
            long X_address, int Xstart, int strideX, 
            long Y_address, int Ystart, int strideY,
            int width, int length)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Xstart < 0) throw new IllegalArgumentException(String.format("Xstart { got %d } < 0", Xstart));
            if(Ystart < 0) throw new IllegalArgumentException(String.format("Ystart { got %d } < 0", Ystart));
            if(strideX < width) throw new IllegalArgumentException(String.format(
                    "strideX { got %d } < width { got %d }", strideX, width));
            if(strideY < width) throw new IllegalArgumentException(String.format(
                    "strideY { got %d } < width { got %d }", strideX, width));
            if(length < width) throw new IllegalArgumentException(String.format(
                    "length { got %d } < width { got %d }", length, width));
            if(length % width != 0) throw new IllegalArgumentException(String.format(
                    "length { got %d } %% width { got %d } != 0", length, width));
            if(width < 0) throw new IllegalArgumentException(String.format(
                    "width { got %d } < 0", width));
        }
        return base.img_gappedMemcpy2D(
                X_address, Xstart, strideX,
                Y_address, Ystart, strideY, 
                width, length);
    }
    
    public Syncer extract_3channels(
            long X_address, int IC, 
            long Y_address, int c0, int c1, int c2, 
            int lengthv)//lengthv = N*IH*IW = X.lengthv 
    {
        int stride =  (IC + 3) >> 2 << 2;//width -> stride
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(c0 < 0 || c0 >= IC) throw new IllegalArgumentException(String.format(
                    "channel0 { got %d } should belong to [%d, %d)", c0, 0, IC));
            if(c1 < 0 || c1 >= IC) throw new IllegalArgumentException(String.format(
                    "channel1 { got %d } should belong to [%d, %d)", c1, 0, IC));
            if(c2 < 0 || c2 >= IC) throw new IllegalArgumentException(String.format(
                    "channel2 { got %d } should belong to [%d, %d)", c2, 0, IC));
            if(lengthv < 0) throw new IllegalArgumentException(String.format(
                    "lengthv(N*IH*IW) { got %d } must >= 0", lengthv));
        }
        return base.extract_3channels(
                X_address, stride,//IC <- stride
                Y_address,
                c0, c1, c2, lengthv);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
}
