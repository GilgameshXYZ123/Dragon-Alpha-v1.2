/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import z.dragon.common.MemStatus;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Result.IndexedResult;
import z.dragon.engine.Syncer.ChainSyncer;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.engine.memp.Mempool;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;
import z.dragon.nn.unit.Unit;
import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 */
public class Engine implements MemStatus {
    protected EngineCore core;
    protected boolean check = true;
    protected boolean sync = true;
    
    protected SyncRandomEngine create_sync_random_engine() { return new SyncRandomEngine(this); }
    protected ImageEngine create_image_engine() { return new ImageEngine(this); }
    protected EngineAlloc create_engine_alloc() { return new EngineAlloc(this); }
    
    public final ImageEngine img = create_image_engine();
    public final SyncRandomEngine srand = create_sync_random_engine();
    public final EngineAlloc alloc = create_engine_alloc();
    
    public static final float HALF_PI = (float) (0.5 * Math.PI);
    
    public Engine() {}
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public ExRandom random() { return core.exr; }
    public Engine random(ExRandom random) { core.random(random); return this; }
    
    public EngineCore engineCore() { return core; }
    public Mempool mempool() { return core.mempool; } 
    public synchronized Engine engineCore(EngineCore core) {
        if(core == null) throw new NullPointerException("EngineCore is null");
        this.core = core;
        this.check = core.check;
        return this;
    }
    
    public EngineBase engineBase() { return core.base; }

    public final boolean check() {return check;}
    public Engine check(boolean flag) { check = flag; core.check(flag); return this; }
    
    public final boolean sync() { return sync; }
    public Engine sync(boolean flag) { this.sync = flag; return this; }
    
    @Override public long max_mem_size() { return core.max_mem_size(); }
    @Override public long total_mem_size() { return core.total_mem_size(); }
    @Override public long used_mem_size() {return core.used_mem_size_MB();}
    
    public Engine max_mem_size(long maxMemSize) { core.max_mem_size(maxMemSize); return this; }

    public long bufferedMemSize() {return core.buffered_mem_size();}
    public long bufferedMemSize_MB() {return core.buffered_mem_size_MB();}
    
    public final String dataType() { return core.dataType(); }
    public final String dataType_int32() { return core.dataType_int32(); }
    public final String dataType_int8() { return core.dataType_int8(); }
    public final long sizeOf_dataType() { return core.sizeOf_dataType(); }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append("\nsync = ").append(sync);
        sb.append("\ncheck = ").append(check);
        Mempool mempool = mempool();
        if(mempool != null) mempool.meta_data().forEach((String key, Object value)-> {
            sb.append("\n\tmempool.").append(key).append(" = ").append(value);
        });
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 37 * hash + Objects.hashCode(this.core);
        return hash;
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof Engine)) return false;
        Engine eg = (Engine) o;
        return Objects.equals(eg.core, core);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="built-int-checkers">
    //<editor-fold defaultstate="collapsed" desc="checker: require-dataType">
    protected void require_dtype(Tensor... Xs) { for(Tensor X : Xs) require_dtype(X); }
    protected void require_dtype(Collection<Tensor> Xs) { for(Tensor X : Xs) require_dtype(X); }
    protected void require_dtype(Tensor X) {
        if(!this.equals(X.engine())) throw new IllegalArgumentException("Invalid Engine");
        if(!X.dataType.equals(core.dataType())) throw new IllegalArgumentException(String.format(
                "tensor.dataType { got %s } != engine.dataType { got %s }", 
                X.dataType, core.dataType()));
    }
    
    protected void require_dtype(Tensor[] Xs, String name) { 
        for(int i=0; i<Xs.length; i++) require_dtype(Xs[i], "X[" + i + "]");
    }
    protected void require_dtype(Collection<Tensor> Xs, String name) {
        int idx = 0; for(Tensor X : Xs) require_dtype(X, name + '[' + (idx++) + ']');
    }
    protected void require_dtype(Tensor X, String name) {
        if(!this.equals(X.engine())) throw new IllegalArgumentException("Invalid Engine");
        if(!X.dataType.equals(core.dataType())) throw new IllegalArgumentException(String.format(
                "%s.dataType { got %s } != engine.dataType { got %s }", 
                name, X.dataType, core.dataType()));
    }
    
    protected void require_int32(Tensor X) {
        if(!this.equals(X.engine())) throw new IllegalArgumentException("Invalid Engine");
        if(!X.dataType.equals(core.dataType_int32())) throw new IllegalArgumentException(String.format(
                "tensor.dataType { got %s } != engine.int32 { got %s }", 
                X.dataType, core.dataType_int32()));
    }
    protected void require_int32(Tensor X, String name) {
        if(!this.equals(X.engine())) throw new IllegalArgumentException("Invalid Engine");
        if(!X.dataType.equals(core.dataType_int32())) throw new IllegalArgumentException(String.format(
                "%s.dataType { got %s } != engine.int32 { got %s }", 
                name, X.dataType, core.dataType_int32()));
    }
    
    protected void require_int8(Tensor... Xs) { for(Tensor X : Xs) require_int8(X); }
    protected void require_int8(Tensor X) {
        if(!this.equals(X.engine())) throw new IllegalArgumentException("Invalid Engine");
        if(!X.dataType.equals(core.dataType_int8())) throw new IllegalArgumentException(String.format(
                "tensor.dataType { got %s } != engine.int8 { got %s }",
                X.dataType, core.dataType_int8()));
    }
     protected void require_int8(Tensor X, String name) {
        if(!this.equals(X.engine())) throw new IllegalArgumentException("Invalid Engine");
        if(!X.dataType.equals(core.dataType_int8())) throw new IllegalArgumentException(String.format(
                "%s.dataType { got %s } != engine.int8 { got %s }",
                name, X.dataType, core.dataType_int8()));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="checker: must_size">
    protected final void must_equal(int V1, String name1, int V2) {
        if(V1 != V2) throw new IllegalArgumentException(String.format(
                "%s { got %d } must == %d", name1, V1, V2));
    }
     
    protected final void must_greater_equal(int V1, String name1, int V2) {
        if(V1 < V2) throw new IllegalArgumentException(String.format(
                "%s { got %d } must >= %d", name1, V1, V2));
    }
    
    protected final void must_smaller_equal(int V1, String name1, int V2) {
        if(V1 > V2) throw new IllegalArgumentException(String.format(
                "%s { got %d } must <=  %d", name1, V1, V2));
    }
    protected final void must_smaller_equal(int V1, String name1, int V2, String name2) {
        if(V1 > V2) throw new IllegalArgumentException(String.format(
                "%s { got %d } must <= %s { got %d }", name1, V1, name2, V2));
    }
    
    protected final void must_non_negative(int[] arr, String name) {
        for(int a : arr) if(a < 0) throw new IllegalArgumentException(String.format(
                "%s { got %s } must be non-negative", name, Arrays.toString(arr)));
    }
    
    protected final void must_positive(int[] arr, String name) {
        for(int a : arr) if(a <= 0) throw new IllegalArgumentException(String.format(
                "%s { got %s } must be positive", name, Arrays.toString(arr)));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="checker: equals">
    protected final void equals(int V1, String name1, int V2) {
        if(V1 != V2) throw new IllegalArgumentException(String.format(
                "%s { got %d } != %d }", name1, V1, V2));
    }
    
    protected final void equals(int V1, String name1, int V2, String name2) {
        if(V1 != V2) throw new IllegalArgumentException(String.format(
                "%s { got %d } != %s { got %d }", name1, V1, name2, V2));
    }
    
    protected final void equals_dim(Tensor X1, String name1, Tensor X2, String name2) {
        if(!X1.dimEquals(X2)) throw new IllegalArgumentException(String.format(
                "%s.dim { got %s } != %s.dim { got %s}", 
                name1, Arrays.toString(X1.dim), 
                name2, Arrays.toString(X2.dim)));
    }
    
    protected void equals_dataType(Tensor X1, String name1, Tensor X2, String name2) {
        if(!X1.dataType.equals(X2.dataType)) throw new IllegalArgumentException(String.format(
                "%s.dataType { got %s } != %s.dataType { got %s }",
                name1, X1.dataType, 
                name2, X2.dataType));
    }
    
    protected final void equals_valueStructure(Tensor X1, String name1, Tensor X2, String name2){
        if(!X1.valueStrucEquals(X2))throw new IllegalArgumentException(String.format(
                "%s.valueStructure is different from that of %s",  name1, name2));
    }
    
    protected final void equals_valueStructure(Tensor X1, String name1, Collection<Tensor> Xs, String name2) {
        int idx = 0; for(Tensor X2 : Xs)
            if(!X1.valueStrucEquals(X2)) throw new IllegalArgumentException(String.format(
                "%s.valueSturcture is different from that of %s[%d]", name1, name2, idx++));
    }
    
    protected final void equals_valueStructure(Tensor X1, String name1, Tensor[] Xs, String name2) {
        for(int i=0; i<Xs.length; i++)
            if(!X1.valueStrucEquals(Xs[i])) throw new IllegalArgumentException(String.format(
                "%s.valueSturcture is different from that of %s[%d]", name1, name2, i));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="extra: Tensor<int32>">
    //<editor-fold defaultstate="collapsed" desc="Tensor<int32>: create">
    public Tensor empty_int32_like(Tensor X) { return empty_int32(X.dim); }
    public Tensor empty_int32(int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType_int32(), dim);
        ts.bindMemory(core.malloc_int32(ts.length_4x));
        
        int compare_length = ((ts.length << 2) + (1 << core.L_sizeof_datatype) - 1) >> core.L_sizeof_datatype;
        if(ts.mem_size >= compare_length) {//(1) -> (2)
            Syncer sc = core.memset_int32(ts.address, 0, ts.length_4x);
            if(sync) sc.sync(); else ts.setSyncer(sc);
        }
        return ts;
    }
    
    public Tensor zeros_int32_like(Tensor X) { return zeros_int32(X.dim); }
    public Tensor zeros_int32(int... dim) { 
        Tensor ts = new Tensor(check, this, core.dataType_int32(), dim);
        ts.bindMemory(core.malloc_int32(ts.length_4x)); 
        Syncer sc = core.memset_int32(ts.address, 0, ts.length_4x);
        if(sync) sc.sync(); else ts.setSyncer(sc);
        return ts;
    }
    
    public Tensor tensor_int32_like(int[] value, Tensor X) { return tensor_int32(value, X.dim); }
    public Tensor tensor_int32(int[] value, int... dim) {
        dim = negative_dim(value.length, dim);
        Tensor ts = this.empty_int32(dim).c();
        set_int32(ts, value);
        return ts;
    }
    
    public Tensor tensor_int32(int[][] values, int... dim) {
        if(check) { checkMatrix(values, "batch_values"); }//pixels.length = batchSize
        int[] value = toIntVector(values);
        dim = Vector.append(values.length, dim);//[batchsize, ....dim]
        return tensor_int32(value, dim);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Tensor<int32>: memory operations">
    public Tensor zero_int32(Tensor X) {
        if(check) { require_int32(X, "X"); }
        Syncer sc = core.memset_int32(X.address, 0, X.length_4x);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    public int[] valueOf_int32(Tensor X) {
        if(check) { require_int32(X, "X"); }
        if(X.ndim() == 1) { X.c(); return core.get1D_int32(X.address, X.length); }
        int width = X.lastDim(); X.c();
        return (width & 3) == 0 ? 
                core.get1D_int32(X.address, X.length)://no memory alignment
                core.get2D_int32(X.address, X.length / width, width);//[height, width]
    }
    
    @StrictSync
    public Tensor set_int32(Tensor X, int[] value)  {
        if(check) {
            require_int32(X, "X");
            equals(value.length, "value<int>.length", X.length, "X<int32>.length"); 
        }
        
        if(X.ndim() == 1) core.set1D_int32(X.address, value);
        else {
            int width = X.lastDim(); 
            if((width & 3) == 0) core.set1D_int32(X.address, value);//no memory alignment
            else core.set2D_int32(X.address, value, width);//[height, width]
        }
        return X;
    }
    
    public Tensor copy_int32(Tensor X) {
        if(check) { require_int32(X, "X"); }
        Tensor ts = this.empty_int32(X.dim).c();
        Syncer sc = core.memcpy_int32(ts.address, X.address, X.lengthv);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return ts;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="extra: Tensor<int8>"> 
    //<editor-fold defaultstate="collapsed" desc="Tensor<int8>: create">
    public Tensor emtpy_int8_like(Tensor X) { return empty_int8(X.dim); }
    public Tensor empty_int8(int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType_int8(), dim);
        ts.bindMemory(core.malloc_int8(ts.length_4x));

        int compare_length = (ts.length + (1 << core.L_sizeof_datatype) - 1) >> core.L_sizeof_datatype;
        if(ts.mem_size >= compare_length) {//(1) -> (2)
            Syncer sc = core.memset_int8(ts.address, 0, ts.length_4x);
            if(sync) sc.sync(); else ts.setSyncer(sc);
        }
        return ts;
    }
    
    public Tensor zeros_int8_like(Tensor X) { return zeros_int8(X.dim); }
    public Tensor zeros_int8(int... dim) { 
        Tensor ts = new Tensor(check, this, core.dataType_int8(), dim);
        ts.bindMemory(core.malloc_int8(ts.length_4x)); 
        Syncer sc = core.memset_int8(ts.address, 0, ts.length_4x);
        if(sync) sc.sync(); else ts.setSyncer(sc);
        return ts;
    }
    
    public Tensor constants_int8_like(int value, Tensor X) { return constants_int8(value, X.dim);}
    @Passed("CudaFloat32EngieBase")
    public Tensor constants_int8(int value, int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType_int8(), dim);
        ts.bindMemory(core.malloc(ts.length_4x));
        
        if(ts.mem_size > ts.length) {
            core.memset_int8(ts.address, 0, ts.length_4x).sync(); 
        }
        return constant_int8(ts, value);
    }
    
    public Tensor tensor_int8_like(byte[] value, Tensor X) { return tensor_int8(value, X.dim); }
    public Tensor tensor_int8(byte[] value, int... dim) {
        dim = negative_dim(value.length, dim);
        Tensor ts = this.empty_int8(dim).c();
        set_int8(ts, value);
        return ts;
    }
    
    public Tensor tensor_int8(byte[][] values, int... dim) {//pixels.length = batchSize
        if(check) { checkMatrix(values, "batch_values"); }
        byte[] value = toByteVector(values);
        dim = Vector.append(values.length, dim);//[batchsize, ....dim]
        return tensor_int8(value, dim);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Tensor<int32>: memory operations">
    public Tensor zero_int8(Tensor X) {
        if(check) { require_int8(X, "X"); }
        Syncer sc = core.memset_int8(X.address, 0, X.length_4x);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    public Tensor constant_int8(Tensor X, int value) {
        if(check) { require_int8(X); }
        Syncer sc;
        if(X.ndim() == 1) sc = core.set1D_int8(X.address, value, X.length);
        else {
            int width = X.lastDim();
            sc = ((width & 3) == 0?
                    core.set1D_int8(X.address, value, X.length):
                    core.set2D_int8(X.address, value, X.length / width, width));
        }
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    public byte[] valueOf_int8(Tensor X) {
        if(check) { require_int8(X, "X"); }
        if(X.ndim() == 1) { X.c(); return core.get1D_int8(X.address, X.length); }
        int width = X.lastDim(); X.c();
        return (width & 3) == 0 ? 
                core.get1D_int8(X.address, X.length)://no memory alignment
                core.get2D_int8(X.address, X.length / width, width);//[height, width]
    }
    
    @StrictSync
    public Tensor set_int8(Tensor X, byte[] value) {
        if(check) {
            require_int8(X, "X"); 
            equals(value.length, "value<byte>.length", X.length, "X<int8>.length"); 
        }
        
        if(X.ndim() == 1) core.set1D_int8(X.address, value);
        else {
            int width = X.lastDim(); 
            if((width & 3) == 0) core.set1D_int8(X.address, value);//no memory alignment
            else core.set2D_int8(X.address, value, width);//[height, width]
        }
        return X;
    }
    
    public Tensor copy_int8(Tensor X) {
        if(check) { require_int8(X, "X"); }
        Tensor ts = this.empty_int8(X.dim).c();
        Syncer sc = core.memcpy_int8(ts.address, X.address, X.lengthv);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return ts;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor: create & delete">
    //<editor-fold defaultstate="collapsed" desc="auxilary: negative_dim">
    protected int[] negative_dim(int length, int... dim) {//default: firstDim = -1
        for(int i=0; i<dim.length; i++) {
            if(dim[i] == -1) {
                int mul = -Vector.mul(dim);
                if(length % mul != 0) throw new IllegalArgumentException(
                        "Illegal dimension:" + Arrays.toString(dim));
                
                int[] real_dim = Vector.arrayCopy(dim);
                real_dim[i] = length / mul;//correct -1
                return real_dim;
            }
        }
        return dim;//no change
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="auxilary: checkMatrix">
    protected void checkMatrix(byte[][] mat, String name) {
        if(mat == null) throw new NullPointerException();
        if(mat[0] == null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++) {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(String.format(
                    "%s[%d].length { got %d } != width { got %d }", 
                    name, i, mat[i].length, width));
        }
    }
    
    protected void checkMatrix(int[][] mat, String name) {
        if(mat == null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++) {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(String.format(
                    "%s[%d].length { got %d} != width { got %d }", 
                    name, i, mat[i].length, width));
        }
    }
    
    protected void checkMatrix(float[][] mat, String name) {
        if(mat == null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++) {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(String.format(
                    "%s[%d].length { got %d} != width { got %d }", 
                    name, i, mat[i].length, width));
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="fast: Mat to Vector ">
    protected byte[] toByteVector(byte[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        byte[] vec = new byte[dim0 * dim1];
        
        int index = 0;
        for(int i=0; i<dim0; i++) {
            System.arraycopy(mat[i], 0, vec, index, dim1);
            index += dim1;
        }
        return vec;
    }
    
    protected int[] toIntVector(int[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        int[] vec = new int[dim0 * dim1];
        
        int index = 0;
        for(int i=0; i<dim0; i++) {
            System.arraycopy(mat[i], 0, vec, index, dim1);
            index += dim1;
        }
        return vec;
    }
    
    protected float[] toFloatVector(float[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        float[] vec = new float[dim0 * dim1];

        int index = 0;
        for(int i=0; i<dim0; i++) {
            System.arraycopy(mat[i], 0, vec, index, dim1);
            index += dim1;
        }
        return vec;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="create: empty(dim)">
    public Tensor empty_like(Tensor X) { return empty(X.dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor empty(int...dim) {
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x)); 
        
        //memset zero: (1) when tensor.mem_length > tensor.length_4x or (2) ts.isMemAligned.
        if(ts.length_4x > ts.length) {// (1) -> (2) not mem_siize
            Syncer sc = core.memset(ts.address, 0, ts.length_4x);
            if(sync) sc.sync(); else ts.setSyncer(sc);
        }
        return ts;
    }
    
    public Tensor inplace(Tensor X, Tensor Y) {//delete X, let X = Y
        if (check) { this.equals_dataType(X, "X", Y, "Y"); }
        long old_memLen = X.mem_size, old_addr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        core.free(old_memLen, old_addr);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: constant(C, dim)">
    public Tensor zeros_like(Tensor X) { return zeros(X.dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor zeros(int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x)); 
        Syncer sc = core.memset(ts.address, 0, ts.length_4x);
        if(sync) sc.sync(); else ts.setSyncer(sc);
        return ts;
    }

    public Tensor ones_like(Tensor X) { return constants(1.0f, X.dim); }
    public Tensor ones(int... dim) { return constants(1.0f, dim); }
    
    public Tensor constants_like(float value, Tensor X) { return constants(value, X.dim);}
    @Passed("CudaFloat32EngieBase")
    public Tensor constants(float value, int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x));
        
        if(ts.mem_size > ts.length) {
            core.memset(ts.address, 0, ts.length_4x).sync(); 
        }
        return constant(ts, value);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(byte[], dim), byte = int8">
    public Tensor tensor_like(byte[] value, Tensor X) {  return tensor(value, X.dim);  }
    public Tensor tensor_like(float alpha, byte[] value, float beta, Tensor X) {  
        return tensor(alpha, value, beta, X.dim);  
    }
    
    public Tensor tensor(byte[] value, int... dim) { return tensor(1.0f, value, 0.0f, dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float alpha, byte[] value, float beta, int... dim) {
        dim = negative_dim(value.length, dim);
        Tensor ts = this.empty(dim).c();
        set(ts, alpha, value, beta);
        return ts;
    }
    
    public Tensor tensor(byte[][] values, int... dim) { return tensor(1.0f, values, 0.0f, dim); }
    public Tensor tensor(float alpha, byte[][] values, float beta, int... dim) {//pixels.length = batchSize
        if(check) { checkMatrix(values, "batch_values"); }
        byte[] value = toByteVector(values);
        dim = Vector.append(values.length, dim);//[batchsize, ....dim]
        return tensor(alpha, value, beta, dim);
    }
    
    public Tensor onehot(byte[] labels, int num_class) { return onehot(labels, 1.0f, 0.0f, num_class); }
    public Tensor onehot(byte[] labels, float alpha, int num_class)  {
        float beta = (1.0f - alpha) / (num_class - 1);
        return onehot(labels, alpha, beta, num_class);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor onehot(byte[] labels, float alpha, float beta, int num_class)  {
        Tensor BX = this.tensor_int8(labels, labels.length);
        Tensor Y = this.empty(labels.length, num_class).c();
        
        Syncer sc = core.onehot2D_row_int8(Y.address, BX.address, 
                alpha, beta, BX.length, //IX.length = field_length
                Y.lengthv, Y.lastDim());
        sc.sync(); delete(BX);
        return Y;
    }
    
    public Tensor pix_to_tensor(byte[][] pixels, int... dim) {//pixels.length = batchSize
        if(check) { checkMatrix(pixels, "pixels"); }
        byte[] pixel = toByteVector(pixels);
        dim = Vector.append(pixels.length, dim);//[batchsize, ....dim]
        return pixel_to_tensor(pixel, dim);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pixel_to_tensor(byte[] pixel, int...dim) {
        dim = negative_dim(pixel.length, dim);
        Tensor BX = tensor_int8(pixel, dim);
        Tensor Y = this.empty(dim).c();
        
        Syncer sc = core.pix2tensor2D(Y.address, BX.address, 
                Y.lengthv, Y.lastDim());
        sc.sync(); delete(BX);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(int[], dim), byte = int32">
    public Tensor tensor_like(int[] value, Tensor X) {  return tensor(value, X.dim);  }
    public Tensor tensor_like(float alpha, int[] value, float beta, Tensor X) {  
        return tensor(alpha, value, beta, X.dim);  
    }
    
    public Tensor tensor(int[] value, int... dim) { return tensor(1.0f, value, 0.0f, dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float alpha, int[] value, float beta, int... dim) {
        dim = negative_dim(value.length, dim);
        Tensor ts = this.empty(dim).c();
        set(ts, alpha, value, beta);
        return ts;
    }
    
    public Tensor tensor(int[][] values, int... dim) { return tensor(1.0f, values, 0.0f, dim); }
    public Tensor tensor(float alpha, int[][] values, float beta, int... dim) {//pixels.length = batchSize
        if(check) { checkMatrix(values, "batch_values"); }
        int[] value = toIntVector(values);
        dim = Vector.append(values.length, dim);//[batchsize, ....dim]
        return tensor(alpha, value, beta, dim);
    }
    
    public Tensor onehot(int[] labels, int num_class) {
        return onehot(labels, 1.0f, 0.0f, num_class);
    }
    public Tensor onehot(int[] labels, float alpha, int num_class)  {
        float beta = (1.0f - alpha) / (num_class - 1);
        return onehot(labels, alpha, beta, num_class);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor onehot(int[] labels, float alpha, float beta, int num_class) {
        Tensor IX = this.tensor_int32(labels, labels.length);
        Tensor Y = this.empty(labels.length, num_class).c();
        
        Syncer sc = core.onehot2D_row_int32(Y.address, IX.address, 
                alpha, beta, IX.length, //IX.length = field_length
                Y.lengthv, Y.lastDim());
        sc.sync(); IX.delete();
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(float[], dim)">
    public Tensor tensor_like(float[] value, Tensor X) { return tensor(value, X.dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float[] value, int... dim) {
        dim = negative_dim(value.length, dim);
        Tensor ts = this.empty(dim).c();
        set(ts, value);
        return ts;
    }
    
    public Tensor tensor(float[][] values, int... dim) {//pixels.length = batchSize
        if(check) { checkMatrix(values, "batch_values");}
        float[] value = toFloatVector(values);
        dim = Vector.append(values.length, dim);//[batchsize, ....dim]
        return tensor(value, dim);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(Tensor)">
    public Tensor tensor(Tensor value, int... dim) {
        if(check) { require_dtype(value, "value"); }
       
        if(dim == null || dim.length == 0) return copy(value);
        dim = negative_dim(value.length, dim);
        
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x));
        if(ts.mem_size > ts.length) {
            core.memset(ts.address, 0, ts.length_4x).sync(); 
        }
        return set(ts, value);
    }

    public Tensor copy(Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor ts = this.empty(X.dim).c();
        Syncer sc = core.memcpy(ts.address, X.address, X.lengthv);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return ts;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="delete Tensor">
    //<editor-fold defaultstate="collaped" desc="delete-core">
    protected void delete_core(Tensor X) {
        //must wait until the compute of the tensor is completed
        //if you delete a tensor participating in a computation, it may effects other tensor
        core.free(X.c().mem_size, X.address);//release the memory of the tensor
        X.address = 0L;
        X.trace = null;
        X.mod_counter = null;
        
        Tensor grad = null, view = null, root = null;
        TensorSet carrier = null;
        synchronized (X) {
            if (X.grad != null) { grad = X.grad; X.grad = null; }
            if (X.view != null) { view = X.view; X.view = null; }
            if (X.root != null) { root = X.root; X.root = null; }
            if (X.carrier != null) { carrier = X.carrier; X.carrier = null; }
        }
        if (grad != null) delete(grad);
        if (view != null) delete(view);
        if (root != null) delete(root);
        if (carrier != null) { delete(carrier); carrier.clear(); }
    }
    //</editor-fold>
    public boolean delete(Tensor X)  {
        if (X == null) return false;
        if (X.eg != this) { return X.eg.delete(X); }
        delete_core(X);
        return true;
    }
    
    public void delete(Tensor... Xs) {
        if(Xs == null || Xs.length == 0) return;
        for(Tensor ts : Xs) delete(ts); 
    }
    
    public void delete(Collection<Tensor> Xs) { 
        if(Xs == null || Xs.isEmpty()) return;
        Xs.forEach((ts) -> { delete(ts); }); 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="delete Parameter">
    public void delete(Parameter... params) {
        if(params == null || params.length == 0) return;
        for(Parameter ts : params) delete(ts); 
    }
    
    public void delete(Parameter param)  {
        if(param == null) return;
        delete(param.tensor);
        if(!param.grads.isEmpty()) {//param.clear_grads()
            param.grads.forEach((Tensor g) -> { delete(g); });
            param.grads.clear();
        }
    }    
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor: set">
    //<editor-fold defaultstate="collapsed" desc="constant(C)">
    public Tensor zero(Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Syncer sc = core.memset(X.address, 0, X.length_4x);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    public Tensor constant(Tensor X, float value) {//constant value setter
        if(check) { require_dtype(X, "X"); }
        Syncer sc;
        if(X.ndim() == 1) sc = core.set1D(X.address, value, X.length);
        else {
            int width = X.lastDim();
            sc = ((width & 3) == 0?
                    core.set1D(X.address, value, X.length):
                    core.set2D(X.address, value, X.length / width, width));
        }
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(byte[]), byte=int8">
    @StrictSync
    public Tensor set(Tensor X, byte[] value) { return set(X, 1.0f, value, 0.0f); }
    public Tensor set(Tensor X, float alpha, byte[] value, float beta) {
        if(check) { 
            require_dtype(X, "X");
            equals(value.length, "value<byte>.length", X.length, "X.length"); 
        } 
        Tensor BX = this.tensor_int8(value, X.dim).c();
        Syncer sc = core.linear2D_int8_to_dtype(X.address, 
                alpha, BX.address, beta,
                X.lengthv, X.lastDim()); 
        sc.sync(); delete(BX);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(int[])", int=int32">
    @StrictSync
    public Tensor set(Tensor X, int[] value) { return set(X, 1.0f, value, 0.0f); }
    public Tensor set(Tensor X, float alpha, int[] value, float beta) {
        if(check) {
            require_dtype(X, "X");
            equals(value.length, "value<int>.length", X.length, "X.length");
        } 
        Tensor IX = this.tensor_int32(value, X.dim).c();
        Syncer sc = core.linear2D_int32_to_dtype(X.address, 
                alpha, IX.address, beta, 
                X.lengthv, X.lastDim());
        sc.sync(); delete(IX);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(float[])">
    @StrictSync
    public Tensor set(Tensor X, float[] value) {
        if(check) { 
            require_dtype(X, "X");
            equals(value.length, "value<float>.length", X.length, "X.length"); 
        } 
        
        if(X.ndim() == 1) core.set1D(X.address, value);
        else {
            int width = X.lastDim();
            if((width & 3) == 0) core.set1D(X.address, value);//no memory alignment
            else core.set2D(X.address, value, X.lastDim());
        }
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(Tensor)">
    public Tensor set(Tensor X, Tensor value) {
        if(X.address == value.address) return X;
        if(check) {
            require_dtype(X, "X"); require_dtype(value, "value");
            equals(X.length, "X.length", value.length, "value.length");
        }
        
        Syncer sc;//ts.length != value.length
        if(X.valueStrucEquals(value)) {//directly copy, All at once
            sc = core.memcpy(X.address, value.address, X.lengthv);
            if(sync) sc.sync(); else X.setSyncer(sc);
            return X;
        }
        
        int value_ndim = value.ndim(), ts_ndim = X.ndim();
        if(value_ndim > 1 && ts_ndim > 1) {//ND to ND, no need to memset(0)
            int src_width = value.lastDim(), dst_width = X.lastDim();
            sc = core.setFrom2Dto2D(
                    value.address, value.length / src_width, src_width,
                    X.address    , X.length     / dst_width, dst_width);
        }
        else if(value_ndim == 1) {//1D to ND
            int dst_width = X.lastDim();
            sc = core.setFrom1Dto2D(
                    value.address, value.length,
                    X.address, X.length / dst_width, dst_width);
        }
        else{//X.ndim == 1, ND to 1D
            int src_width = value.lastDim();
            sc = core.setFrom2Dto1D(
                    value.address, value.length / src_width, src_width,
                    X.address, X.length);
        }
        
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(others)">
    @StrictSync
    public Tensor set(Tensor X, String line) {
        float[] value; try { value = Vector.to_float_vector(line); }
        catch(Exception e) { throw new RuntimeException(e); }
        return set(X, value);
    }
    
    public static boolean SET_STRING_LINES_PRINT = false;
    
    @StrictSync
    public Tensor set(Tensor X, List<String> lines) {
        float[] value = Vector.to_float_vector(lines, X.length);
        if(SET_STRING_LINES_PRINT) {//print if need to check
            float sp = Vector.samePercent_absolute(value, X.value());
            System.out.println("sp = " + sp);
        }
        return set(X, value);
    }
    
    public void set(Tensor X, StateValue value, boolean partial, String msg) {
        if(value == null && !partial) throw new RuntimeException(msg);
        if(value != null) try { set(X, value.toStringLines()); }
        catch(Exception e) {
            throw new RuntimeException(msg, e);
        }
    }
    //</editor-fold>
    //</editor-fold>        
    
    //<editor-fold defaultstate="collapsed" desc="Tensor: dataType && valueOf">
    @StrictSync
    public float[] valueOf(Tensor ts) {
        if(ts.ndim() == 1) { ts.c(); return core.get1D(ts.address, ts.length); }
        int width = ts.lastDim(); ts.c();
        return (width & 3) == 0 ? 
                core.get1D(ts.address, ts.length)://no memory alignment
                core.get2D(ts.address, ts.length / width, width);//[height, width]
    }
    
    public boolean is_dtype(Tensor X) { return X.dataType.equals(dataType()); }
    public boolean is_int32(Tensor X) { return X.dataType.equals(dataType_int32()); }
    public boolean is_int8(Tensor X) { return X.dataType.equals(dataType_int8()); }
    
    public <T> T raw_data(Tensor X) {
        if(is_dtype(X)) return (T) valueOf(X);
        if(is_int32(X)) return (T) valueOf_int32(X);
        if(is_int8(X)) return (T) valueOf_int8(X);
        throw new RuntimeException("Unknown dataType");
    }
    
    public <T> T data(Tensor X) {
        int ndim = X.ndim(), dim[] = X.dim;
        if(is_dtype(X)) {
            float[] value = valueOf(X);
            if(ndim == 1) return (T) value;
            if(ndim == 2) return (T) Vector.to2D(value, dim[0], dim[1]);
            if(ndim == 3) return (T) Vector.to3D(value, dim[0], dim[1], dim[2]);
            else return (T) Vector.to4D(value, dim[0], dim[1], dim[2], dim[3]);
        }
        if(is_int32(X)) {
            int[] value = valueOf_int32(X);
            if(ndim == 1) return (T) value;
            if(ndim == 2) return (T) Vector.to2D(value, dim[0], dim[1]);
            if(ndim == 3) return (T) Vector.to3D(value, dim[0], dim[1], dim[2]);
            else return (T) Vector.to4D(value, dim[0], dim[1], dim[2], dim[3]);
        }
        if(is_int8(X)) {
            byte[] value = valueOf_int8(X);
            if(ndim == 1) return (T) value;
            if(ndim == 2) return (T) Vector.to2D(value, dim[0], dim[1]);
            if(ndim == 3) return (T) Vector.to3D(value, dim[0], dim[1], dim[2]);
            else return (T) Vector.to4D(value, dim[0], dim[1], dim[2], dim[3]);
        }
        throw new RuntimeException("Unknown dataType");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //<editor-fold defaultstate="collapsed" desc="Normal_MatMul(2D, 2D)">
    //<editor-fold defaultstate="collapsed" desc="matMul: C = A * B">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul(Tensor A, Tensor B) {
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B"); 
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(A.dim(1), "A.width", B.dim(0), "B.height");
        }
        
        int dimA[] = A.dim, AH = dimA[0], AW = A.dim(1);//A[N, K]
        int dimB[] = B.dim, BW = dimB[1];//B[K, M]
        Tensor C = this.empty(AH, BW).c();
        
        Syncer sc = core.matMul(C.address, A.address, B.address, AH, BW, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
 
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul(Tensor C, Tensor A, Tensor B) {
        if(check) {
            require_dtype(C, "C"); require_dtype(A, "A"); require_dtype(B, "B"); 
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(C.ndim(), "C.ndim", 2);
            equals(C.dim(0), "C.height", A.dim(0), "A.height");//N = CH = AH
            equals(C.dim(1), "C.width",  B.dim(1), "B.width"); //M = CW = BW
            equals(A.dim(1), "A.width",  B.dim(0), "B.height");//K = AW = BH
        }
        
        int dimA[] = A.dim, AH = dimA[0], AW = A.dim(1);//A[N, K]
        int dimB[] = B.dim, BW = dimB[1];//B[K, M]
        
        Syncer sc = core.matMul(C.address, A.address, B.address, 
                AH, BW, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMul: C = A * B + Bias">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul_biased(Tensor A, Tensor B, Tensor Bias) {
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B"); require_dtype(Bias, "Bias"); 
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(A.dim(1), "A.width", B.dim(0), "B.height");//K = AW = BH
            equals(Bias.lastDim(), "Bias.lastDim", B.dim(1), "B.width");
        }
        
        int dimA[] = A.dim, AH = dimA[0], AW = A.dim(1);//A[N, K]
        int dimB[] = B.dim, BW = dimB[1];//B[K, M]
        Tensor C = this.empty(AH, BW).c();
        
        Syncer sc = core.matMul_biased(C.address, A.address, B.address, AH, BW, AW,
                Bias.address, C.lengthv);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul_biased(Tensor C, Tensor A, Tensor B, Tensor Bias) {
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B"); 
            require_dtype(C, "C"); require_dtype(Bias, "Bias");
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(C.ndim(), "C.ndim", 2);
            equals(C.dim(0), "C.height", A.dim(0), "A.height");
            equals(C.dim(1), "C.width",  B.dim(1), "B.width");
            equals(A.dim(1), "A.width",  B.dim(0), "B.height");
            equals(Bias.lastDim(), "Bias.lastDim", B.dim(1), "B.width");
        }
        
        int dimA[] = A.dim, AH = dimA[0], AW = A.dim(1);//A[N, K]
        int dimB[] = B.dim, BW = dimB[1];//B[K, M]
        
        Syncer sc = core.matMul_biased(C.address, A.address, B.address, AH, BW, AW,
                Bias.address, C.lengthv);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT1: C = (A^T) * B">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT1(Tensor A, Tensor B) {
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B");
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(A.dim(0), "A.height", B.dim(0), "B.height");//K = AH = BH
        }
        
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];//A[K, N]
        int dimB[] = B.dim, BW = dimB[1];//B[K, M]
        Tensor C = this.empty(AW, BW).c();
        
        Syncer sc = core.matMulT1(C.address, A.address, B.address, AW, BW, AH);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT1(Tensor C, Tensor A, Tensor B){
        if(check) {
            require_dtype(C, "C"); require_dtype(A, "A"); require_dtype(B, "B");
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(C.ndim(), "C.ndim", 2);
            equals(C.dim(0), "C.height", A.dim(1), "A.width"); //N = CH = AW
            equals(C.dim(1), "C.width",  B.dim(1), "B.width"); //M = CW = BW
            equals(A.dim(0), "A.height", B.dim(0), "B.height");//K = AH = BW
        }
          
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];//A[K, N]
        int dimB[] = B.dim, BW = dimB[1];//B[K, M]
        
        Syncer sc = core.matMulT1(C.address, A.address, B.address, AW, BW, AH);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT2: C = A * (B^T)">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT2(Tensor A, Tensor B) {
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B");
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(A.dim(1), "A.width", B.dim(1), "B.width");//K = AW = BW
        }
        
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];//A[N, K]
        int dimB[] = B.dim, BH = dimB[0];//B[M, K]
        Tensor C = this.empty(AH, BH).c();
        
        Syncer sc = core.matMulT2(C.address, A.address, B.address, AH, BH, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT2(Tensor C, Tensor A, Tensor B) {
        if(check) {
            require_dtype(C); require_dtype(A); require_dtype(B);
            equals(A.ndim(), "A.ndim", 2);
            equals(B.ndim(), "B.ndim", 2);
            equals(C.ndim(), "C.ndim", 2);
            equals(C.dim(0), "C.height", A.dim(0), "A.height");//N = CH = AH
            equals(C.dim(1), "C.width",  B.dim(0), "B.height");//M = CW = BH
            equals(A.dim(1), "A.width",  B.dim(1), "B.width"); //K = AW = BW
        }

        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];//A[N, K]
        int dimB[] = B.dim, BH = dimB[0];//B[M, K]
        
        Syncer sc = core.matMulT2(C.address, A.address, B.address, AH, BH, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="Batch_MatMul(3D+, 3D+)">
    //<editor-fold defaultstate="collapsed" desc="batchMatMul">
    public Tensor batchMatMul(Tensor A, Tensor B) { return batchMatMul(true, A, B); }
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMul(boolean likeA, Tensor A, Tensor B) {
        int ndimA = A.ndim(), ndimB = B.ndim(); 
        if(ndimA == 2 && ndimB == 2) return matMul(A, B);
        
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(ndimA, "A.ndim", 3);
            must_greater_equal(ndimB, "B.ndim", 3);
            equals(A.dim(-1), "A.width", B.dim(-2), "B.height");//K = AW = BH
            equals(A.length / (A.dim(-2) * A.dim(-1)), "A.batch", 
                   B.length / (B.dim(-2) * B.dim(-1)), "B.batch");
        }
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
    
        int[] dimC;//A[N: AH, K: AW] * B[K: BH, M: BW] -> C[N: AH, M: BW]
        if(likeA) {//likeA
            dimC = new int[ndimA]; dimC[ndimA - 2] = AH; dimC[ndimA - 1] = BW;
            for(int i=0; i<ndimA - 2; i++) dimC[i] = dimA[i];
        }
        else {//like B
            dimC = new int[ndimB]; dimC[ndimB - 2] = AH; dimC[ndimB - 1] = BW;
            for(int i=0; i<ndimB - 2; i++) dimC[i] = dimB[i];
        }

        Tensor C = this.empty(dimC).c();
        Syncer sc = core.batchMatMul(C.address, A.address, B.address, batchA, AH, BW, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMul(Tensor C, Tensor A, Tensor B)  {
        int ndimA = A.ndim(), ndimB = B.ndim(), ndimC = C.ndim();
        if(ndimA == 2 && ndimB == 2 && ndimC == 2) return matMul(C, A, B);
        
        if(check) {
            require_dtype(C, "C"); require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(ndimA, "A.ndim", 3);
            must_greater_equal(ndimB, "B.ndim", 3);
            must_greater_equal(ndimC, "C.ndim", 3);
            equals(C.dim(-2), "C.height", A.dim(-2), "A.height");//N = CH = AH
            equals(C.dim(-1), "C.width",  B.dim(-1), "B.width"); //M = CW = BW
            equals(A.dim(-1), "A.width",  B.dim(-2), "B.height");//K = AW = BH
            int batchA = A.length / (A.dim(-2) * A.dim(-1));
            equals(batchA, "A.batch", B.length / (B.dim(-2) * B.dim(-1)), "B.batch");
            equals(batchA, "A.batch", C.length / (C.dim(-2) * C.dim(-1)), "C.batch");
        }
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        //A[N: AH, K: AW] * B[K: BH, M: BW] -> C[N: AH, M: BW]
        Syncer sc = core.batchMatMul(C.address, A.address, B.address, batchA, AH, BW, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT1">
    public Tensor batchMatMulT1(Tensor A, Tensor B) { return batchMatMulT1(true, A, B); }
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT1(boolean likeA, Tensor A, Tensor B) {
        int ndimA = A.ndim(), ndimB = B.ndim(); 
        if(ndimA == 2 && ndimB == 2) return matMulT1(A, B);
        
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(ndimA, "A.ndim", 3);
            must_greater_equal(ndimB, "B.ndim", 3);
            equals(A.dim(-2), "A.height", B.dim(-2), "B.height");//K = AH = BH
            equals(A.length / (A.dim(-2) * A.dim(-1)), "A.batch", 
                   B.length / (B.dim(-2) * B.dim(-1)), "B.batch");
        }
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        int[] dimC;//A^T[N: AW, K: AH] * B[K: BH, M: BW] -> C[N: AW, M: BW]
        if(likeA) {//likeA
            dimC = new int[ndimA]; dimC[ndimA - 2] = AW; dimC[ndimA - 1] = BW;
            for(int i=0; i<ndimA - 2; i++) dimC[i] = dimA[i];
        }
        else {//like B
            dimC = new int[ndimB]; dimC[ndimB - 2] = AW; dimC[ndimB - 1] = BW;
            for(int i=0; i<ndimB - 2; i++) dimC[i] = dimB[i];
        }
        
        Tensor C = this.empty(dimC).c();
        Syncer sc = core.batchMatMulT1(C.address, A.address, B.address, batchA, AW, BW, AH);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT1(Tensor C, Tensor A, Tensor B) {
        int ndimA = A.ndim(), ndimB = B.ndim(), ndimC = C.ndim();
        if(ndimA == 2 && ndimB == 2 && ndimC == 2) return matMulT1(C, A, B);

        if(check) {
            require_dtype(C); require_dtype(A); require_dtype(B);
            must_greater_equal(ndimA, "A.ndim", 3);
            must_greater_equal(ndimB, "B.ndim", 3);
            must_greater_equal(ndimC, "C.ndim", 3);
            equals(C.dim(-2), "C.height", A.dim(-1), "A.width"); //N = CH = AH
            equals(C.dim(-1), "C,width",  B.dim(-1), "B.wdith"); //M = CW = BW
            equals(A.dim(-2), "A.height", B.dim(-2), "B.height");//K = AW = BH
            int batchA = A.length / (A.dim(-1) * A.dim(-2));
            equals(batchA, "A.batch", B.length / (B.dim(-2) * B.dim(-1)), "B.batch");
            equals(batchA, "A.batch", C.length / (C.dim(-2) * C.dim(-1)), "C.batch");
        }
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        //A^T[N: AW, K: AH] * B[K: BH, M: BW] -> C[N: AW, M: BW]
        Syncer sc = core.batchMatMulT1(C.address, A.address, B.address, batchA, AW, BW, AH);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT2">
    public Tensor batchMatMulT2(Tensor A, Tensor B) { return batchMatMulT2(true, A, B); }
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT2(boolean likedA, Tensor A, Tensor B) {
        int ndimA = A.ndim(), ndimB = B.ndim();
        if(ndimA == 2 && ndimB == 2) return matMulT2(A, B);
        
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(ndimA, "A.ndim", 3);
            must_greater_equal(ndimB, "B.ndim", 3);
            equals(A.dim(-1), "A.width", B.dim(-1), "B.width");//K = AW = BW
            equals(A.length / (A.dim(-2) * A.dim(-1)), "A.batch",
                   B.length / (B.dim(-2) * B.dim(-1)), "B.batch");
        }
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BH = dimB[ndimB - 2];
        int batchA = A.length / (AH * AW);
        
        int[] dimC;//A[N: AH, K: AW] * B^T[K: BW, M: BH] -> C[N: AH, M: BH]
        if(likedA) {//likeA
            dimC = new int[ndimA]; dimC[ndimA - 2] = AH; dimC[ndimA - 1] = BH;
            for(int i=0; i<ndimA - 2; i++) dimC[i] = dimA[i];
        }
        else {//likedB
            dimC = new int[ndimB]; dimC[ndimB - 2] = AH; dimC[ndimB - 1] = BH;
            for(int i=0; i<ndimB - 2; i++) dimC[i] = dimB[i];
        }
        
        Tensor C = this.empty(dimC).c();
        Syncer sc = core.batchMatMulT2(C.address, A.address, B.address, batchA, AH, BH, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT2(Tensor C, Tensor A, Tensor B) {
        int ndimA = A.ndim(), ndimB = B.ndim(), ndimC = C.ndim();
        if(ndimA == 2 && ndimB == 2 && ndimC == 2) return matMulT2(C, A, B);
        
        if(check) {
            require_dtype(A, "A"); require_dtype(B, "B"); require_dtype(C, "C");
            must_greater_equal(ndimA, "A.ndim", 3);
            must_greater_equal(ndimB, "B.ndim", 3);
            must_greater_equal(ndimC, "C.ndim", 3);
            equals(C.dim(-2), "C.height", A.dim(-2), "A.height");//N = CH = AH
            equals(C.dim(-1), "C.width",  B.dim(-2), "B.height");//M = CW = BH
            equals(A.dim(-1), "A.width",  B.dim(-1), "B.width"); //K = AW = BW
            int batchA = A.length / (A.dim(-2) * A.dim(-1));
            equals(batchA, "A.batch", B.length / (B.dim(-2) * B.dim(-1)), "B.batch");
            equals(batchA, "A.batch", C.length / (C.dim(-2) * C.dim(-1)), "C.batch");
        }
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BH = dimB[ndimB - 2];
        int batchA = A.length / (AH * AW);
        
        //A[N: AH, K: AW] * B^T[K: BW, M: BH] -> C[N: AH, M: BH]
        Syncer sc = core.batchMatMulT2(C.address, A.address, B.address, batchA, AH, BH, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="fullconnect(3D+, 2D)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation: Y = X * W">
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect(Tensor X, Tensor W) {
        int ndimX = X.ndim(); if(ndimX == 2) return matMul(X, W);
        if(check) {
            require_dtype(X, "X"); require_dtype(W, "W");
            must_greater_equal(ndimX, "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 2);
            equals(X.dim(-1), "X.features", W.dim(0), "W.in_features");
        }

        int dimW[] = W.dim, WOW = dimW[1];//W[IW, OW]
        int dimX[] = X.dim, XIW = dimX[ndimX - 1];//X[N, H, IW]
        int GN = X.length / XIW;//GN = X.N * X.H
        
        int[] dimY = new int[ndimX]; dimY[ndimX - 1] = WOW;
        for(int i=0; i<ndimX - 1; i++) dimY[i] = dimX[i];
        Tensor Y = this.empty(dimY).c();
        
        //reshape: [batch * H, IW] * [IW, OW] -> [batch * H, OW]
        Syncer sc = core.matMul(Y.address, X.address, W.address, GN, WOW, XIW);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect(Tensor Y, Tensor X, Tensor W) {
        int ndimX = X.ndim(), ndimY = Y.ndim();
        if(ndimX == 2 && ndimY == 2) return matMul(Y, X, W);
        
        if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W");
            must_greater_equal(ndimX, "Y.ndim", 2);
            must_greater_equal(ndimX, "X.ndim", 2);
            equals(W.ndim(), "W.ndim", 2);
            equals(X.dim(-1), "X.features", W.dim(0), "W.in_features");
            equals(Y.dim(-1), "Y.features", W.dim(1), "W.out_features");
            equals(X.length / X.dim(-1), "(X.batch * X.height)", 
                   Y.length / Y.dim(-1), "(Y.batch * Y.height)");
        }

        int WOW = W.dim[1], XIW = X.dim[ndimX - 1];
        int GN = X.length / XIW;//GN = X.N * X.H
        
        //reshape: [batch * H, IW] * [IW, OW] -> [batch * H, OW]
        Syncer sc = core.matMul(Y.address, X.address, W.address, GN, WOW, XIW);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward-propagation: Y = X * W + Bias">
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect_biased(Tensor X, Tensor W, Tensor Bias) {
        int ndimX = X.ndim(); if(ndimX == 2) return matMul_biased(X, W, Bias); 
        if(check) {
            require_dtype(X, "X"); require_dtype(W, "W"); require_dtype(Bias, "Bias");
            must_greater_equal(ndimX, "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 2);
            equals(X.dim(-1), "X.features", W.dim(0), "W.in_features");//K = XW = WH
            equals(Bias.length, "Bias.length", W.dim(1), "W.out_features");
        }
        
        int dimW[] = W.dim, WOW = dimW[1];//W[IW, OW]
        int dimX[] = X.dim, XIW = dimX[ndimX - 1];//X[N, H, IW]
        int GN = X.length / XIW;//GN = X.N * X.H
        
        int[] dimY = new int[ndimX];  dimY[ndimX - 1] = WOW;
        for(int i=0; i<ndimX - 1; i++) dimY[i] = dimX[i];
        Tensor Y = this.empty(dimY).c();
        
        //reshape: [batch * H, IW] * [IW, OW] -> [batch * H, OW]
        Syncer sc = core.matMul_biased(Y.address, X.address, W.address, GN, WOW, XIW, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect_biased(Tensor Y, Tensor X, Tensor W, Tensor Bias) {
        int ndimX = X.ndim(), ndimY = Y.ndim();
        if(ndimX == 2 && ndimY == 2) return matMul_biased(Y, X, W, Bias); 
        
        if(check) {
            require_dtype(X, "X"); require_dtype(W, "W");
            require_dtype(Y, "Y"); require_dtype(Bias, "Bias");
            must_greater_equal(ndimX, "Y.ndim", 2);
            must_greater_equal(ndimX, "X.ndim", 2);
            equals(W.ndim(), "W.ndim", 2);
            equals(X.dim(-1), "X.features", W.dim(0), "W.in_features");
            equals(Y.dim(-1), "Y.features", W.dim(1), "W.out_features");
            equals(X.length / X.dim(-1), "(X.batch * X.height)",
                   Y.length / Y.dim(-1), "(Y.batch * Y.height)");
            equals(Bias.length, "Bias.length", W.dim(1), "W.out_features");
        }
        
        int WOW = W.dim[1], XIW = X.dim[ndimX - 1];
        int GN = X.length / XIW;//GN = X.N * X.H
        
        //reshape: [batch * H, IW] * [IW, OW] -> [batch * H, OW]
        Syncer sc = core.matMul_biased(Y.address, X.address, W.address, GN, WOW, XIW, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect_deltaX(Tensor deltaY, Tensor W) {
        int ndimY = deltaY.ndim(); if(ndimY == 2) return matMulT2(deltaY, W);
        if(check) {
            require_dtype(deltaY, "deltaY");  require_dtype(W, "W");
            must_greater_equal(ndimY, "deltaY.ndim", 3);
            equals(W.ndim(), "W.ndim", 2);
            equals(deltaY.dim(-1), "deltaY.features", W.dim(1), "W.out_features");
        }
        
        int dimW[] = W.dim, WIW = dimW[0];//W: [IW, OW]
        int dimY[] = deltaY.dim, YOW = dimY[ndimY - 1];//Y: [N, H, OW]
        int GN = deltaY.length / YOW;//GN = Y.N * Y.H
        
        int[] dimX = new int[ndimY]; dimX[ndimY - 1] = WIW;
        for(int i=0; i<ndimY - 1; i++) dimX[i] = dimY[i];
        Tensor deltaX = this.empty(dimX).c();
        
        //reshape: [batch * H, OW] * [OW, IW] = [batch * H, IW]: deltaX = deltaY * W^T
        Syncer sc = core.matMulT2(deltaX.address, deltaY.address, W.address, GN, WIW, YOW);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect_deltaW(Tensor X, Tensor deltaY) {
        int ndimY = deltaY.ndim(), ndimX = X.ndim();
        if(ndimY == 2 && ndimX == 2) return matMulT1(X, deltaY);
        
        if(check) {
            require_dtype(X, "X"); require_dtype(deltaY, "deltaY");
            must_greater_equal(ndimX, "X.ndim", 3);
            must_greater_equal(ndimY, "deltaY.ndim", 3);
            equals(X.length / X.dim(-1), "(X.batch * X.height)",
                   deltaY.length / deltaY.dim(-1), "(deltaY.batch * Y.height)"); 
        }
        
        int dimX[] = X.dim, XIW = dimX[ndimX - 1];//X[N, H, IW]
        int dimY[] = deltaY.dim, YOW = dimY[ndimY - 1];//Y[N, H, OW]
        int GK = deltaY.length / YOW;//GK = Y.N * Y.H
        Tensor deltaW = this.empty(XIW, YOW).c();
         
        //reshape: [K, batch * N] * [batch * N, M] = [K, M]: deltaW =  X^T * deltaY
        Syncer sc = core.matMulT1(deltaW.address,  X.address, deltaY.address, XIW, YOW, GK);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    //<editor-fold defaultstate="collapsed" desc="reshape && view: X[oldDim] -> X'[newDim]">
    public Tensor view_copy(Tensor X) { return view(false, X, X.dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor view(boolean inplace, Tensor X, int... dim) {
        dim = (dim == null || dim.length == 0 ?
                new int[] { X.dim(0), X.length() / X.dim(0) } ://flatten
                negative_dim(X.length, dim));//newDim.length == oldDim.length
        
        if(!X.memStrucEquals(X.length, dim)) 
            throw new IllegalArgumentException("the old MemStructure is different from the New one");
        
        if(inplace) { X.setDim(check, dim); return X; }
        
        Tensor Y = new Tensor(check, this, core.dataType(), dim);
        Y.copy_memoryMetaData(X);
        Y.syncer = X.syncer;//inherent syncer
        Y.mod_counter = X.mod_counter;//inherent mod-count
        Y.need_carry = X.need_carry; Y.carrier = X.carrier;//inherent carrier
        
        X.view = Y; Y.root = X;
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor reshape(boolean inplace, Tensor X, int...dim) {
        dim = (dim == null || dim.length == 0? 
                new int[] { X.dim(0), X.length() / X.dim(0) } ://flatten
                negative_dim(X.length, dim));//newDim.length == oldDim.length
        
        if(!inplace) return tensor(X, dim);
        
        //inplace = true, use the old memory, no need to memcpy
        if(X.memStrucEquals(X.length, dim)) { X.setDim(check, dim); return X; } 
        
        //inplace = true, but need to reorganize the mem structure
        Tensor Y = tensor(X, dim);//Y[newDim, newAddress]
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim, X.address = Y.address
        
        Syncer sc = Syncer.dual(Y.syncer, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="concat: X[] -> Y">
    //<editor-fold defaultstate="collapsed" desc="auxilary: exclude_null_tensors">
    Tensor[] exclude_null_tensors(Tensor... Xs) {
        int count = 0; for (Tensor X : Xs) if (X != null) count++;
        if(count == Xs.length) return Xs;
        
        Tensor[] nX = new Tensor[count]; int index = 0;
        for(Tensor X : Xs) if(X != null) nX[index++] = X;
        return nX;
    }
    //</editor-fold>
    
    public Tensor concat(Tensor... X) { return concat(-1, X); }
    @Passed("CudaFloat32EngieBase")
    public Tensor concat(int dimIdx, Tensor... X)  {
        X = exclude_null_tensors(X);
        final int ndim = X[0].ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int[][] dimX = new int[X.length][];
        for(int i=0; i<X.length; i++) dimX[i] = X[i].dim;
        
        if(check) {
            require_dtype(X);
            if(X.length < 2) throw new IllegalArgumentException("At least input two tensors to concat");
            for(int i=0; i<dimX.length; i++) 
                if(dimX[i].length <= dimIdx) throw new IllegalArgumentException(
                    "dimIndex exceeds X[" + i + "].ndim");
            for(int i=0; i<dimX.length; i++) 
                if(dimX[i].length != ndim) throw new IllegalArgumentException(String.format(
                        "X[%d].ndim(%d) ! = X[0].ndim(%d)", i, dimX[i].length, ndim));
            for(int i=1; i<X.length; i++) 
                for(int j=0; j<ndim; j++) {
                    if(j == dimIdx) continue;//only dimIndex is different
                    if(dimX[0][j] != dimX[i][j]) throw new IllegalArgumentException(
                            String.format("X[%d].dim[%d](%d) != X[0].dim[%d](%d)",i, j, dimX[i][j], j, dimX[0][j]));
                }
        }
        
        //compute the dim of output tensor--------------------------------------
        int[] dimY = Vector.arrayCopy(dimX[0]);//concat_dim = sum(X[i].dim(dimIndex), 0, n-1) 
        for(int i=1; i<X.length; i++) dimY[dimIdx] += dimX[i][dimIdx];//dimIndex: the concat dim
        Tensor Y = this.empty(dimY);
        
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize mul: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[0][i];
        
        int[] copyWidth = new int[X.length];
        int[] strideX = new int[X.length];
        for(int i=0; i<X.length; i++) {
            copyWidth[i] = commonWidth * dimX[i][dimIdx];//from dimIndex to End
            
            int width = X[i].lastDim();//consider memAlignment: 
            int stride = ((width + 3) >> 2) << 2;
            strideX[i] = copyWidth[i] / width * stride;
            if(dimIdx != ndim - 1) copyWidth[i] = strideX[i];
        } 
        
        int strideY = commonWidth * dimY[dimIdx]; {//from dimIndex to End
            int width = Y.lastDim();//consider mem alignment
            int stride = ((width + 3) >> 2) << 2;
            strideY = strideY / width * stride;
        }
                
        Syncer[] sc = new Syncer[X.length]; Y.c();//Y is synchronized 
        for(int i=0, Ystart = 0; i<X.length; Ystart += copyWidth[i++]) {
            int length = (dimIdx == ndim - 1 ? X[i].length : X[i].lengthv);
            sc[i] = core.gappedMemcpy2D(
                    X[i].address, 0, strideX[i], 
                    Y.address, Ystart, strideY, 
                    copyWidth[i], length);
        }
        
        if(sync) { for(Syncer syncer : sc) syncer.sync(); }
        else Y.setSyncer(new ChainSyncer(sc));
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="split & chunk: X -> Y[]">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] chunk(Tensor X, int dimIdx, int n) {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int dimX[] = X.dim, dimSize = dimX[dimIdx]; //dimSize = dimX[dimIndex] = sum(section)
        if(check) {
            require_dtype(X, "X"); must_greater_equal(n, "n", 2);
            if(n > dimSize) throw new IllegalArgumentException(String.format(
                    "n { got %d }> X.dim[%d] { got %d }", n, dimIdx, dimSize));
        }
        
        int[] section = new int[n];
        int div = dimSize  / n, rm = dimSize % n;
        for(int i=0; i<n; i++) section[i] = div;
        section[n - 1] += rm;
        
        return __split(X, dimIdx, section);
    }
    
    //<editor-fold defaultstate="collapsed" desc="auxilary: negativeSection">
    protected int negativeSection(int dimSize, int...section) {
        int index = -1;
        for(int i=0; i<section.length; i++)  
            if(section[i] == -1) { index = i; break; }
        
        int sectionSum = Vector.sum(section);
        if(index != -1) // -1 -> 0;
            section[index] = dimSize - (sectionSum + 1);
        return sectionSum;
    }
    //</editor-fold>
    @Passed("CudaFloat32EngieBase") 
    public Tensor[] split(Tensor X, int dimIdx, int...section) {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        int dimX[] = X.dim, dimSize = dimX[dimIdx]; //dimSize = dimX[dimIndex] = sum(section)
        int sectionSum = negativeSection(dimSize, section);//exclude -1 in section
         
        if(check) {
            require_dtype(X, "X");
            must_greater_equal(section.length, "section.length", 2);
            must_positive(section, "section");
            if(sectionSum != dimSize) throw new IllegalArgumentException(String.format(
                    "sum(section) { got %d } != X.dim[%d] { got %d }", 
                    sectionSum, dimIdx, dimSize));
        }
        
        return __split(X, dimIdx, section);
    }
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: __split">
    private Tensor[] __split(Tensor X, int dimIdx, int[] section) {
        //create sub Tensor[] Y based on section--------------------------------
        int dimX[] = X.dim, ndim = X.ndim();
        
        Tensor[] Y = new Tensor[section.length];
        for(int i=0, dimY[] = Vector.arrayCopy(dimX); i<section.length; i++) {
            dimY[dimIdx] = section[i]; 
            Y[i] = this.empty(dimY);
        }
       
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize mul: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[i];
        
        int[] copyWidth = new int[Y.length];
        int[] strideY = new int[Y.length];
        for(int i = 0; i<copyWidth.length; i++){
            copyWidth[i] = commonWidth * section[i];//from dimIndex to End
            
            int width = Y[i].lastDim();//consider memory alginment
            int stride = ((width + 3) >> 2) << 2;
            strideY[i] = copyWidth[i] / width * stride;
            
            //width the same mem_struture, is dimIdex != -1
            if(dimIdx != ndim - 1) copyWidth[i] = strideY[i];
        }
        
        //compute the start index in X(src) for each element of Y[](dst)--------
        int strideX = commonWidth * dimX[dimIdx]; {//from dimIndex to End
            int width = X.lastDim();//consider memAlignment
            int stride = ((width + 3) >> 2) << 2;
            strideX = strideX / width * stride;
        }
        
        Syncer[] scs = new Syncer[Y.length];
        for(int i=0, Xstart = 0; i<Y.length; Xstart += copyWidth[i++]) {
            int length = ((dimIdx == ndim - 1) ? Y[i].length : Y[i].lengthv);
            scs[i] = core.gappedMemcpy2D(
                    X.address, Xstart, strideX, 
                    Y[i].c().address, 0, strideY[i],//Y[i] is synchronized
                    copyWidth[i], length);
        }
        
        if(sync) for (Syncer sc : scs) sc.sync();
        else for(int i=0; i<Y.length; i++) Y[i].setSyncer(scs[i]);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="split & chunk: (outs, X) -> Y[]">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] chunk(Tensor X, boolean[] outs, int dimIdx) {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int dimX[] = X.dim, dimSize = dimX[dimIdx];//dimSize = dimX[dimIndex] = sum(section)
        int n = outs.length;
        if(check) {
            require_dtype(X, "X"); must_greater_equal(n, "n", 2);
            if(n > dimSize) throw new IllegalArgumentException(String.format(
                    "n { got %d }> X.dim[%d] { got %d }", n, dimIdx, dimSize));
        }
        
        int[] section = new int[n];
        int div = dimSize  / n, rm = dimSize % n;
        for(int i=0; i<n; i++) section[i] = div;
        section[n - 1] += rm;
        
        return __split(X, outs, dimIdx, section);
    }
    
    @Passed("CudaFloat32EngieBase") 
    public Tensor[] split(Tensor X, boolean[] outs, int dimIdx, int...section) {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        int dimX[] = X.dim, dimSize = dimX[dimIdx];//dimSize = dimX[dimIndex] = sum(section)
        int sectionSum = negativeSection(dimSize, section);//exclude -1 in section
         
        if(check) {
            require_dtype(X, "X");
            equals(outs.length, "outs.length", section.length, "section.length");
            must_greater_equal(section.length, "section.length", 2);
            must_positive(section, "section");
            if(sectionSum != dimSize) throw new IllegalArgumentException(String.format(
                    "sum(section) { got %d } != X.dim[%d] { got %d }", 
                    sectionSum, dimIdx, dimSize));
        }
        
        return __split(X, outs, dimIdx, section);
    }
    
    //<editor-fold defaultstate="collapsed" desc="__split">
    private Tensor[] __split(Tensor X, boolean[] outs, int dimIdx, int[] section) {
        //create sub Tensor[] Y based on section--------------------------------
        int dimX[] = X.dim, ndim = X.ndim();
        
        Tensor[] Y = new Tensor[section.length];
        for(int i=0, dimY[] = Vector.arrayCopy(dimX); i<section.length; i++) {
            dimY[dimIdx] = section[i]; 
            Y[i] = (outs[i] ?
                    this.empty(dimY) ://with memspace
                    new Tensor(check, this, core.dataType(), dimY));//without memspace
        }
       
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize mul: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[i];
        
        int[] copyWidth = new int[Y.length];
        int[] strideY = new int[Y.length];
        for(int i = 0; i<copyWidth.length; i++){
            copyWidth[i] = commonWidth * section[i];//from dimIndex to End
            
            int width = Y[i].lastDim();//consider memory alginment
            int stride = ((width + 3) >> 2) << 2;
            strideY[i] = copyWidth[i] / width * stride;
            
            //width the same mem_struture, is dimIdex != -1
            if(dimIdx != ndim - 1) copyWidth[i] = strideY[i];
        }
        
        //compute the start index in X(src) for each element of Y[](dst)--------
        int strideX = commonWidth * dimX[dimIdx]; {//from dimIndex to End
            int width = X.lastDim();//consider memAlignment
            int stride = ((width + 3) >> 2) << 2;
            strideX = strideX / width * stride;
        }
        
        Syncer[] scs = new Syncer[Y.length];
        for(int i=0, Xstart = 0; i<Y.length; Xstart += copyWidth[i++]) {
            if(!outs[i]) continue;//only process tensor with out = true
            int length = ((dimIdx == ndim - 1) ? Y[i].length : Y[i].lengthv);
            scs[i] = core.gappedMemcpy2D(
                    X.address, Xstart, strideX, 
                    Y[i].c().address, 0, strideY[i],//Y[i] is synchronized
                    copyWidth[i], length);
        }
        
        if(sync) { for(int i=0; i<Y.length; i++) if(outs[i]) scs[i].sync(); }
        else { for(int i=0; i<Y.length; i++) if(outs[i]) Y[i].setSyncer(scs[i]); }
        for(int i=0; i<Y.length; i++) if(!outs[i]) Y[i] = null;//delete tensor with out = false
        
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="transpose(2D -> 4D): X -> X^T">
    @Passed("CudaFloat32EngieBase")
    public Tensor transpose(boolean inplace, Tensor X, int dimIdx1, int dimIdx2) {
        final int[] Xdim = X.dim;
        if(dimIdx1 < 0) dimIdx1 = Xdim.length + dimIdx1;
        if(dimIdx2 < 0) dimIdx2 = Xdim.length + dimIdx2;
        if(dimIdx1 == dimIdx2) return (inplace? X : this.copy(X));
        
        if(check) {
            require_dtype(X, "X");
            must_greater_equal(Xdim.length, "X.ndim", 2);
            must_smaller_equal(dimIdx1, "dimIdx1", Xdim.length, "X.ndim");
            must_smaller_equal(dimIdx2, "dimIdx2", Xdim.length, "X.ndim");
        }
        
        int[] Ydim = Vector.arrayCopy(Xdim);
        int t = Ydim[dimIdx1]; Ydim[dimIdx1] = Ydim[dimIdx2]; Ydim[dimIdx2] = t;
        Tensor Y = empty(Ydim).c();//Y[newDim, newAddress]
        
        Syncer sc1 = core.transpose(
                Y.address, Ydim, 
                X.address, Xdim,
                dimIdx1, dimIdx2,
                X.lastDim(), Y.lastDim(),
                X.length);
      
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="rot180: X -> Xr">
    @Passed("CudaFloat32EngieBase")
    public Tensor rot180(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 3); }
        
        int dimX[] = X.dim, ndim = dimX.length;
        int IH = dimX[ndim - 3], IW = dimX[ndim - 2], IC = dimX[ndim - 1];
        int N = (ndim == 3 ? 1 :  X.length / (IH * IW * IC));//[N, IH, IW, C]
       
        Tensor Y = this.empty(X.dim()).c();
        Syncer sc1 = core.rot180(Y.address, X.address, N, IH, IW, IC);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//X.dim = Y.dim, X.address = Y.address
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pad(2D -> 4D)">
    public Tensor pad2D(boolean inplace, Tensor X, int...p) {
        p = Vector.append(p, 0);//[0..., p, 0]
        return pad(inplace, X, p, Vector.arrayCopy(p));
    }
    public Tensor pad2D(boolean inplace, Tensor X, int[] p0, int[] p1) {
        p0 = Vector.append(p0, 0);//[0..., p0, 0]
        p1 = Vector.append(p1, 0);//[0..., p1, 0]
        return pad(inplace, X, p0, p1);
    }
    
    public Tensor pad(boolean inplace, Tensor X, int...p) { return pad(inplace, X, p, Vector.arrayCopy(p)); }
    @Passed("CudaFloat32EngieBase")
    public Tensor pad(boolean inplace, Tensor X, int[] p0, int[] p1) {
        int ndim = X.ndim();
        if(check) {
            require_dtype(X, "X");
            if(p0 != null) { 
                must_smaller_equal(p0.length, "p0.length", ndim, "Xndim");
                must_non_negative(p0, "p0");
            }
            if(p1 != null) { 
                must_smaller_equal(p1.length, "p1.length", ndim, "Xndim");
                must_non_negative(p1, "p1");
            }
        }
        
        //------[determine Y.dim]-----------------------------------------------
        p0 = Vector.expand_from_head(p0, ndim);//expand p0 (can be null) to ndim with 0s, [0, ...p0]
        p1 = Vector.expand_from_head(p1, ndim);//expand p1 (can be null) to ndim with 0s, [0,....p1]
        int[] Xdim = X.dim, Ydim = new int[ndim];
        for(int i=0; i<ndim; i++) Ydim[i] = p0[i] + Xdim[i] + p1[i];
        Tensor Y = this.zeros(Ydim).c();
        //------[determine Y.dim]-----------------------------------------------
       
        Syncer sc1 = (ndim > 1?
                core.pad(Y.address, Ydim, X.address, Xdim, p0):
                core.memcpy(Y.address, p0[0], X.address, 0, X.length));//ndim = 1, direct copy
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="trim(2D -> 4D)">
    public Tensor trim2D(boolean inplace, Tensor X, int... t) {
        t = Vector.append(t, 0);//[0..., t, 0]
        return trim(inplace, X, t, Vector.arrayCopy(t));
    }
    public Tensor trim2D(boolean inplace, Tensor X, int[] t0, int[] t1) {
        t0 = Vector.append(t0, 0);//[0..., p0, 0]
        t1 = Vector.append(t1, 0);//[0..., p1, 0]
        return trim(inplace, X, t0, t1);
    }
    
    public Tensor trim(boolean inplace, Tensor X, int... t) { return trim(inplace, X, t, Vector.arrayCopy(t)); }
    @Passed("CudaFloat32EngieBase")
    public Tensor trim(boolean inplace, Tensor X, int[] t0, int[] t1) {
        int ndim = X.ndim();
        if(check) {
            require_dtype(X, "X");
            if(t0 != null) {
                must_smaller_equal(t0.length, "t0.length", ndim, "X.ndim");
                must_non_negative(t0, "t0");
            }
            if(t1 != null) {
                must_smaller_equal(t1.length, "t1.length", ndim, "X.ndim");
                must_non_negative(t1, "t1");
            }
        }
        
        //------[determine Y.dim]-----------------------------------------------
        t0 = Vector.expand_from_head(t0, ndim);//expand t0 (can be null) to ndim with 0s, [0..., t0]
        t1 = Vector.expand_from_head(t1, ndim);//expand t1 (can be null) to ndim with 0s, [0..., t1]
        int[] Xdim = X.dim, Ydim = new int[ndim];
        for(int i=0; i<ndim; i++) Ydim[i] = Xdim[i] - t0[i] - t1[i];
        Tensor Y = this.zeros(Ydim).c();
        //------[determine Y.dim]-----------------------------------------------
       
        Syncer sc1 = (ndim > 1?
                core.trim(Y.address, Ydim, X.address, Xdim, t0):
                core.memcpy(Y.address, 0, X.address, t0[0], Y.length));//ndim = 1, direct copy
                
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="auxilary: start_from center">
    protected int[] start_from_center(int[] small_dim, int[] big_dim) {
        int ndim = small_dim.length;
        int[] arr = new int[ndim];//expand from the center
        for(int i=0; i<ndim; i++) {
            arr[i] = (big_dim[i] - small_dim[i]) >> 1;
            if(arr[i] < 0) arr[i] = 0;
        }
        return arr;
    }
    //</editor-fold>
    public static final int[] from_center = new int[0];
    
    //<editor-fold defaultstate="collapsed" desc="expand(2D -> 4D)">
    public Tensor expand2D(boolean inplace, Tensor X, int... out_dim) {
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return expand(inplace, X, from_center, out_dim);
    }
    public Tensor expand2D(boolean inplace, Tensor X, int[] start, int[] out_dim) {
        start = Vector.append(start, 0);//[0..., start, 0]
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return expand(inplace, X, start, out_dim);
    }
    
    public Tensor expand(boolean inplace, Tensor X, int... out_dim) { return expand(inplace, X, from_center, out_dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor expand(boolean inplace, Tensor X, int[] start, int[] out_dim) {
        int ndim = X.ndim(), Xdim[] = X.dim;
        if(check) {
            require_dtype(X, "X");
            if(out_dim != null) must_smaller_equal(out_dim.length, "out_dim.length", ndim, "X.ndim");
            if(start != null) {
                must_smaller_equal(start.length, "start.length", ndim, "X.ndim");
                must_non_negative(start, "start");
            }
        }
        
        //------[determine Y.dim]-----------------------------------------------
        int[] Ydim = Vector.expand_from_head_positive(out_dim, Xdim);//[Xdim..., out_dim], Ydim >= 0
        int[] p0 = (start == from_center?             //t >= 0
                start_from_center(X.dim, Ydim):            //[(Yim - Xdim) / 2...]
                Vector.expand_from_head(start, ndim));//[0..., p0]
        for(int i=0; i<ndim; i++) { int y2 = X.dim[i] + p0[i]; if(Ydim[i] < y2) Ydim[i] = y2; }
        //------[determine Y.dim]-----------------------------------------------
        
        Tensor Y = this.zeros(Ydim).c();
        Syncer sc1 = (ndim > 1?
                core.pad(Y.address, Y.dim, X.address, X.dim, p0) :
                core.memcpy(Y.address, p0[0], X.address, 0, X.length));//ndim = 1, direct copy
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="crop(2D -> 4D)">
    public Tensor crop2D(boolean inplace, Tensor X, int... out_dim) {
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return crop(inplace, X, from_center, out_dim);
    }
    public Tensor crop2D(boolean inplace, Tensor X, int[] start, int[] out_dim) {
        start = Vector.append(start, 0);//[0..., start_point, 0]
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return crop(inplace, X, start, out_dim);
    }
    
    public Tensor crop(boolean inplace, Tensor X, int...out_dim) { return crop(inplace, X, from_center, out_dim);  }
    @Passed("CudaFloat32EngieBase")
    public Tensor crop(boolean inplace, Tensor X, int[] start, int[] out_dim) {
        int ndim = X.ndim(), Xdim[] = X.dim;
        if(check) {
            require_dtype(X, "X");
            if(out_dim != null) must_smaller_equal(out_dim.length, "out_dim.length", ndim, "X.dim");
            if(start != null) {
                must_smaller_equal(start.length, "start.length", ndim, "X.dim");
                must_non_negative(start, "start");
            }
        }
       
        //------[determine Y.dim]-----------------------------------------------
        int[] Ydim = Vector.expand_from_head_positive(out_dim, Xdim);//[Xdim..., out_dim], Y.dim >= 0
        int[] t0 = (start == from_center ?            //t >= 0
                start_from_center(Ydim, Xdim) :            //[(Ydim - Xdim) / 2...]
                Vector.expand_from_head(start, ndim));//[0..., start_point]
        for(int i=0; i<ndim; i++) { int y = X.dim[i] - t0[i]; if(Ydim[i] > y) Ydim[i] = y; }
        //------[determine Y.dim]-----------------------------------------------
        
        Tensor Y = this.zeros(Ydim).c();
        Syncer sc1 = (ndim > 1?
                core.trim(Y.address, Ydim, X.address, Xdim, t0):
                core.memcpy(Y.address, 0, X.address, t0[0], Y.length));//ndim = 1, direct copy
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convlution 3D (NHWC)">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y)">
    public Tensor conv3D(Tensor Y, Tensor X, Tensor W, int sh, int sw) { return conv3D(Y, X, W, sh, sw, -1, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D(Tensor Y, Tensor X, Tensor W, int sh, int sw, int ph, int pw) {
       if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(Y.ndim(), "Y.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(Y.dim(0), "Y.batch", X.dim(0), "X.batch");
            equals(W.dim(0), "W.OC", Y.dim(3), "Y.OC");
            equals(W.dim(3), "W.IC", X.dim(3), "X.IC");
        }
        
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv3D(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW,
                XN, WIC, WOC,
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y, bias)">
    public Tensor conv3D_biased(Tensor Y, Tensor X, Tensor W, int sh, int sw, Tensor Bias) { return Engine.this.conv3D_biased(Y, X, W, sh, sw, -1, -1, Bias); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_biased(Tensor Y, Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias) {
        if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(Y.ndim(), "Y.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(Y.dim(0), "Y.batch", X.dim(0), "X.batch");
            equals(W.dim(0), "W.OC", Y.dim(3), "Y.OC");
            equals(W.dim(3), "W.IC", X.dim(3), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(0), "W.OC");
        }
        
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int OH  = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW, 
                XN, WIC, WOC, 
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW) -> Y">
    public Tensor conv3D(Tensor X, Tensor W, int sh, int sw, int ph, int pw) { return conv3D(X, W, -1, -1, sh, sw, ph, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D(Tensor X, Tensor W, int OH, int OW, int sh, int sw, int ph, int pw) {
        if(check) {
            require_dtype(X, "W"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(W.dim(3), "W.IC ", X.dim(3), "X.IC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(OH == -1) OH = (IH - FH + (ph << 1)) / sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        
        Tensor Y = this.empty(XN, OH, OW, WOC).c(); 
        Syncer sc = core.conv3D(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW,
                XN, WIC, WOC,
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW, bias)-> Y">
    public Tensor conv3D_biased(Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias) { return conv3D_biased(X, W, -1, -1, sh, sw, ph, pw, Bias); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_biased(Tensor X, Tensor W, int OH, int OW, 
            int sh, int sw, int ph, int pw, Tensor Bias) { 
        if(check) {
            require_dtype(X, "W"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(W.dim(3), "W.IC ", X.dim(3), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(0), "W.OC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(OH == -1) OH = (IH - FH + (ph << 1)) / sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        Tensor Y = this.empty(XN, OH, OW, WOC).c();
        
        Syncer sc = core.conv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW, 
                XN, WIC, WOC, 
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaW)">
    public Tensor conv3D_deltaW(Tensor deltaW, Tensor X, Tensor deltaY, int sh, int sw) { return conv3D_deltaW(deltaW, X, deltaY, sh, sw, -1, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_deltaW(Tensor deltaW, Tensor X, Tensor deltaY, int sh, int sw, int ph, int pw) {
        if(check) {
            require_dtype(deltaW, "deltaW"); require_dtype(X, "X"); require_dtype(deltaY, "deltaY"); 
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(deltaW.ndim(), "deltaW.ndim", 4);
            equals(deltaY.dim(0), "deltaY.batch", X.dim(0), "X.batch");
            equals(deltaW.dim(0), "deltaW.OC", deltaY.dim(3), "deltaY.OC");
            equals(deltaW.dim(3), "deltaW.IC", X.dim(3), "X.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim, dimW = deltaW.dim;
        int OH  = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]

        if(ph == -1) ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv3D_deltaW(
                deltaW.address, FH, FW,
                X.address, IH, IW,
                deltaY.address, OH, OW,
                XN, WIC, WOC,
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (FH, FW) -> deltaW">
    public Tensor conv3D_deltaW(Tensor X, Tensor deltaY, int sh, int sw, int ph, int pw) { return conv3D_deltaW(X, deltaY, -1, -1, sh, sw, ph, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_deltaW(Tensor X, Tensor deltaY, int FH, int FW, int sh, int sw, int ph, int pw) {
        if(check) {
            require_dtype(X, "X"); require_dtype(deltaY, "deltaY"); 
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(deltaY.dim(0), "deltaY.batch", X.dim(0), "X.batch");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim;
        int OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[N, IH, IW, IC]
        
        if(FH == -1) FH = IH + (ph << 1) - (OH - 1)*sh;//ceiling
        if(FW == -1) FW = IW + (pw << 1) - (OW - 1)*sw;//ceiling
        Tensor deltaW = this.empty(YOC, FH, FW, XIC).c();
        
        Syncer sc = core.conv3D_deltaW(
                deltaW.address, FH, FW,
                X.address,      IH, IW, 
                deltaY.address, OH, OW, 
                XN, XIC, YOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX)">
    public Tensor conv3D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sh, int sw) {  return conv3D_deltaX(deltaX, deltaY, W, sh, sw, -1, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sh, int sw, int ph, int pw) {
        if(check) {
            require_dtype(deltaX, "deltaX"); require_dtype(deltaY, "deltaY"); require_dtype(W, "W"); 
            equals(deltaX.ndim(), "deltaX.ndim", 4);
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(W.dim(0), "W.OC", deltaY.dim(3), "Y.OC");
            equals(deltaX.dim(0), "deltaX.batch", deltaY.dim(0), "deltaY.batch");
            equals(deltaX.dim(3), "deltaX.IC", W.dim(3), "W.IC");
        }
        
        int[] dimY = deltaY.dim, dimW = W.dim, dimX = deltaX.dim;
        int IH  = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int YN  = dimY[0], OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv3D_deltaX(
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW, 
                W.address,      FH, FW, 
                YN, WIC, WOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (IH, IW) -> deltaX">
    public Tensor conv3D_deltaX(Tensor deltaY, Tensor W, int sh, int sw, int ph, int pw) { return conv3D_deltaX(deltaY, W, -1, -1, sh, sw, ph, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_deltaX(Tensor deltaY, Tensor W, int IH, int IW, int sh, int sw, int ph, int pw) {
         if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(W, "W"); 
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(W.dim(0), "W.OC", deltaY.dim(3), "Y.OC");
        }
        
        int[] dimY = deltaY.dim, dimW = W.dim;
        int YN  = dimY[0], OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(IH == -1) IH = (OH - 1)*sh + FH - (ph << 1);//floor
        if(IW == -1) IW = (OW - 1)*sw + FW - (pw << 1);//floor
        Tensor deltaX = this.empty(YN, IH, IW, WIC).c();
        
        Syncer sc = core.conv3D_deltaX(
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW, 
                W.address,      FH, FW, 
                YN, WIC, WOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Deconvolution 3D (NHWC)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor deconv3D(Tensor Y, Tensor X, Tensor W, int sh, int sw) { return conv3D_deltaX(Y, X, W, sh, sw, -1, -1); }
    public Tensor deconv3D(Tensor Y, Tensor X, Tensor W, int sh, int sw, int ph, int pw) {
        return conv3D_deltaX(Y, X, W, sh, sw, ph, pw);
    }
    
    public Tensor deconv3D(Tensor X, Tensor W, int sh, int sw, int ph, int pw) { return conv3D_deltaX(X, W, -1, -1, sh, sw, ph, pw); }
    public Tensor deconv3D(Tensor X, Tensor W, int OH, int OW, int sh, int sw, int ph, int pw) {
        return conv3D_deltaX(X, W, OH, OW, sh, sw, ph, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y, bias)">
    public Tensor deconv3D_biased(Tensor Y, Tensor X, Tensor W, int sh, int sw, Tensor Bias) { return deconv3D_biased(Y, X, W, sh, sw, -1, -1, Bias);}
    @Passed("CudaFloat32EngieBase")
    public Tensor deconv3D_biased(Tensor Y, Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias) {
        if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(Y.ndim(), "Y.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(Y.dim(0), "Y.N", X.dim(0), "N");
            equals(W.dim(3), "W.OC", Y.dim(3), "Y.OC");
            equals(W.dim(0), "W.IC", X.dim(3), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(3), "W.OC");
        }
        
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WIC = dimW[0], FH = dimW[1], FW = dimW[2], WOC = dimW[3];//W[IC, FH, FW, OC]
        
        if(ph == -1) ph = ((IH - 1)*sh + FH - OH + 1) >> 1;//ceiling
        if(pw == -1) pw = ((IW - 1)*sw + FW - OW + 1) >> 1;//ceiling
        
        Syncer sc = core.deconv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW, 
                W.address, FH, FW, 
                XN, WIC, WOC,
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW, bias) -> Y">
    public Tensor deconv3D_biased(Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias) { 
        return deconv3D_biased(X, W, -1, -1, sh, sw, ph, pw, Bias);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor deconv3D_biased(Tensor X, Tensor W, int OH, int OW, int sh, int sw, int ph, int pw, Tensor Bias) {
        if(check) {
            require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(W.dim(0), "W.IC", X.dim(3), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(3), "W.OC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WIC = dimW[0], FH = dimW[1], FW = dimW[2], WOC = dimW[3];//W[IC, FH, FW, OC]
        
        if(OH == -1) OH = (IH - 1)*sh + FH - (ph << 1);//floor
        if(OW == -1) OW = (IW - 1)*sw + FW - (pw << 1);//floor
        Tensor Y = this.empty(XN, OH, OW, WOC).c();
        
        Syncer sc = core.deconv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW, 
                W.address, FH, FW, 
                XN, WIC, WOC,
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaW">
    public Tensor deconv3D_deltaW(Tensor deltaW, Tensor deltaY, Tensor X, int sh, int sw) {
        return conv3D_deltaW(deltaW, X, deltaY, sh, sw, -1, -1);//X is the filters
    }
    public Tensor deconv3D_deltaW(Tensor deltaW, Tensor deltaY, Tensor X, int sh, int sw, int ph, int pw) {
        return conv3D_deltaW(deltaW, X, deltaY, sh, sw, ph, pw);//X is the filters
    }
    
    public Tensor deconv3D_deltaW(Tensor deltaY, Tensor X, int sh, int sw, int ph, int pw) {
        return conv3D_deltaW(X, deltaY, -1, -1, sh, sw, ph, pw);//X is the filters
    }
    public Tensor deconv3D_deltaW(Tensor deltaY, Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) {
        return conv3D_deltaW(X, deltaY, FH, FW, sh, sw, ph, pw);//X is the filters
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Tensor deconv3D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sh, int sw) {
        return conv3D(deltaX, deltaY, W, sh, sw, -1, -1);
    }
    public Tensor deconv3D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sh, int sw, int ph, int pw) {
        return conv3D(deltaX, deltaY, W, sh, sw, ph, pw);
    }
    
    public Tensor deconv3D_deltaX(Tensor deltaY, Tensor W, int sh, int sw, int ph, int pw) {
        return conv3D(deltaY, W, -1, -1, sh, sw, ph, pw);
    }
    public Tensor deconv3D_deltaX(Tensor deltaY, Tensor W, int IH, int IW, int sh, int sw, int ph, int pw) {
        return conv3D(deltaY, W, IH, IW, sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convlution 2D (NWC)">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y)">
    public Tensor conv2D(Tensor Y, Tensor X, Tensor W, int sw) { return conv2D(Y, X, W, sw, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D(Tensor Y, Tensor X, Tensor W, int sw, int pw) {
       if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(Y.ndim(), "Y.ndim", 3);
            equals(X.ndim(), "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(Y.dim(0), "Y.batch", X.dim(0), "X.batch");
            equals(W.dim(0), "W.OC", Y.dim(2), "Y.OC");
            equals(W.dim(2), "W.IC", X.dim(2), "X.IC");
        }
        
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int OW  = dimY[1];//Y[N, OW, OC]
        int XN  = dimX[0], IW = dimX[1];//X[N, IW, IC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv2D(
                Y.address, OW,
                X.address, IW,
                W.address, FW,
                XN, WIC, WOC,
                sw, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y, bias)">
    public Tensor conv2D_biased(Tensor Y, Tensor X, Tensor W, int sw, Tensor Bias) { return conv2D_biased(Y, X, W, sw, -1, Bias); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D_biased(Tensor Y, Tensor X, Tensor W, int sw, int pw, Tensor Bias) {
        if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(Y.ndim(), "Y.ndim", 3);
            equals(X.ndim(), "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(Y.dim(0), "Y.batch", X.dim(0), "X.batch");
            equals(W.dim(0), "W.OC", Y.dim(2), "Y.OC");
            equals(W.dim(2), "W.IC", X.dim(2), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(0), "W.OC");
        }
        
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int OW  = dimY[1];//Y[N, OW, OC]
        int XN  = dimX[0], IW = dimX[1];//X[N, IW, IC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv2D_biased(
                Y.address, OW,
                X.address, IW,
                W.address, FW, 
                XN, WIC, WOC, 
                sw, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OW) -> Y">
    public Tensor conv2D(Tensor X, Tensor W, int sw, int pw) { return conv2D(X, W, -1, sw, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D(Tensor X, Tensor W, int OW, int sw, int pw) {
        if(check) {
            require_dtype(X, "W"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(W.dim(2), "W.IC ", X.dim(2), "X.IC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IW = dimX[1];//X[N, IW, IC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]
        
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        Tensor Y = this.empty(XN, OW, WOC).c();
         
        Syncer sc = core.conv2D(
                Y.address, OW,
                X.address, IW,
                W.address, FW,
                XN, WIC, WOC,
                sw, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OW, bias)-> Y">
    public Tensor conv2D_biased(Tensor X, Tensor W, int sw, int pw, Tensor Bias) { return conv2D_biased(X, W, -1, sw, pw, Bias); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D_biased(Tensor X, Tensor W, int OW, int sw, int pw, Tensor Bias) { 
        if(check) {
            require_dtype(X, "W"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(W.dim(2), "W.IC ", X.dim(2), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(0), "W.OC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IW = dimX[1];//X[N, IW, IC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]
        
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        Tensor Y = this.empty(XN, OW, WOC).c();
        
        Syncer sc = core.conv2D_biased(
                Y.address, OW,
                X.address, IW,
                W.address, FW, 
                XN, WIC, WOC, 
                sw, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaW)">
    public Tensor conv2D_deltaW(Tensor deltaW, Tensor X, Tensor deltaY, int sw) { return conv2D_deltaW(deltaW, X, deltaY, sw, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D_deltaW(Tensor deltaW, Tensor X, Tensor deltaY, int sw, int pw) {
        if(check) {
            require_dtype(deltaW, "deltaW"); require_dtype(X, "X"); require_dtype(deltaY, "deltaY"); 
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals(X.ndim(), "X.ndim", 3);
            equals(deltaW.ndim(), "deltaW.ndim", 3);
            equals(deltaY.dim(0), "deltaY.batch", X.dim(0), "X.batch");
            equals(deltaW.dim(0), "deltaW.OC", deltaY.dim(2), "deltaY.OC");
            equals(deltaW.dim(2), "deltaW.IC", X.dim(2), "X.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim, dimW = deltaW.dim;
        int OW  = dimY[1];//Y[N, OW, OC]
        int XN  = dimX[0], IW = dimX[1];//X[N, IW, IC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]

        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv2D_deltaW(
                deltaW.address, FW,
                X.address,      IW,
                deltaY.address, OW,
                XN, WIC, WOC,
                sw, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (FW) -> deltaW">
    public Tensor conv2D_deltaW(Tensor X, Tensor deltaY, int sw, int pw) { return conv2D_deltaW(X, deltaY, -1, sw, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D_deltaW(Tensor X, Tensor deltaY, int FW, int sw, int pw) {
        if(check) {
            require_dtype(X, "X"); require_dtype(deltaY, "deltaY"); 
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals(X.ndim(), "X.ndim", 3);
            equals(deltaY.dim(0), "deltaY.batch", X.dim(0), "X.batch");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim;
        int OW = dimY[1], YOC = dimY[2];//Y[N, OW, OC]
        int XN = dimX[0], IW  = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(FW == -1) FW = IW + (pw << 1) - (OW - 1)*sw;//ceiling
        Tensor deltaW = this.empty(YOC, FW, XIC).c();
        
        Syncer sc = core.conv2D_deltaW(
                deltaW.address, FW,
                X.address,      IW, 
                deltaY.address, OW, 
                XN, XIC, YOC, 
                sw, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX)">
    public Tensor conv2D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sw) { return conv2D_deltaX(deltaX, deltaY, W, sw, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sw, int pw) {
        if(check) {
            require_dtype(deltaX, "deltaX"); require_dtype(deltaY, "deltaY"); require_dtype(W, "W"); 
            equals(deltaX.ndim(), "deltaX.ndim", 3);
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(W.dim(0), "W.OC", deltaY.dim(2), "Y.OC");
            equals(deltaX.dim(0), "deltaX.batch", deltaY.dim(0), "deltaY.batch");
            equals(deltaX.dim(2), "deltaX.IC", W.dim(2), "W.IC");
        }
        
        int[] dimY = deltaY.dim, dimW = W.dim, dimX = deltaX.dim;
        int IW  = dimX[1];//X[N, IW, IC]
        int YN  = dimY[0], OW = dimY[1];//Y[N, OW, OC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.conv2D_deltaX(
                deltaX.address, IW, 
                deltaY.address, OW, 
                W.address,      FW, 
                YN, WIC, WOC, 
                sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (IW) -> deltaX">
    public Tensor conv2D_deltaX(Tensor deltaY, Tensor W, int sw, int pw) { return conv2D_deltaX(deltaY, W, -1, sw, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor conv2D_deltaX(Tensor deltaY, Tensor W, int IW, int sw, int pw) {
         if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(W, "W"); 
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(W.dim(0), "W.OC", deltaY.dim(2), "Y.OC");
        }
        
        int[] dimY = deltaY.dim, dimW = W.dim;
        int YN =  dimY[0], OW = dimY[1];//Y[N, OW, OC]
        int WOC = dimW[0], FW = dimW[1], WIC = dimW[2];//W[OC, FW, IC]
        
        if(IW == -1) IW = (OW - 1)*sw + FW - (pw << 1);//floor
        Tensor deltaX = this.empty(YN, IW, WIC).c();
        
        Syncer sc = core.conv2D_deltaX(
                deltaX.address, IW, 
                deltaY.address, OW, 
                W.address,      FW, 
                YN, WIC, WOC, 
                sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Deconvolution 2D (NWC)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor deconv2D(Tensor Y, Tensor X, Tensor W, int sw) { return conv2D_deltaX(Y, X, W, sw, -1); }
    public Tensor deconv2D(Tensor Y, Tensor X, Tensor W, int sw, int pw) { return conv2D_deltaX(Y, X, W, sw, pw); }
    
    public Tensor deconv2D(Tensor X, Tensor W, int sw, int pw) { return conv2D_deltaX(X, W, -1, sw, pw); }
    public Tensor deconv2D(Tensor X, Tensor W, int OW, int sw, int pw) { return conv2D_deltaX(X, W, OW, sw,  pw); }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y, bias)">
    public Tensor deconv2D_biased(Tensor Y, Tensor X, Tensor W, int sh, int sw, Tensor Bias) { return deconv3D_biased(Y, X, W, sh, sw, -1, -1, Bias);}
    @Passed("CudaFloat32EngieBase")
    public Tensor deconv2D_biased(Tensor Y, Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias) {
        if(check) {
            require_dtype(Y, "Y"); require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(Y.ndim(), "Y.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(Y.dim(0), "Y.N", X.dim(0), "N");
            equals(W.dim(3), "W.OC", Y.dim(3), "Y.OC");
            equals(W.dim(0), "W.IC", X.dim(3), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(3), "W.OC");
        }
        
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WIC = dimW[0], FH = dimW[1], FW = dimW[2], WOC = dimW[3];//W[IC, FH, FW, OC]
        
        if(ph == -1) ph = ((IH - 1)*sh + FH - OH + 1) >> 1;//ceiling
        if(pw == -1) pw = ((IW - 1)*sw + FW - OW + 1) >> 1;//ceiling
        
        Syncer sc = core.deconv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW, 
                W.address, FH, FW, 
                XN, WIC, WOC,
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW, bias) -> Y">
    public Tensor deconv2D_biased(Tensor X, Tensor W, int sw, int pw, Tensor Bias) { return deconv2D_biased(X, W, -1, sw, pw, Bias); }
    @Passed("CudaFloat32EngieBase")
    public Tensor deconv2D_biased(Tensor X, Tensor W, int OW, int sw, int pw, Tensor Bias) {
        if(check) {
            require_dtype(X, "X"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 3);
            equals(W.ndim(), "W.ndim", 3);
            equals(W.dim(0), "W.IC", X.dim(2), "X.IC");
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(2), "W.OC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IW = dimX[1];//X[N, IW, IC]
        int WIC = dimW[0], FW = dimW[1], WOC = dimW[2];//W[IC, FW, OC]
        
        if(OW == -1) OW = (IW - 1)*sw + FW - (pw << 1);//floor
        Tensor Y = this.empty(XN, OW, WOC).c();
        
        Syncer sc = core.deconv2D_biased(
                Y.address, OW,
                X.address, IW, 
                W.address, FW, 
                XN, WIC, WOC,
                sw, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaW">
    public Tensor deconv2D_deltaW(Tensor deltaW, Tensor deltaY, Tensor X, int sw) { return conv2D_deltaW(deltaW, X, deltaY, sw, -1); }//X is the filters
    public Tensor deconv2D_deltaW(Tensor deltaW, Tensor deltaY, Tensor X, int sw, int pw) { return conv2D_deltaW(deltaW, X, deltaY, sw, pw); }//X is the filters
    
    public Tensor deconv2D_deltaW(Tensor deltaY, Tensor X, int sw, int pw) { return conv2D_deltaW(X, deltaY, -1, sw, pw); }//X is the filters
    public Tensor deconv2D_deltaW(Tensor deltaY, Tensor X, int FW, int sw, int pw) { return conv2D_deltaW(X, deltaY, FW, sw, pw); }//X is the filters
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    public Tensor deconv2D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sw) { return conv2D(deltaX, deltaY, W, sw, -1); }
    public Tensor deconv2D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sw, int pw) { return conv2D(deltaX, deltaY, W, sw, pw); }
    
    public Tensor deconv2D_deltaX(Tensor deltaY, Tensor W, int sw, int pw) { return conv2D(deltaY, W, -1, sw, pw); }
    public Tensor deconv2D_deltaX(Tensor deltaY, Tensor W, int IW, int sw, int pw) { return conv2D(deltaY, W, IW, sw, pw); }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="DepthWise Convlution 3D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW) -> Y">
    public Tensor depthwise_conv3D(Tensor X, Tensor W, int sh, int sw, int ph, int pw) { return depthwise_conv3D(X, W, -1, -1, sh, sw, ph, pw);  }
    @Passed("CudaFloat32EngieBase")
    public Tensor depthwise_conv3D(Tensor X, Tensor W, int OH, int OW, int sh, int sw, int ph, int pw) {
        if(check) {//W[FH, FW, OC]
            require_dtype(X, "W"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 3);
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], IC = dimX[3];//X[N, IH, IW, IC]
        int FH = dimW[0], FW = dimW[1], OC = dimW[2];//W[FH, FW, OC]
        
        if(OH == -1) OH = (IH - FH + (ph << 1)) / sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        
        Tensor Y = this.empty(XN, OH, OW, OC).c(); 
        Syncer sc = core.depthwise_conv3D(
                Y.address, OH, OW, 
                X.address, IH, IW, 
                W.address, FH, FW, 
                XN, IC, OC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW, bias)-> Y">
    public Tensor depthwise_conv3D_biased(Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias) {
        return depthwise_conv3D_biased(X, W, -1, -1, sh, sw, ph, pw, Bias); 
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor depthwise_conv3D_biased(Tensor X, Tensor W, int OH, int OW, 
            int sh, int sw, int ph, int pw, Tensor Bias) { 
        if(check) {//W[FH, FW, OC]
            require_dtype(X, "W"); require_dtype(W, "W"); 
            equals(X.ndim(), "X.ndim", 4);
            equals(W.ndim(), "W.ndim", 3);
            equals(Bias.lastDim(), "Bias.lastDim", W.dim(2), "W.OC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], IC = dimX[3];//X[N, IH, IW, IC]
        int FH = dimW[0], FW = dimW[1], OC = dimW[2];//W[FH, FW, OC]
        
        if(OH == -1) OH = (IH - FH + (ph << 1)) / sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        Tensor Y = this.empty(XN, OH, OW, OC).c();
        
        Syncer sc = core.depthwise_conv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW, 
                XN, IC, OC, 
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (FH, FW) -> deltaW">
    public Tensor depthwise_conv3D_deltaW(Tensor X, Tensor deltaY, int sh, int sw, int ph, int pw) {
        return depthwise_conv3D_deltaW(X, deltaY, -1, -1, sh, sw, ph, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor depthwise_conv3D_deltaW(Tensor X, Tensor deltaY, int FH, int FW, int sh, int sw, int ph, int pw) {
        if(check) {
            require_dtype(X, "X"); require_dtype(deltaY, "deltaY"); 
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(X.ndim(), "X.ndim", 4);
            equals(deltaY.dim(0), "deltaY.batch", X.dim(0), "X.batch");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim;
        int OH = dimY[1], OW = dimY[2], OC = dimY[3];//Y[N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], IC = dimX[3];//X[ N, IH, IW, IC]
        
        if(FH == -1) FH = IH + (ph << 1) - (OH - 1)*sh;//ceiling
        if(FW == -1) FW = IW + (pw << 1) - (OW - 1)*sw;//ceiling
        Tensor deltaW = this.empty(FH, FW, OC).c();
        
        Syncer sc = core.conv3D_deltaW(
                deltaW.address, FH, FW,
                X.address, IH, IW, 
                deltaY.address, OH, OW, 
                XN, IC, OC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (IH, IW) -> deltaX">
    public Tensor depthwise_conv3D_deltaX(Tensor deltaY, Tensor W, int IC, int sh, int sw, int ph, int pw) { 
        return depthwise_conv3D_deltaX(deltaY, W, -1, -1, IC, sh, sw, ph, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor depthwise_conv3D_deltaX(Tensor deltaY, Tensor W, int IH, int IW, int IC, int sh, int sw, int ph, int pw) {
         if(check) {//W[FH, FW, OC]
            require_dtype(deltaY, "deltaY"); require_dtype(W, "W"); 
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(W.ndim(), "W.ndim", 4);
            equals(W.dim(3), "W.OC", deltaY.dim(3), "Y.OC");
        }
        
        int[] dimY = deltaY.dim, dimW = W.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int FH = dimW[0], FW = dimW[1], OC = dimW[2];//W[FH, FW, OC]
        
        if(IC == -1) IC = OC;//multiplier = 1
        if(IH == -1) IH = (OH - 1)*sh + FH - (ph << 1);//floor
        if(IW == -1) IW = (OW - 1)*sw + FW - (pw << 1);//floor
        Tensor deltaX = this.empty(YN, IH, IW, IC).c();
        
        Syncer sc = core.depthwise_conv3D_deltaX(
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW, 
                W.address, FH, FW, 
                YN, IC, OC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="DepthWise Deconvlution 3D">
    
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Pooling2D (NHWC)">
    //<editor-fold defaultstate="collapsed" desc="pool2D_param_check">
    protected final void check_pool2D(Tensor Y, Tensor X) {
        require_dtype(Y, "Y"); require_dtype(X, "X");
        equals(Y.ndim(), "Y.ndim", 4);
        equals(X.ndim(), "X.ndim", 4);
        equals(Y.dim(0), "Y.batch", X.dim(0), "X.batch");
        equals(Y.dim(3), "Y.IC", X.dim(3), "X.IC");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Max Pooling 2D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y)">
    public Tensor pool2D_max(Tensor Y, Tensor X, int FH, int FW, int sh, int sw) { 
        return pool2D_max(Y, X, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_max(Tensor Y, Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) {
        if(check) check_pool2D(Y, X);
        int[] dimY = Y.dim, dimX = X.dim;
        int OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[N, IH, IW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.pool2D_max(
                Y.address, OH, OW, 
                X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW) -> Y">
    public Tensor pool2D_max(Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) {
        return pool2D_max(X, FH, FW, -1, -1, sh, sw, ph, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_max(Tensor X, int FH, int FW, int OH, int OW, int sh, int sw, int ph, int pw) {
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 3); }
        
        int[] dimX = X.dim;
        //[ndim = 3]------------------------------------------------------------
        if (X.ndim() == 3) {
            int IH = dimX[0], IW = dimX[1], XIC = dimX[2];//X[IH, IW, IC]
            if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
            if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
            Tensor Y = this.empty(OH, OW, XIC).c();
            
            Syncer sc = core.pool2D_max(
                Y.address, OH, OW, 
                X.address, IH, IW,
                FH, FW, XIC, 
                sh, sw, ph, pw);
            if(sync) sc.sync(); else Y.setSyncer(sc);
            return Y;
        }
        
        //[ndim = 4]------------------------------------------------------------
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];
        if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        Tensor Y = this.empty(XN, OH, OW, XIC).c();
        
        Syncer sc = core.pool2D_max(
                Y.address, OH, OW, 
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX)">
    public Tensor unpool2D_max(Tensor deltaX, Tensor deltaY, Tensor Y, Tensor X, 
            int FH, int FW, int sh, int sw) {
        return unpool2D_max(deltaX, deltaY, Y, X, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool2D_max(Tensor deltaX, Tensor deltaY, Tensor Y, Tensor X, 
            int FH, int FW, int sh, int sw, int ph, int pw)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); 
            require_dtype(deltaX, "deltaX"); require_dtype(X, "X");
            equals(deltaX.ndim(), "deltaX.ndim", 4);
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals_valueStructure(deltaX, "deltaX", X, "X");
            equals(X.dim(0), "X.batch", Y.dim(0), "Y.batch");
            equals(X.dim(3), "X.IC", Y.dim(3), "Y.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.unpool2D_max(
                deltaY.address, Y.address, OH, OW, 
                deltaX.address, X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (X) -> deltaX">
    public Tensor unpool2D_max(Tensor deltaY, Tensor Y, Tensor X, 
            int FH, int FW, int sh, int sw) {
        return unpool2D_max(deltaY, Y, X, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool2D_max(Tensor deltaY, Tensor Y, Tensor X, 
            int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); require_dtype(X, "X");
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(X.dim(0), "X.batch", Y.dim(0), "Y.batch");
            equals(X.dim(3), "X.IC", Y.dim(3), "Y.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim;
        int OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[N, IH, IW, IC]
                
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        Tensor deltaX = this.empty(XN, IH, IW, XIC).c();
        
        Syncer sc = core.unpool2D_max(
                deltaY.address, Y.address, OH, OW, 
                deltaX.address, X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Max Pooling 2D indexed">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y, Index)">
    public Tensor pool2D_max_indexed(Tensor Y, Tensor Index, Tensor X, 
            int FH, int FW, int sh, int sw)  {
        return pool2D_max_indexed(Y, Index, X, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_max_indexed(Tensor Y, Tensor Index, Tensor X, 
            int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        if(check) { check_pool2D(Y, X);
            require_int32(Index, "Index");
            equals(Index.ndim(), "Index.ndim", 4);
            equals_dim(Index, "Index<int32>", Y, "Y");
        }
        
        int[] dimY = Y.dim, dimX = X.dim;
        int OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floors
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.pool2D_max_indexed(
                Y.address, Index.address, OH, OW,
                X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW) -> [Y, Index]">
    public Tensor[] pool2D_max_indexed(Tensor X, int FH, int FW, 
            int sh, int sw, int ph, int pw) {
        return pool2D_max_indexed(X, FH, FW, -1, -1, sh, sw, ph, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] pool2D_max_indexed(Tensor X,
            int FH, int FW, int OH, int OW,
            int sh, int sw, int ph, int pw) 
    {
        if(check) { require_dtype(X, "X"); equals(X.ndim(), "X.ndim", 4); }
        
        int[] dimX = X.dim;//X[ N, IH, IW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];
        
        if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        Tensor Y = this.empty(XN, OH, OW, XIC);
        Tensor Index = this.empty_int32(XN, OH, OW, XIC);
        
        Syncer sc = core.pool2D_max_indexed(
                Y.c().address, Index.c().address, OH, OW,
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[]{ Y, Index };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX, Index)">
    public Tensor unpool2D_max_indexed(Tensor deltaX, Tensor deltaY, Tensor Index, 
            int FH, int FW, int sh, int sw) {
        return unpool2D_max_indexed(deltaX, deltaY, Index, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool2D_max_indexed(Tensor deltaX, Tensor deltaY, Tensor Index, 
            int FH, int FW, int sh, int sw, int ph, int pw)
    {
         if(check) {
            require_dtype(deltaX, "deltaX"); require_int32(Index, "Index"); require_dtype(deltaY, "deltaY");
            equals(deltaX.ndim(), "deltaX.ndim", 4);
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals_valueStructure(Index, "Index<int32>", deltaY, "Y");
            equals(deltaY.dim(0), "deltaY.batch", deltaX.dim(0), "deltaX.batch");
            equals(deltaY.dim(3), "deltaY.IC", deltaX.dim(3), "deltaX.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
       
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.unpool2D_max_Indexed(
                deltaX.address, IH, IW, 
                deltaY.address, Index.address, OH, OW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (Index) -> deltaX">
    public Tensor unpool2D_max_indexed(Tensor deltaY, Tensor Index, 
            int IH, int IW, int FH, int FW, 
            int sh, int sw) {
        return unpool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool2D_max_indexed(Tensor deltaY, Tensor Index, 
            int IH, int IW, int FH, int FW, 
            int sh, int sw, int ph, int pw) 
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_int32(Index, "Index");
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals_valueStructure(Index, "Index<in32>", deltaY, "deltaY");
        }
        
        int[] dimY = deltaY.dim;//Y[ N, OH, OW, OC]
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();
        
        Syncer sc = core.unpool2D_max_Indexed(
                deltaX.address, IH, IW, 
                deltaY.address, Index.address, OH, OW, 
                FH, FW, YN, YIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Avg Pooling 2D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y)">
    public Tensor pool2D_avg(boolean ignore_padding, 
            Tensor Y, Tensor X, int FH, int FW, int sh, int sw) {
        return pool2D_avg(ignore_padding, Y, X, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_avg(boolean ignore_padding, 
            Tensor Y, Tensor X, int FH, int FW, 
            int sh, int sw, int ph, int pw) 
    {
        if(check) check_pool2D(Y, X);
        int[] dimY = Y.dim, dimX = X.dim;
        int OH = dimY[1], OW = dimY[2];//Y[N, OH, OW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[N, IH, IW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.pool2D_avg(ignore_padding,
                Y.address, OH, OW, 
                X.address, IH, IW, 
                FH, FW, 
                XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OH, OW) -> Y">
    public Tensor pool2D_avg(boolean ignore_padding,
            Tensor X, int FH, int FW, 
            int sh, int sw, int ph, int pw) {
        return pool2D_avg(ignore_padding, X, FH, FW, -1, -1, sh, sw, ph, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_avg(boolean ignore_padding, 
            Tensor X, int FH, int FW, int OH, int OW, 
            int sh, int sw, int ph, int pw) 
    {
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X,ndim", 3); }

        int[] dimX = X.dim;
        //[ndim = 3]------------------------------------------------------------
        if (X.ndim() == 3) {
            int IH = dimX[0], IW = dimX[1], XIC = dimX[2];//X[IH, IW, IC]
            if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
            if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
            Tensor Y = this.empty(OH, OW, XIC).c();
            
            Syncer sc = core.pool2D_avg(ignore_padding,
                    Y.address, OH, OW, 
                    X.address, IH, IW,
                    FH, FW, XIC, 
                    sh, sw, ph, pw);
            if(sync) sc.sync(); else Y.setSyncer(sc);
            return Y;
        }
        
        //[ndim = 4]------------------------------------------------------------
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        Tensor Y = this.empty(XN, OH, OW, XIC).c();
        
        Syncer sc = core.pool2D_avg(ignore_padding,
                Y.address, OH, OW,
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX)">
    public Tensor unpool2D_avg(boolean ignore_padding, 
            Tensor deltaX, Tensor deltaY, 
            int FH, int FW, int sh, int sw) {
        return unpool2D_avg(ignore_padding, deltaX, deltaY, FH, FW, sh, sw, -1, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool2D_avg(boolean ignore_padding,
            Tensor deltaX, Tensor deltaY, 
            int FH, int FW, int sh, int sw, int ph, int pw)
    {
        if(check) {
            require_dtype(deltaX, "deltaX"); require_dtype(deltaY, "deltaY");
            equals(deltaX.ndim(), "deltaX.ndim", 4);
            equals(deltaY.ndim(), "deltaY.ndim", 4);
            equals(deltaY.dim(0), "deltaY.batch", deltaX.dim(0), "deltaX.batch");
            equals(deltaY.dim(3), "deltaY.IC", deltaX.dim(3), "deltaX.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(ph == -1) ph = ((OH - 1)*sh + FH - IH) >> 1;//floor
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.unpool2D_avg(ignore_padding,
                deltaX.address, IH, IW, deltaY.address, OH, OW,
                FH, FW, YN, XIC, sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (IH, IW) -> deltaX">
    public Tensor unpool2D_avg(boolean ignore_padding,
            Tensor deltaY, int FH, int FW, 
            int sh, int sw, int ph, int pw) {
        return unpool2D_avg(ignore_padding, deltaY, FH, FW, -1, -1, sh, sw, ph, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool2D_avg(boolean ignore_padding,
            Tensor deltaY, int FH, int FW, int IH, int IW,
            int sh, int sw, int ph, int pw)
    {
        if(check) { require_dtype(deltaY, "deltaY"); equals(deltaY.ndim(), "deltaY.ndim", 4); }
        
        int[] dimY = deltaY.dim;//Y[ N, OH, OW, OC]
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];
        
        if(IH == -1) IH = (OH - 1)*sh + FH - (ph << 1);//floor
        if(IW == -1) IW = (OW - 1)*sw + FW - (pw << 1);//floor
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();  
        
        Syncer sc = core.unpool2D_avg(ignore_padding,
                deltaX.address, IH, IW, deltaY.address, OH, OW,
                FH, FW, YN, YIC, sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Pooling1D (NWC)">
    //<editor-fold defaultstate="collapsed" desc="pool1D_param_check">
    protected final void check_pool1D(Tensor Y, Tensor X) {
        require_dtype(Y, "Y"); require_dtype(X, "X");
        equals(Y.ndim(), "Y.ndim", 3);
        equals(X.ndim(), "X.ndim", 3);
        equals(Y.dim(0), "Y.batch", X.dim(0), "X.batch");
        equals(Y.dim(2), "Y.IC", X.dim(2), "X.IC");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Max Pooling 1D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y)">
    public Tensor pool1D_max(Tensor Y, Tensor X, int FW, int sw) { return pool1D_max(Y, X, FW, sw, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool1D_max(Tensor Y, Tensor X, int FW, int sw, int pw)  {
        if(check) check_pool1D(Y, X);
        int[] dimY = Y.dim, dimX = X.dim;
        int OW = dimY[1];//Y[N, OW, OC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.pool1D_max(
                Y.address, OW, 
                X.address, IW, 
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OW) -> Y">
    public Tensor pool1D_max(Tensor X, int FW, int sw, int pw) { return pool1D_max(X, FW, -1, sw, pw); }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool1D_max(Tensor X, int FW, int OW, int sw, int pw) {
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 3); }
       
        int[] dimX = X.dim;
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//[N, IW, IC]
        
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        Tensor Y = this.empty(XN, OW, XIC).c();
        
        Syncer sc = core.pool1D_max(
                Y.address, OW, 
                X.address, IW,
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX)">
    public Tensor unpool1D_max(Tensor deltaX, Tensor deltaY, Tensor Y, Tensor X, int FW, int sw) {
        return unpool1D_max(deltaX, deltaY, Y, X, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool1D_max(Tensor deltaX, Tensor deltaY, Tensor Y, Tensor X, int FW, int sw, int pw) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); 
            require_dtype(deltaX, "deltaX"); require_dtype(X, "X");
            equals(deltaX.ndim(), "deltaX.ndim", 3);
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals_valueStructure(deltaX, "deltaX", X, "X");
            equals(X.dim(0), "X.batch", Y.dim(0), "Y.batch");
            equals(X.dim(2), "X.IC", Y.dim(2), "Y.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int OW = dimY[1];//Y[N, OW, OC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.unpool1D_max(
                deltaY.address, Y.address, OW, 
                deltaX.address, X.address, IW, 
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (X) -> deltaX">
    public Tensor unpool1D_max(Tensor deltaY, Tensor Y, Tensor X, int FW, int sw) {
        return unpool1D_max(deltaY, Y, X, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool1D_max(Tensor deltaY, Tensor Y, Tensor X, int FW, int sw, int pw)  {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); require_dtype(X, "X");
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(X.dim(0), "X.batch", Y.dim(0), "Y.batch");
            equals(X.dim(2), "X.IC", Y.dim(2), "Y.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = X.dim;
        int OW = dimY[1];//Y[N, OW, IC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
                
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        Tensor deltaX = this.empty(XN, IW, XIC).c();
        
        Syncer sc = core.unpool1D_max(
                deltaY.address, Y.address, OW, 
                deltaX.address, X.address, IW, 
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Max Pooling 1D indexed">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y, Index)">
    public Tensor pool1D_max_indexed(Tensor Y, Tensor Index, Tensor X, int FW, int sw)  {
        return pool1D_max_indexed(Y, Index, X, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool1D_max_indexed(Tensor Y, Tensor Index, Tensor X, int FW, int sw, int pw)  {
        if(check) { check_pool1D(Y, X);
            require_int32(Index, "Index");
            equals(Index.ndim(), "Index.ndim", 3);
            equals_dim(Index, "Index<int32>", Y, "Y");
        }
        
        int[] dimY = Y.dim, dimX = X.dim;
        int OW = dimY[1];//Y[N, OW, OC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.pool1D_max_indexed(
                Y.address, Index.address, OW,
                X.address,IW, 
                FW, XN, XIC, 
               sw,  pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OW) -> [Y, Index]">
    public Tensor[] pool1D_max_indexed(Tensor X, int FW, int sw, int pw) {
        return pool1D_max_indexed(X, FW, -1, sw, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] pool1D_max_indexed(Tensor X, int FW, int OW, int sw, int pw)  {
        if(check) { require_dtype(X, "X"); equals(X.ndim(), "X.ndim", 3); }
        
        int[] dimX = X.dim;//X[N, IW, IC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];
        
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        Tensor Y = this.empty(XN, OW, XIC);
        Tensor Index = this.empty_int32(XN, OW, XIC);
        
        Syncer sc = core.pool1D_max_indexed(
                Y.c().address, Index.c().address, OW,
                X.address, IW,
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[]{ Y, Index };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX, Index)">
    public Tensor unpool1D_max_indexed(Tensor deltaX, Tensor deltaY, Tensor Index, int FW, int sw) {
        return unpool1D_max_indexed(deltaX, deltaY, Index, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool1D_max_indexed(Tensor deltaX, Tensor deltaY, Tensor Index, int FW, int sw, int pw) {
         if(check) {
            require_dtype(deltaX, "deltaX"); require_int32(Index, "Index"); require_dtype(deltaY, "deltaY");
            equals(deltaX.ndim(), "deltaX.ndim", 3);
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals_valueStructure(Index, "Index<int32>", deltaY, "Y");
            equals(deltaY.dim(0), "deltaY.batch", deltaX.dim(0), "deltaX.batch");
            equals(deltaY.dim(2), "deltaY.IC", deltaX.dim(2), "deltaX.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int OW = dimY[1];//Y[N, OW, IC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
       
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.unpool1D_max_Indexed(
                deltaX.address, IW, 
                deltaY.address, Index.address, OW, 
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (Index) -> deltaX">
    public Tensor unpool1D_max_indexed(Tensor deltaY, Tensor Index, int IW, int FW, int sw) {
        return unpool1D_max_indexed(deltaY, Index, IW, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool1D_max_indexed(Tensor deltaY, Tensor Index, int IW, int FW, int sw, int pw) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_int32(Index, "Index");
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals_valueStructure(Index, "Index<in32>", deltaY, "deltaY");
        }
        
        int[] dimY = deltaY.dim;//Y[N, OW, OC]
        int YN = dimY[0], OW = dimY[1], YIC = dimY[2];
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        Tensor deltaX = this.empty(YN, IW, YIC).c();
        
        Syncer sc = core.unpool1D_max_Indexed(
                deltaX.address, IW, 
                deltaY.address, Index.address, OW, 
                FW, YN, YIC, 
                sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Avg Pooling 1D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (Y)">
    public Tensor pool1D_avg(boolean ignore_padding, Tensor Y, Tensor X, int FW, int sw) {
        return pool1D_avg(ignore_padding, Y, X, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool1D_avg(boolean ignore_padding, Tensor Y, Tensor X, int FW, int sw, int pw) {
        if(check) check_pool1D(Y, X);
        int[] dimY = Y.dim, dimX = X.dim;
        int OW = dimY[1];//Y[N, OW, IC]
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.pool1D_avg(ignore_padding,
                Y.address, OW, 
                X.address, IW, 
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: (OW) -> Y">
    public Tensor pool1D_avg(boolean ignore_padding, Tensor X, int FW, int sw, int pw) {
        return pool1D_avg(ignore_padding, X, FW, -1, sw, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor pool1D_avg(boolean ignore_padding, Tensor X, int FW, int OW, int sw, int pw) {
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X,ndim", 3); }

        int[] dimX = X.dim;
        int XN = dimX[0], IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        Tensor Y = this.empty(XN, OW, XIC).c();
        
        Syncer sc = core.pool1D_avg(ignore_padding,
                Y.address, OW,
                X.address, IW,
                FW, XN, XIC, 
                sw, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (deltaX)">
    public Tensor unpool1D_avg(boolean ignore_padding, Tensor deltaX, Tensor deltaY, int FW, int sw) {
        return unpool1D_avg(ignore_padding, deltaX, deltaY, FW, sw, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool1D_avg(boolean ignore_padding, Tensor deltaX, Tensor deltaY, int FW, int sw, int pw) {
        if(check) {
            require_dtype(deltaX, "deltaX"); require_dtype(deltaY, "deltaY");
            equals(deltaX.ndim(), "deltaX.ndim", 3);
            equals(deltaY.ndim(), "deltaY.ndim", 3);
            equals(deltaY.dim(0), "deltaY.batch", deltaX.dim(0), "deltaX.batch");
            equals(deltaY.dim(2), "deltaY.IC", deltaX.dim(2), "deltaX.IC");
        }
        
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int YN = dimY[0], OW = dimY[1];//Y[N, OW, OC]
        int IW = dimX[1], XIC = dimX[2];//X[N, IW, IC]
        
        if(pw == -1) pw = ((OW - 1)*sw + FW - IW) >> 1;//floor
        
        Syncer sc = core.unpool1D_avg(ignore_padding,
                deltaX.address, IW, deltaY.address, OW,
                FW, YN, XIC, sw, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: (IW) -> deltaX">
    public Tensor unpool1D_avg(boolean ignore_padding, Tensor deltaY, int FW, int sw, int pw) {
        return unpool1D_avg(ignore_padding, deltaY, FW, -1, sw, pw);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor unpool1D_avg(boolean ignore_padding, Tensor deltaY, int FW, int IW, int sw, int pw) {
        if(check) { require_dtype(deltaY, "deltaY"); equals(deltaY.ndim(), "deltaY.ndim", 3); }
        
        int[] dimY = deltaY.dim;
        int YN = dimY[0], OW = dimY[1], YIC = dimY[2];//Y[N, OW, OC]
        
        if(IW == -1) IW = (OW - 1)*sw + FW - (pw << 1);//floor
        Tensor deltaX = this.empty(YN, IW, YIC).c();  
        
        Syncer sc = core.unpool1D_avg(ignore_padding,
                deltaX.address, IW, deltaY.address, OW,
                FW, YN, YIC, sw,pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="func_param_check">
    protected final void check_row(Tensor X1, String name1, Tensor X2, String name2) {
        //X1.width == X2.width -> X1.stride == X2.stride
        if(X1.ndim() < 2) throw new IllegalArgumentException(String.format(
                "%s.ndim { got %d } must >= 2", name1, X1.ndim()));
        if(X1.lastDim() != X2.lastDim()) throw new IllegalArgumentException(String.format(
                "%s.lastDim { got %d } != %s.lastDim { got %d }",
                name1, X1.lastDim(), name2, X2.lastDim()));
    }   
    
    public final void check_center(Tensor X1, String name1, Tensor X2, String name2, int dim2) {
        if(X1.ndim() < 3) throw new IllegalArgumentException(String.format(
                "%s.ndim { got %d } must >= 3", name1, X1.ndim()));
        if(X2.ndim() < 2) throw new IllegalArgumentException(String.format(
                "%s.ndim { got %d } must >= 2", name2, X2.ndim()));
        if(X1.lastDim() != X2.lastDim()) throw new IllegalArgumentException(String.format(
                "%s.lastDim { got %d } != %s.lastDim { got %d }",
                name1, X1.lastDim(), name2, X2.lastDim()));
        if(X1.length % X2.length != 0) throw new IllegalArgumentException(String.format(
                "%s.length { got %d } %% %s.length { got %d } != 0", name1, X1.length, name2, X2.length));
        if(X2.length % dim2 != 0) throw new IllegalArgumentException(String.format(
                "%s.length { got %d } %% dim2 { got %d } != 0", name2, X2.length, dim2));
    }
    
    protected final void check_field(Tensor X1, String name1, Tensor X2, String name2) {
        if(X1.ndim() < 2) throw new IllegalArgumentException(String.format(
                "%s.ndim { got %d } must >= 2", name1, X1.ndim()));
        if(X2.ndim() != 1 && X2.isMemAligned()) throw new IllegalArgumentException(String.format(
                "%s.ndim { got %d } must == 1", name2, X2.ndim()));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="equal, linear, quadratic"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs">
    public Tensor equal(Tensor X1, Tensor X2) { return equal_abs(true, X1, X2, 0, 0); }
    public Tensor equal(boolean likeX1, Tensor X1, Tensor X2) { return equal_abs(likeX1, X1, X2, 0.0f, 0.0f); }
    
    public Tensor equal_abs(Tensor X1, Tensor X2, float min, float max) { return equal_abs(true, X1, X2, min, max); }
    @Passed("CudaFloat32EngieBase")//min <= |X1 - X2| <= max
    public Tensor equal_abs(boolean likeX1, Tensor X1, Tensor X2, float min, float max) {
        if(X1.dataType.equals(core.dataType_int8()) && X2.dataType.equals(core.dataType_int8()))
            return equal_abs_int8(likeX1, X1, X2, (byte)min, (byte)max);
        
        if(X1.dataType.equals(core.dataType_int32()) && X2.dataType.equals(core.dataType_int32()))
            return equal_abs_int32(likeX1, X1, X2, (int)min, (int)max);
 
        Tensor Y1 = X1.dataType.equals(core.dataType())? X1 : this.to_dtype(false, X1);
        Tensor Y2 = X2.dataType.equals(core.dataType())? X2 : this.to_dtype(false, X2);
        
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.equal_abs2D(Y.address,
                X1.address, X2.address, min, max, 
                Y.lengthv, Y.lastDim());
        
        if(sync) { sc.sync(); if(Y1 != X1) Y1.delete(); if(Y2 != X2) Y2.delete(); }
        else { Y.setSyncer(Syncer.dual(sc, ()->{  if(Y1 != X1) Y1.delete(); if(Y2 != X2) Y2.delete(); })); }
        return Y;
    }
    
    public Tensor equal_abs_int8(Tensor X1, Tensor X2, byte min, byte max) { 
        return equal_abs_int8(true, X1, X2, min, max);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor equal_abs_int8(boolean likeX1, Tensor X1, Tensor X2, byte min, byte max) {
        if(check) { require_int8(X1, "X1"); require_int8(X2, "X2"); }
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.equal_abs2D_int8(Y.address, 
                X1.address, X2.address, min, max,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
     
    public Tensor equal_abs_int32(Tensor X1, Tensor X2, int min, int max) {
        return equal_abs_int32(true, X1, X2, min, max);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor equal_abs_int32(boolean likeX1, Tensor X1, Tensor X2, int min, int max) {
        if(check) { require_int32(X1, "X1"); require_int32(X2, "X2"); }
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.equal_abs2D_int32(Y.address, 
                X1.address, X2.address, min, max,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear_greater">
    public Tensor gt(boolean inplace, Tensor X, float v) { return linear_greater(inplace, 1.0f, X, -v); }//X > v => X - v > 0
    public Tensor lt(boolean inplace, Tensor X, float v) { return linear_greater(inplace, -1.0f, X, v); }//X < v => X - v < 0 -> -X + v > 0
    
    @Passed("CudaFloat32EngieBase")//alpha*X + beta > 0
    public Tensor linear_greater(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim)).c();
        Syncer sc = core.linear_greater2D(Y.address, 
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater2">
    public Tensor gt2(boolean inplace, Tensor X1, Tensor X2) { //X1 > X2 -> X1 - X2 > 0
        return linear_greater2(inplace, X1, X2, 1.0f, -1.0f, 0.0f); 
    }
    public Tensor lt2(boolean inplace, Tensor X1, Tensor X2) {//X1 < X2 -> X1 - X2 < 0 -> -X1 + X2 > 0
        return linear_greater2(inplace, X1, X2, -1.0f, 1.0f, 0.0f);
    }
    
    public Tensor linear_greater2(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) {
        return linear_greater2(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//alpha*X1 + beta*X2 + gamma > 0
    public Tensor linear_greater2(boolean inplace, boolean likeX1,
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear_greater2_2D(Y.address, 
                X1.address, X2.address,
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater_switch">
    public Tensor gt_switch(boolean inplace, Tensor X, float v, float v1, float v2) {
        return linear_greater_switch(inplace, 1.0f, X, -v, v1, v2);//X > v => X - v > 0
    }
    public Tensor lt_switch(boolean inplace, Tensor X, float v, float v1, float v2) {
        return linear_greater_switch(inplace, -1.0f, X, v, v1, v2);//X < v => X - v < 0 -> -X + v > 0
    }
    
    @Passed("CudaFloat32EngieBase")//alpha*X + beta > 0 ? v1 : v2
    public Tensor linear_greater_switch(boolean inplace, 
            float alpha, Tensor X, float beta, 
            float v1, float v2) 
    {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim)).c();
        Syncer sc = core.linear_greater_switch2D(Y.address,
                alpha, X.address, beta,
                v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater_switch_mul">
    public Tensor gt_switch_mul(boolean inplace, 
            Tensor X1, float v, 
            Tensor X2, float v1, float v2) {//X1 > v => X1 - v > 0
        return linear_greater_switch_mul(inplace, 1.0f, X1, -v, X2, v1, v2);
    }
    public Tensor lt_switch_mul(boolean inplace, 
            Tensor X1, float v, 
            Tensor X2, float v1, float v2) {//X < v => X - v < 0 -> -X + v > 0
        return linear_greater_switch_mul(inplace, -1.0f, X1, v, X2, v1, v2);
    }
    
    public Tensor linear_greater_switch_mul(boolean inplace,//default likeX1
            float alpha, Tensor X1, float beta, 
            Tensor X2, float v1, float v2) {
        return linear_greater_switch_mul(inplace, true, alpha, X1, beta, X2, v1, v2);
    }
    @Passed("CudaFloat32EngieBase")//X2 * (alpha*X1 + beta > 0 ? v1 : v2)
    public Tensor linear_greater_switch_mul(boolean inplace, boolean likeX1,
            float alpha, Tensor X1, float beta, 
            Tensor X2, float v1, float v2) 
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear_greater_switch_mul2D(Y.address, 
                alpha, X1.address, beta,
                X2.address, v1, v2,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_bound_switch_mul">
    public Tensor linear_bound_switch_mul(boolean inplace,//default likeX1
            float alpha, Tensor X1, float vmin, float vmax,
            Tensor X2, float v1, float v2, float v3) {
        return linear_bound_switch_mul(inplace, true, alpha, X1, vmin, vmax, X2, v1, v2, v3);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_bound_switch_mul(boolean inplace, boolean likeX1,
            float alpha, Tensor X1, float vmin, float vmax,
            Tensor X2, float v1, float v2, float v3) 
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear_bound_switch_mul2D(Y.address,
                alpha, X1.address, vmin, vmax, 
                X2.address, v1, v2, v3,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear">
    public Tensor sadd(boolean inplace, Tensor X, float C) { return linear(inplace, 1.0f, X,  C); }
    public Tensor ssub(boolean inplace, Tensor X, float C) { return linear(inplace, 1.0f, X, -C); }
    public Tensor smul(boolean inplace, Tensor X, float C) { return linear(inplace, C, X, 0.0f); }
    public Tensor sdiv(boolean inplace, Tensor X, float C) { return linear(inplace, 1.0f / C, X, 0.0f); }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.linear2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: change datatype">
    //<editor-fold defaultstate="collapsed" desc="linear: int8 to dtype">
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_int8_to_dtype(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_int8(X, "X"); }
        Tensor Y = this.empty(X.dim).c();
        Syncer sc1 = core.linear2D_int8_to_dtype(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); Y.setSyncer(sc1); return Y;  }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int8> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_dtype_to_int8(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty_int8(X.dim).c();
        Syncer sc1 = core.linear2D_dtype_to_int8(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<dtype> -> X<int8>
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int32 to dtype">
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_int32_to_dtype(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_int32(X, "X"); } 
        Tensor Y = this.empty(X.dim).c();
        Syncer sc1 = core.linear2D_int32_to_dtype(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int32> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
     
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_dtype_to_int32(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty_int32(X.dim).c();
        Syncer sc1 = core.linear2D_dtype_to_int32(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
       
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<dtype> -> X<int32>
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    public Tensor dtype_to_int8(boolean inplace, Tensor X) { return linear_dtype_to_int8(inplace, 1.0f, X, 0.0f); }
    public Tensor dtype_to_int32(boolean inplace, Tensor X) { return linear_dtype_to_int32(inplace, 1.0f, X, 0.0f); }
    
    public Tensor to_dtype(boolean inplace, Tensor X) {
        if(X.dataType.equals(core.dataType_int8()))  return linear_int8_to_dtype(inplace, 1.0f, X, 0.0f);
        if(X.dataType.equals(core.dataType_int32())) return linear_int32_to_dtype(inplace, 1.0f, X, 0.0f);
        return inplace? X : copy(X);
    }
    
    public Tensor linear_to_dtype(boolean inplace, float alpha, Tensor X, float beta) {
        if(X.dataType.equals(core.dataType_int8()))  return linear_int8_to_dtype(inplace, alpha, X, beta);
        if(X.dataType.equals(core.dataType_int32())) return linear_int32_to_dtype(inplace, alpha, X, beta);
        return linear(inplace, alpha, X, beta);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_2out">
    public Tensor[] sadd_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, 0, C1, 0, C2);
    }
    public Tensor[] ssub_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, 0, -C1, 0, -C2);
    }
    public Tensor[] smul_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, C1, 0, C2, 0);
    }
    public Tensor[] sdiv_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, (1.0f / C1), 0, (1.0f / C2), 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear_2out(boolean inplace, Tensor X, 
            float alpha1, float beta1,
            float alpha2, float beta2) 
    {
        if(check) { require_dtype(X, "X"); }
        
        Tensor Y1 = (inplace? X : this.empty(X.dim));
        Tensor Y2 = this.empty(X.dim);
        
        Syncer sc = core.linear_2out2D(
                Y1.c().address,//result0
                Y2.c().address,//result1
                X.address,
                alpha1, beta1, 
                alpha2, beta2,
                X.lengthv, X.lastDim());
      
        if(sync) sc.sync(); else { Y1.setSyncer(sc); Y2.setSyncer(sc); }
        return new Tensor[] { Y1, Y2 };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear2">
    public Tensor sub(boolean inpalce, Tensor X1, Tensor X2) { return linear2(inpalce, X1, X2, 1.0f, -1.0f, 0.0f);}
    public Tensor sub(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2(inplace, X1, X2, alpha, -beta, 0.0f);
    }
    
    public Tensor add(boolean inplace, Tensor X1, Tensor X2) { return linear2(inplace, X1, X2, 1.0f, 1.0f, 0.0f); }
    public Tensor add(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2(inplace, X1, X2, alpha, beta, 0.0f);
    }
    
    public Tensor linear2(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) {
        return linear2(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2,
            float alpha, float beta, float gamma)
    {
        if(check) {//Y = alpha*X1 + beta*X2 + gamma
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_2D(Y.address,
                X1.address, X2.address,
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_row">
    public Tensor sub_row(boolean inplace, Tensor X1, Tensor X2) { 
        return linear2_row(inplace, X1, X2, 1.0f, -1.0f, 0);
    }
    public Tensor add_row(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_row(inplace, X1, X2, 1.0f, 1.0f, 0);
    }
    public Tensor add_row(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_row(inplace, X1, X2, alpha, beta, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_row(boolean inplace, Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) 
    {
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_row(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc= core.linear2_2D_row(Y.address, 
                X1.address,
                X2.address, X2.lengthv, //X2.lengthv = row_lengthv
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_center">
    //<editor-fold defaultstate="collapsed" desc="foward-propagation">
    public Tensor sub_center(boolean inplace, Tensor X1, Tensor X2) { return linear2_center(inplace, X1, X2, -1, 1.0f, -1.0f, 0);}
    public Tensor sub_center(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_center(inplace, X1, X2, -1, alpha, -beta, 0);
    }
    
    public Tensor add_center(boolean inplace, Tensor X1, Tensor X2) { return linear2_center(inplace, X1, X2, -1, 1.0f, 1.0f, 0); }
    public Tensor add_center(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_center(inplace, X1, X2, -1, alpha, beta, 0);
    }
    
    public Tensor linear2_center(boolean inplace, Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return linear2_center(inplace, X1, X2, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_center(boolean inplace, 
            Tensor X1, Tensor X2, int dim2,
            float alpha, float beta, float gamma)
    {
        if(dim2 == -1) dim2 = X1.lastDim();
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_center(X1, "X1", X2, "X2", dim2);
        }
        
        Tensor Y = (inplace? X1 : empty(X1.dim));
        int dim1 = X1.length / X2.length;
        int dim0 = X2.length / dim2;//X2.length = dim1 * dim2
        
        Syncer sc = core.linear2_2D_center(Y.c().address, 
                X1.address, X2.address,
                alpha, beta, gamma, 
                dim0, dim1, dim2,
                X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="foward-propagation">
    public Tensor[] linear2_center_deltaX(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2, int dim2, 
            float alpha, float beta, float gamma) {
        return quadratic2_center_deltaX(inplace, deltaY, X1, X2, dim2, 
                0.0f, 0.0f, 0.0f, 
                alpha, beta, gamma);
    }

    public Tensor linear2_center_deltaX1(boolean inplace, Tensor deltaY, 
            Tensor X1, Tensor X2, int dim2, 
            float alpha) {
        return quadratic2_center_deltaX1(inplace, deltaY, X1, X2, dim2, 
                0.0f, 0.0f, alpha);
    }

    public Tensor linear2_center_deltaX2(Tensor deltaY, 
            Tensor X1, Tensor X2, int dim2,
            float beta) {
        return quadratic2_center_deltaX2(deltaY, X1, X2, dim2, 
                0.0f, 0.0f, beta);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_field">
    public Tensor sub_field(boolean inplace, Tensor X1, Tensor X2) { return linear2_field(inplace, X1, X2, 1.0f, -1.0f, 0.0f); }
    public Tensor sub_field(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_field(inplace, X1, X2, alpha, -beta, 0.0f);
    }
    
    public Tensor add_field(boolean inplace, Tensor X1, Tensor X2) { return linear2_field(inplace, X1, X2, 1.0f, 1.0f, 0.0f); }
    public Tensor add_field(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_field(inplace, X1, X2, alpha, beta, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_field(boolean inplace, Tensor X1, Tensor X2,
            float alpha, float beta, float gamma)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); 
            check_field(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : empty(X1.dim).c());
        Syncer sc = core.linear2_2D_field(Y.address, 
                X1.address, 
                X2.address, X2.length,//X2.length = field_length
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_sum">
    public Tensor mean(boolean inplace, Tensor... Xs) {
        float alpha = (float) (1.0 / Xs.length);
        return Engine.this.linear_sum(inplace, alpha, 0.0f, Xs);//Y = sum(Xs[i])
    }
    public Tensor sum(boolean inplace, Tensor... X) {//Y = sum(Xs[i])
        return Engine.this.linear_sum(inplace, 1.0f, 0.0f, X); 
    }
    public Tensor sum(boolean inplace, float alpha, Tensor... Xs) {//Y = alpha * sum(Xs[i])
        return Engine.this.linear_sum(inplace, alpha, 0.0f, Xs);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_sum(boolean inplace, float alpha, float beta, Tensor... Xs) {
        Tensor X0 = Xs[0]; 
        if(Xs.length == 1) { return linear(inplace, alpha, X0, beta); }
        if(check) { require_dtype(Xs, "Xs");  equals_valueStructure(X0, "Xs[0]", Xs, "Xs"); }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs.length]; 
        for(int i=0; i<Xs.length; i++) addrs[i] = Xs[i].address;
        
        Syncer sc = core.linear_summary2D(Y.c().address, 
                alpha, beta, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor mean(boolean inplace, Collection<Tensor> Xs) {
        float alpha = (float) (1.0 / Xs.size());
        return linear_sum(inplace, alpha, 0.0f, Xs);
    }
    public Tensor sum(boolean inplace, Collection<Tensor> Xs) {//Y = sum(Xs[i])
        return linear_sum(inplace, 1.0f, 0.0f, Xs); 
    }
    public Tensor sum(boolean inplace, float alpha, Collection<Tensor> Xs) {//Y = alpha * sum(Xs[i])
        return linear_sum(inplace, alpha, 0.0f, Xs);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_sum(boolean inplace, float alpha, float beta, Collection<Tensor> Xs) {//inplace: Xs[0]
        int Xs_size = Xs.size();
        Iterator<Tensor> iter = Xs.iterator(); Tensor X0 = iter.next();
        if(Xs_size == 1) { return linear(inplace, alpha, X0, beta); }
        if(check) { require_dtype(Xs, "Xs"); equals_valueStructure(X0, "Xs[0]", Xs, "Xs"); }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs_size]; int index = 0;
        for(Tensor X : Xs) addrs[index++] = X.address;
        
        Syncer sc = core.linear_summary2D(Y.c().address, 
                alpha, beta, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_iteration">
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_iteration(boolean inplace,//inplace: X1
            float alpha, float beta, float gamma,
            Tensor X1, Tensor... X2) 
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "Xs");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim));
        long[] addrs = new long[X2.length]; int index = 0;
        for(Tensor X : X2) addrs[index++] = X.address;
        
        Syncer sc = core.linear2_iteration2D(Y.address,
                X1.address, addrs,
                alpha, beta, gamma,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_iteration(boolean inplace,//inplace: X1
            float alpha, float beta, float gamma,
            Tensor X1, Collection<Tensor> X2) 
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim));
        long[] addrs = new long[X2.size()]; int index = 0;
        for(Tensor X : X2) addrs[index++] = X.address;
        
        Syncer sc = core.linear2_iteration2D(Y.address,
                X1.address, addrs,
                alpha, beta, gamma,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="mul_linear2">
    @Passed("CudaFloat32EngieBase")
    public Tensor mul_linear2(boolean inplace, 
            Tensor X, Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma)
    {
        if(check) { 
            require_dtype(X, "X"); require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X, "X", X1, "X1"); 
            equals_valueStructure(X, "X", X2, "X2"); 
        }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.mul_linear2_2D(Y.address, 
                X.address, X1.address, X2.address, 
                alpha, beta, gamma, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic">
    public Tensor square(boolean inplace, Tensor X) { return quadratic(inplace, X, 1.0f, 0.0f, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic(boolean inplace, Tensor X, float alpha, float beta, float gamma) {
        if(check) { require_dtype(X, "X"); }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.quadratic2D(Y.address,
                X.address, alpha, beta, gamma,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic_deltaX(boolean inplace, Tensor deltaY, Tensor X, float alpha, float beta) {
        if(check) { 
            require_dtype(X, "X"); require_dtype(deltaY, "deltaY");
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.quadratic2D_deltaX(deltaX.address,
                deltaY.address, 
                X.address, alpha, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor var(boolean inplace, Tensor X_sqmean, Tensor X_mean) {//X_sqmean - X_mean^2
        return quadratic2(inplace, X_sqmean, X_mean, 0, 0, -1.0f, 1.0f, 0, 0);
    }
    
    public Tensor mul(boolean inplace, Tensor X1, Tensor X2) { 
        return quadratic2(inplace, X1, X2, 0, 1.0f, 0, 0, 0, 0);
    }
    public Tensor mul(boolean inplace, float alpha, Tensor X1, Tensor X2) { 
        return quadratic2(inplace, X1, X2, 0, alpha, 0, 0, 0, 0);
    }
  
    public Tensor sqadd(boolean inplace, Tensor X1, Tensor X2) {//X1^2 + X2^2
        return quadratic2(inplace, X1, X2, 1.0f, 0, 1.0f, 0, 0, 0); 
    }
    public Tensor sqadd(boolean inplace, Tensor X1, Tensor X2, float alpha, float beta) {//alpha*X1^2 + beta*X2^2
        return quadratic2(inplace, X1, X2, alpha, 0, beta, 0, 0, 0);
    }
    
    public Tensor sqsub(boolean inplace, Tensor X1, Tensor X2) {//X1^2 - X2^2
        return quadratic2(inplace, X1, X2, 1.0f, 0, -1.0f, 0, 0, 0); 
    }
    public Tensor sqsub(boolean inplace, Tensor X1, Tensor X2, float alpha, float beta) {//alpha*X1^2 - beta*X2^2
        return quadratic2(inplace, X1, X2, alpha, 0, -beta, 0, 0, 0);
    }
    
    public Tensor quadratic2(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return quadratic2(inplace, true, X1, X2, k11, k12, k22, k1, k2, C);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(check) {//Y = k11*X1^2 + k12*X1*X2 + k22*X2^2 + k1*X1 + k2*X2 + C
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.quadratic2_2D(Y.address, 
                X1.address, X2.address,
                k11, k12, k22,
                k1, k2, C, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public Tensor quadratic2_deltaX1(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,
            float k11, float k12, float k1) {//deltaX1 = deltaY * (k11*2*X1 + k12*X2 + k1)
        //deltaX1 = (2*k11) * (deltaY * X1) + k12 * (deltaY * X2) + k1*deltaY
        //(1) k11 = 0: deltaX1 = k12 * (deltaY * X2) + k1*deltaY
        //(2) k12 = 0: deltaX1 = (2*k11) * (deltaY * X1) + k1*deltaY
        if(k11 == 0.0f) return quadratic2(inplace, deltaY, X2, 0, k12,   0, k1, 0, 0);
        if(k12 == 0.0f) return quadratic2(inplace, deltaY, X1, 0, 2*k11, 0, k1, 0, 0);
        return mul_linear2(inplace, deltaY, X1, X2, 2*k11, k12, k1);
    }
    
    public Tensor quadratic2_deltaX2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,
            float k22, float k12, float k2) {//(2) deltaX2 = deltaY * (k22*2*X2 + k12*X1 + k2)
        //deltaX2 = (2*k22) * (deltaY * X2) + k12 * (deltaY * X1) + k2*deltaY
        //(1) k22 = 0: deltaX2 = k12 * (deltaY * X1) + k2*deltaY
        //(2) k12 = 0: deltaX2 = (2*k22) * (deltaY * X2) + k2*deltaY
        if(k22 == 0.0f) return quadratic2(inplace, deltaY, X1, 0, k12,   0, k2, 0, 0);
        if(k12 == 0.0f) return quadratic2(inplace, deltaY, X2, 0, 2*k22, 0, k2, 0, 0);
        return mul_linear2(inplace, deltaY, X2, X1, 2*k22, k12, k2);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] quadratic2_deltaX(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,
            float k11, float k12, float k22,
            float k1, float k2)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.quadratic2_2D_deltaX(
                deltaX1.c().address,//(1) deltaX1 = deltaY * (k11*2*X1 + k12*X2 + k1)
                deltaX2.c().address,//(2) deltaX2 = deltaY * (k22*2*X2 + k12*X1 + k2)
                deltaY.address,
                X1.address, X2.address,
                k11, k12, k22, 
                k1, k2, 
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2_row">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor mul_row(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_row(inplace, X1, X2, 0, 1.0f, 0, 0, 0, 0); }
    public Tensor mul_row(boolean inplace, float alpha, Tensor X1, Tensor X2) { 
        return quadratic2_row(inplace, X1, X2, 0, alpha, 0, 0, 0, 0); 
    }
    public Tensor sqadd_row(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_row(inplace, X1, X2, 1.0f, 0, 1.0f, 0, 0, 0); }
    public Tensor sqadd_row(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) { 
        return quadratic2_row(inplace, X1, X2, alpha, 0, beta, 0, 0, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2_row(boolean inplace, Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_row(X1, "X1", X2, "X2"); 
        }

        Tensor Y = (inplace? X1 : empty(X1.dim).c());
        Syncer sc = core.quadratic2_2D_row(Y.address,
                X1.address,
                X2.address, X2.lengthv, //X2.lengthv = row_lengthv
                k11, k12, k22,
                k1, k2, C, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public Tensor quadratic2_row_deltaX1(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,
            float k11, float k12, float k1) { 
        //deltaX1 = (2*k11) * (deltaY * X1) + k12 * (deltaY * X2) + k1*deltaY
        //(1) k11 = 0: deltaX1 = k12 * (deltaY * X2) + k1*deltaY
        //(2) k12 = 0: deltaX1 = (2*k11) * (deltaY * X1) + k1*deltaY
        if(k11 == 0.0f) return quadratic2_row(inplace, deltaY, X2, 0, k12,   0, k1, 0, 0);
        if(k12 == 0.0f) return quadratic2_row(inplace, deltaY, X1, 0, 2*k11, 0, k1, 0, 0);
        
        Tensor deri = linear2_row(false, X1, X2, 2*k11, k12, k1).c();//(k11*2*X1 + k12*X2 + k1)
        Tensor deltaX = mul(inplace, deltaY, deri);//inplace: deltaY -> deltaX
        if(sync) deri.delete(); else deltaX.dual(()-> {deri.delete();});
        return deltaX;
    }
    
    public Tensor quadratic2_row_deltaX2(Tensor deltaY, 
            Tensor X1, Tensor X2, int row_length,
            float k22, float k12, float k2) {
        //deltaX2 = (2*k22) * (deltaY * X2) + k12 * (deltaY * X1) + k2*deltaY
        //(1) k22 = 0: deltaX2 = k12 * (deltaY * X1) + k2*deltaY
        //(2) k12 = 0: deltaX2 = (2*k22) * (deltaY * X2) + k2*deltaY
        if(k22 == 0.0f) return field_quadratic2(deltaY, X1, row_length, 0, k12,   0, k2, 0, 0);
        if(k12 == 0.0f) return field_quadratic2(deltaY, X2, row_length, 0, 2*k22, 0, 0, k2, 0);
        
        Tensor deri = linear2_row(false, X2, X1, 2*k22, k12, k2).c();//(k22*2*X2 + k12*X1 + k2)
        Tensor deltaX = field_mul(deltaY, deri, X2.length);//row_length = X2.length
        if(sync) deri.delete(); else deltaX.dual(()-> { deri.delete(); });
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2_center">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor mul_center(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_center(inplace, X1, X2, -1, 0, 1.0f, 0, 0, 0, 0); }
    public Tensor mul_center(boolean inplace, float alpha, Tensor X1, Tensor X2) {//alpha * X1 * XX2
        return quadratic2_center(inplace, X1, X2, -1, 0, alpha, 0, 0, 0, 0);
    }
    
    public Tensor sqadd_center(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_center(inplace, X1, X2, -1, 1.0f, 0, 1.0f, 0, 0, 0); }
    public Tensor sqadd_center(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {//alpha*X1^2 + beta*X2^2
        return quadratic2_center(inplace, X1, X2, -1, alpha, 0, beta, 0, 0, 0);
    }
    
    public Tensor sqsub_center(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_center(inplace, X1, X2, -1, 1.0f, 0, -1.0f, 0, 0, 0); }
    public Tensor sqsub_center(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {//X1^2 + X2^2
        return quadratic2_center(inplace, X1, X2, -1, alpha, 0, -beta, 0, 0, 0);
    }
    
    public Tensor quadratic2_center(boolean inplace, Tensor X1, Tensor X2,
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return quadratic2_center(inplace, X1, X2,  -1, k11, k12, k22,  k1, k2, C);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2_center(boolean inplace, 
            Tensor X1, Tensor X2, int dim2,
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(dim2 == -1) dim2 = X1.lastDim();
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_center(X1, "X1", X2, "X2", dim2);
        }
        
        Tensor Y = (inplace? X1 : empty(X1.dim));
        int dim1 = X1.length / X2.length;//[dim0, dim1, dim2] / [dim0, dim2]
        int dim0 = X2.length / dim2;//[dim0, dim2] / dim2
        
        Syncer sc = core.quadratic2_2D_center(Y.c().address, 
                X1.address, X2.address,
                k11, k12, k22, 
                k1, k2, C, 
                dim0, dim1, dim2,
                X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public Tensor[] quadratic2_center_deltaX(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2, int dim2, 
            float k11, float k12, float k22,
            float k1, float k2, float C) 
    {
        if(dim2 == -1) dim2 = X1.lastDim();
        if(check) {//X1[dim0, dim1, dim2], X2[dim0, dim2]
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); require_dtype(deltaY, "deltaY");
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 3);
            equals_valueStructure(X1, "X1", deltaY, "deltaY");
            check_center(X1, "X1", X2, "X2", dim2);
        }
        
        int dim1 = X1.length / X2.length;
        int dim0 = X2.length / dim2;//X2.length = dim0 * dim2
        
        Tensor deltaX1 = (inplace ? deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        Syncer sc = core.quadratic2_2D_center_deltaX(
                deltaX1.c().address,//result0
                deltaX2.c().address,//result1
                deltaY.address, 
                X1.address, X2.address,
                k11, k12, k22, 
                k1, k2, C,
                dim0, dim1, dim2, 
                deltaY.lastDim());
        if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc);}
        return new Tensor[] { deltaX1, deltaX2 };
    }
    
    public Tensor quadratic2_center_deltaX1(boolean inplace, Tensor deltaY, 
            Tensor X1, Tensor X2, int dim2,
            float k11, float k12, float k1) {
        //deltaX1 = (2*k11) * (deltaY * X1) + k12 * (deltaY * X2) + k1*deltaY
        //(1) k11 = 0: deltaX1 = k12 * (deltaY * X2) + k1*deltaY
        //(2) k12 = 0: deltaX1 = (2*k11) * (deltaY * X1) + k1*deltaY
        if(k11 == 0.0f) return quadratic2_center(inplace, X1, X2, dim2, 0, k12,   0, k1, 0, 0);
        if(k12 == 0.0f) return quadratic2_center(inplace, X1, X2, dim2, 0, 2*k11, 0, k1, 0, 0);
        
        Tensor deri = linear2_center(false, X1, X2, 2*k11, k12, k1).c();//(k11*2*X1 + k12*X2 + k1)
        Tensor deltaX = mul(inplace, deltaY, deri);//inplace: deltaY -> deltaX
        if(sync) deri.delete(); else deltaX.dual(()-> {deri.delete();});
        return deltaX;
    }
    
    public Tensor quadratic2_center_deltaX2(Tensor deltaY, 
            Tensor X1, Tensor X2, int dim2,
            float k22, float k12, float k2) 
    {
        if(dim2 == -1) dim2 = X1.lastDim();
        int dim0 = X2.length / dim2;//X2.length = dim0 * dim2
        
        //deltaX2 = (2*k22) * (deltaY * X2) + k12 * (deltaY * X1) + k2*deltaY
        //(1) k22 = 0: deltaX2 = k12 * (deltaY * X1) + k2*deltaY
        //(2) k12 = 0: deltaX2 = (2*k22) * (deltaY * X2) + k2*deltaY
        if(k22 == 0.0f) return center_quadratic2(deltaY, X1, dim0, dim2, 0, k12,   0, k2, 0, 0);
        if(k12 == 0.0f) return center_quadratic2(deltaY, X2, dim0, dim2, 0, 2*k22, 0, k2, 0, 0);
        
        Tensor deri = linear2_center(false, X1, X2, 2*k22, k12, k2).c();//(k22*2*X1 + k12*X2 + k2)
        Tensor deltaX = center_mul(deltaY, deri, dim0, dim2);//deltaY * deri
        if(sync) deri.delete(); else deltaX.dual(()-> { deri.delete(); });
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_field">
    public Tensor mul_field(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_field(inplace, X1, X2, 0, 1.0f, 0, 0, 0, 0); }
    public Tensor mul_field(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return quadratic2_field(inplace, X1, X2, 0, alpha, 0, 0, 0, 0);
    }
    public Tensor sqadd_field(boolean inplace, Tensor X1, Tensor X2) { return quadratic2_field(inplace, X1, X2, 1.0f, 0, 1.0f, 0, 0, 0); }
    public Tensor sqadd_field(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return quadratic2_field(inplace, X1, X2, alpha, 0, beta, 0, 0, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2_field(boolean inplace, Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_field(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.quadratic2_2D_field(Y.address, 
                X1.address,
                X2.address, X2.length,//X2.length = field_length
                k11, k12, k22,
                k1, k2, C,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic_summary">
    public Tensor sqsum(boolean inplace, Collection<Tensor> Xs) {
        return Engine.this.quadratic_sum(inplace, 1.0f, 0, 0, Xs);//Y = sum(Xs[i]^2)
    }
    public Tensor sqsum(boolean inplace, float alpha, Collection<Tensor> Xs) {
        return Engine.this.quadratic_sum(inplace, alpha, 0, 0, Xs);//Y = alpha*sum(Xs[i]^2)
    }
    
    public Tensor quadratic_sum(boolean inplace,//inplace: Xs[0]
            float alpha, float beta, float gamma, Collection<Tensor> Xs) 
    {
        int Xs_size = Xs.size();
        Iterator<Tensor> iter = Xs.iterator(); Tensor X0 = iter.next();
        if(Xs_size == 1) { return quadratic(inplace, X0, alpha, beta, gamma); }
        if(check) { require_dtype(Xs, "Xs"); equals_valueStructure(X0, "Xs[0]", Xs, "Xs");  }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs_size]; int index = 0;
        for(Tensor X : Xs) addrs[index++] = X.address;
        
        Syncer sc = core.quadratic_summary2D(Y.c().address, 
                alpha, beta, gamma, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor sqsum(boolean inplace, Tensor... Xs) {
        return quadratic_sum(inplace, 1.0f, 0, 0, Xs);//Y = sum(Xs[i]^2)
    }
    public Tensor sqsum(boolean inplace, float alpha, Tensor... Xs) {
        return quadratic_sum(inplace, alpha, 0, 0, Xs);//Y = alpha*sum(Xs[i]^2)
    }
    public Tensor quadratic_sum(boolean inplace,//inplace: Xs[0]
            float alpha, float beta, float gamma, Tensor... Xs)
    {
        Tensor X0 = Xs[0];
        if(Xs.length == 1) { return quadratic(inplace, X0, alpha, beta, gamma); }
        if(check) { require_dtype(Xs, "Xs"); equals_valueStructure(X0, "Xs[0]", Xs, "Xs");  }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs.length]; 
        for(int i=0; i<Xs.length; i++) addrs[i] = Xs[i].address;
        
        Syncer sc = core.quadratic_summary2D(Y.c().address, 
                alpha, beta, gamma, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_iteration">
    public Tensor addSquare_iteration(boolean inplace, Tensor X1, Collection<Tensor> X2) {
        return quadratic2_iteration(inplace, 0, 0, 1.0f, 1.0f, 0, 0, X1, X2);
    }
    public Tensor addSquare_iteration(boolean inplace, float alpha, float beta, Tensor X1, Collection<Tensor> X2) {
        return quadratic2_iteration(inplace, 0, 0, beta, alpha, 0, 0, X1, X2);
    }
    
    public Tensor quadratic2_iteration(boolean inplace,//inplace: X1
            float k11, float k12, float k22,
            float k1, float k2, float C,
            Tensor X1, Collection<Tensor> X2) 
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); 
            equals_valueStructure(X1, "X1", X2, "X2");     
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim));
        long[] addrs = new long[X2.size()]; int index = 0;
        for(Tensor X : X2) addrs[index++] = X.address;
        
        Syncer sc = core.quadratic2_iteration2D(Y.address,
                X1.address, addrs,
                k11, k12, k22, k1, k2, C,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor addSquare_iteration(boolean inplace, Tensor X1, Tensor... X2) {
        return quadratic2_iteration(inplace, 0, 0, 1.0f, 1.0f, 0, 0, X1, X2);
    }
    public Tensor addSquare_iteration(boolean inplace, float alpha, float beta, Tensor X1, Tensor... X2) {
        return quadratic2_iteration(inplace, 0, 0, beta, alpha, 0, 0, X1, X2);
    }
    
    public Tensor quadratic2_iteration(boolean inplace,//inplace: X1
            float k11, float k12, float k22,
            float k1, float k2, float C,
            Tensor X1, Tensor... X2) 
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); 
            equals_valueStructure(X1, "X1", X2, "X2");     
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim));
        long[] addrs = new long[X2.length]; int index = 0;
        for(Tensor X : X2) addrs[index++] = X.address;
        
        Syncer sc = core.quadratic2_iteration2D(Y.address,
                X1.address, addrs,
                k11, k12, k22, k1, k2, C,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="rpl, div, linear2_div"> 
    //<editor-fold defaultstate="collapsed" desc="BP: rpl">
    public Tensor rpl(boolean inplace, Tensor X) { return rpl(inplace, 1.0f, X, 0.0f, 0.0f); }
    public Tensor rpl(boolean inplace, float alpha, Tensor X) { return rpl(inplace, alpha, X, 0.0f, 0.0f); }
    @Passed("CudaFloat32EngieBase")// alpha / (X + beta) + gamma
    public Tensor rpl(boolean inplace, float alpha, Tensor X, float beta, float gamma) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.rpl2D(Y.address, 
                alpha, X.address, beta, gamma,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor rpl_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha, float gamma){
        if(check) { 
            require_dtype(Y, "Y"); require_dtype(deltaY, "deltaY");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.rpl2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, gamma, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor div(boolean inplace, Tensor X1, Tensor X2) { return div(inplace, 1.0f, X1, 0, 1.0f, X2, 0, 0); }
    public Tensor div(boolean inplace, float alpha, Tensor X1, Tensor X2) {//alpha * X1 / X2
        return div(inplace, alpha, X1, 0, 1.0f, X2, 0, 0);
    }
    public Tensor div(boolean inplace,//default likeX1
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma) {
        return div(inplace, true, alpha1, X1, beta1, alpha2, X2, beta2, gamma);
    }
    
    @Passed("CudaFloat32EngieBase")//(alpha1*X1 + beta1) / (alpha1*X2 + beta2) + gamma
    public Tensor div(boolean inplace, boolean likeX1,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma)
    {
        if(check) {//(alpha1*X1 + beta1) / (alpha2*X2 + beta) + gamma
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.div2D(Y.address, 
                alpha1, X1.address, beta1,
                alpha2, X2.address, beta2,
                gamma, X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public Tensor div_deltaX1(boolean inplace, Tensor deltaY,
            Tensor X2, float alpha1, float alpha2, float beta2) {//deltaX1 = (a1*deltaY) / (a2*X2 + b2)
        return div(inplace, alpha1, deltaY, 0, alpha2, X2, beta2, 0);
    }
    
    //deltaX2 = (deltaY * -a2) * (a1*X1 + b1) / { (a2*X2 + b2)^2 }  
    public Tensor div_deltaX2(boolean inplace, Tensor deltaY, 
            Tensor X1, float alpha1, float beta1,
            Tensor X2, float alpha2, float beta2) {
        return mul_squareDiv(inplace, -alpha2, X1, 0, alpha1, X1, beta1, alpha2, X2, beta2, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] div_deltaX(boolean inplace, Tensor deltaY, 
            Tensor X1, float alpha1, float beta1,
            Tensor X2, float alpha2, float beta2)
    {
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(deltaY, "deltaY", X1, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
        
        Tensor deltaX1 = (inplace? deltaY : this.empty(X1.dim));
        Tensor deltaX2 = (inplace? deltaY : this.empty(X2.dim));
        
        Syncer sc = core.div2D_deltaX(
                deltaX1.c().address, //(1) deltaX1 = deltaY * a1 / (a2*X2 + b2) = (a1*deltaY) / (a2*X2 + b2)
                deltaX2.c().address, //(2) deltaX2 = deltaY * -a2 * (a1*X1 + b1) / { (a2*X2 + b2)^2 }  
                deltaY.address,
                X1.address, alpha1, beta1, 
                X2.address, alpha2, beta2, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else {
            deltaX1.setSyncer(sc);
            deltaX2.setSyncer(sc);
        }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div_row">
    public Tensor div_row(boolean inplace, Tensor X1, Tensor X2) {
        return div_row(inplace, 1.0f, X1, 0, 1.0f, X2, 0, 0);
    }
    public Tensor div_row(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return div_row(inplace, alpha, X1, 0, 1.0f, X2, 0, 0);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor div_row(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_row(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.div2D_row(Y.address, 
                alpha1, X1.address, beta1, 
                alpha2, X2.address, beta2, 
                gamma, X2.lengthv,//X2.lengthv = row_lengthv
                X1.lengthv, X1.lastDim());
         if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div_field">
    public Tensor div_field(boolean inplace, Tensor X1, Tensor X2) {
        return div_field(inplace, 1.0f, X1, 0, 1.0f, X2, 0, 0);
    }
    public Tensor div_field(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return div_field(inplace, alpha, X1, 0, 1.0f, X2, 0, 0);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor div_field(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            check_field(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.div2D_field(Y.address, 
                alpha1, X1.address, beta1, 
                alpha2, X2.address, beta2, 
                gamma, X2.length, //X2.length = field_length
                X1.lengthv, X1.lastDim());
         if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_div_row">
    public Tensor normalize_row(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
        return linear2_div_row(inplace, X, X_std, X_mean, 1.0f, -1.0f, 0.0f, eps);//(X1 - mean) / (std + eps)
    }
    public Tensor sub_div_row(boolean inplace, Tensor X1, Tensor X2, Tensor X3) {
        return linear2_div_row(inplace, X1, X2, X3, 1.0f, -1.0f, 0.0f, 0.0f);//(X1 - X2) / X3
    }
    public Tensor add_div_row(boolean inplace, Tensor X1, Tensor X2, Tensor X3) {
        return linear2_div_row(inplace, X1, X2, X3, 1.0f, 1.0f, 0.0f, 0.0f);//(X1 + X2) / X3
    }
    
    @Passed("CudaFloat32EngieBase") //(alpha*X1 + beta*X2 + gamma) / (X3 + delta)
    public Tensor linear2_div_row(boolean inplace, Tensor X1, Tensor X2, Tensor X3,
            float alpha, float beta, float gamma, float delta)
    {
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); require_dtype(X3, "X3");
            check_row(X1, "X1", X2, "X2");
            check_row(X1, "X1", X3, "X3");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.linear2_div2D_row(Y.address, 
                X1.address,
                X2.address, 
                X3.address, X2.lengthv,//X3.lengthv = X2.lengthv = row_lengthv
                alpha, beta, gamma, delta,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_div_field">
    public Tensor normalize_field(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
        return linear2_div_field(inplace, X, X_mean, X_std, 1.0f, -1.0f, 0.0f, eps);//(X1 - mean) / (std + eps)
    }
    public Tensor sub_div_field(boolean inplace, Tensor X1, Tensor X2, Tensor X3) {
        return linear2_div_field(inplace, X1, X2, X3, 1.0f, -1.0f, 0.0f, 0.0f);//(X1 - X2) / X3
    }
    public Tensor add_div_field(boolean inplace, Tensor X1, Tensor X2, Tensor X3) {
        return linear2_div_field(inplace, X1, X2, X3, 1.0f, 1.0f, 0.0f, 0.0f);//(X1 + X2) / X3
    }
    
    @Passed("CudaFloat32EngieBase")//(alpha*X1 + beta*X2 + gamma) / (X3 + delta)
    public Tensor linear2_div_field(boolean inplace, Tensor X1, Tensor X2, Tensor X3,
            float alpha, float beta, float gamma, float delta)
    {
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); require_dtype(X3, "X3");
            check_field(X1, "X1", X2, "X2");
            check_field(X1, "X1", X3, "X3");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.linear2_div2D_field(Y.address, 
                X1.address,
                X2.address, 
                X3.address, X2.length, //X3.length = X2.length = field_length
                alpha, beta, gamma, delta,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="mul_squareDiv">
    @Passed("CudaFloat32EngieBase")
    public Tensor mul_squareDiv(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float alpha3, Tensor X3, float beta3,
            float gamma)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2"); require_dtype(X3, "X3");
            equals_valueStructure(X1, "X1", X2, "X2"); 
            equals_valueStructure(X1, "X1", X3, "X3"); 
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.mul_squareDiv2D(Y.address, 
                alpha1, X1.address, beta1, 
                alpha2, X2.address, beta2, 
                alpha3, X3.address, beta3, 
                gamma, X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="clip, sign, ceil, floor, abs, sqrt">
    @Passed("CudaFloat32EngieBase")
    public Tensor sign(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sign2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor ceil(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.ceil2D(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor floor(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.floor2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor zero_nan(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.zero_nan2D(Y.address,
                X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: abs">
    public Tensor abs(boolean inplace, Tensor X) { return abs(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor abs(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.abs2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor abs_deltaX(boolean inplace, Tensor deltaY,
            Tensor X, float alpha, float beta)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X"); 
            equals_valueStructure(deltaY, "deltaY", X, "deltaX"); 
        }
            
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.abs2D_deltaX(deltaX.address,
                deltaY.address, 
                X.address, alpha, beta,
                deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqrt">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqrt(boolean inplace, float alpha, Tensor X, float beta)  {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sqrt2D(Y.address, alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
   
    @Passed("CudaFloat32EngieBase")//deltaX = 0.5 * alpha * Y * deltaX
    public Tensor sqrt_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        return div(inplace, 
                0.5f*alpha, deltaY, 0.0f, 
                1.0f, Y, 0.0f, 0.0f);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sqrt_quadratic2">
    public Tensor sqrt_quadratic2(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, 
            float k11, float k12 ,float k22, 
            float k1, float k2, float C) {
        return sqrt_quadratic2(inplace, true, X1, X2, k11, k12, k22, k1, k2, C);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqrt_quadratic2(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float k11, float k12 ,float k22, 
            float k1, float k2, float C)
    {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.sqrt_quadratic2_2D(Y.address,
                X1.address, X2.address,
                k11, k12, k22,
                k1, k2, C, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="minValue, max, clip"> 
    //<editor-fold defaultstate="collapsed" desc="BP: minValue">
    public Tensor min(boolean inplace, Tensor X, float vmin){ return min(inplace, 1, X, 0, vmin);}
    @Passed("CudaFloat32EngieBase")
    public Tensor min(boolean inplace, float alpha, Tensor X, float beta, float vmin) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.min2D(Y.address, 
                alpha, X.address, beta, vmin,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor min_deltaX_v1(boolean inplace, Tensor deltaY, 
            Tensor Y, float alpha, float vmin) {
        return linear_greater_switch_mul(inplace, false, //y' = alpha (Y < vmin -> -Y + vmin < 0)  
                -1.0f, Y, vmin,                          //y' = 0, otherwise
                deltaY, alpha, 0);//deltaX = deltaY * (Y - vmin < 0 ? alpha : 0)
    }
   
    public Tensor min_deltaX_v2(boolean inplace, Tensor deltaY, 
            Tensor X, float alpha, float beta, float vmin) {
        return linear_greater_switch_mul(inplace, false, //y' = alpha (alpha*X + beta < vmin -> -alpha*X + (vmin - beta) > 0) 
                -alpha, X, vmin - beta,                  //y' = 0, otherwise
                deltaY, alpha, 0);//deltaX = deltaY * (-alpha*X + (vmin - beta) > 0 ? alpha : 0)
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="min2">
    public Tensor min2(boolean inplace, Tensor X1, Tensor X2) { return min2(inplace, 1.0f, X1, 0.0f, 1.0f, X2, 0.0f); }
    public Tensor min2(boolean inplace,//default likeX1
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        return min2(inplace, true, alpha1, X1, beta1, alpha2, X2, beta2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor min2(boolean inplace, boolean likeX1, 
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.min2_2D(Y.address,
                alpha1, X1.address, beta1,
                alpha2, X2.address, beta2,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: max">
    public Tensor max(boolean inplace, Tensor X, float vmax) { return max(inplace, 1, X, 0, vmax); }
    @Passed("CudaFloat32EngieBase")
    public Tensor max(boolean inplace, float alpha, Tensor X, float beta, float vmax) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.max2D(Y.address,
                alpha, X.address, beta, vmax,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    //y' = alpha (Y > vmax -> Y - vmax > 0) 
    //y' = 0, otherwise
    //deltaX = deltaY * (Y - vmax > 0 ? alpha : 0)
    public Tensor max_deltaX_v1(boolean inplace, Tensor deltaY, 
            Tensor Y, float alpha, float vmax) {
        return linear_greater_switch_mul(inplace, false, 
                1.0f, Y, -vmax, 
                deltaY, alpha, 0);
    }
    
    //y' = alpha (alpha*X + beta > vmax -> alpha*X + (beta - vmax) > 0) 
    //y' = 0, otherwise
    //deltaX = deltaY * (alpha*X + (beta - vmax) > 0 ? alpha : 0)
    public Tensor max_deltaX_v2(boolean inplace, Tensor deltaY, 
            Tensor X, float alpha, float beta, float vmax) {
        return linear_greater_switch_mul(inplace, false, 
                alpha, X, beta - vmax, 
                deltaY, alpha, 0);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="max2">
    public Tensor max2(boolean inplace, Tensor X1, Tensor X2) { return max2(inplace, 1.0f, X1, 0.0f, 1.0f, X2, 0.0f); }
    public Tensor max2(boolean inplace,//default likeX1
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        return max2(inplace, true, alpha1, X1, beta1, alpha2, X2, beta2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor max2(boolean inplace, boolean likeX1, 
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = inplace? (likeX1? X1 : X2) : empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.max2_2D(Y.address,
                alpha1, X1.address, beta1,
                alpha2, X2.address, beta2,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: clip"> 
    public Tensor hard_sigmoid(boolean inplace, Tensor X) { return clip(inplace, 0.2f, X, 0.5f, 0.0f, 1.0f); }//clip(0.2*x + 0.5, 0, 1)
    public Tensor hard_sigmoid_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        return clip_deltaX_v1(inplace, deltaY, Y, 0.2f, 0.0f, 1.0f);//alpha = 0.2, vmin = 0, vmax = 1
    }
    public Tensor hard_sigmoid_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        return clip_deltaX_v2(inplace, deltaY, X, 0.2f, 0.5f, 0.0f, 1.0f);//alpha = 0.2, beta = 0.5, vmin = 0, vmax = 1
    }
    
    public Tensor relu6(boolean inplace, Tensor X) { return clip(inplace, 1.0f, X, 0.0f, 0.0f, 6.0f); }//clip(X, 0, 6)
    public Tensor relu6_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        return clip_deltaX_v1(inplace, deltaY, Y, 1.0f, 0.0f, 6.0f);//alpha = 1, vmin = 0, vmax = 6
    }
    public Tensor relu6_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {//alpha = 1, beta = 0, vmin = 0, vmax = 6
        return clip_deltaX_v2(inplace, deltaY, X, 1.0f, 0.0f, 0.0f, 6.0f);
    }
    
    public Tensor reluN(boolean inplace, Tensor X, float N) { return clip(inplace, 1.0f, X, 0.0f, 0.0f, N); }//clip(X, 0, M)
    public Tensor reluN_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y, float N) {
        return clip_deltaX_v1(inplace, deltaY, Y, 1.0f, 0.0f, N);//alpha = 1, vmin = 0, vmax = N
    }
    public Tensor reluN_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X, float N) {
        return clip_deltaX_v2(inplace, deltaY, X, 1.0f, 0.0f, 0.0f, N);//alpha = 1, beta = 0, vmin = 0, vmax = N
    }
    
    public Tensor clip(boolean inplace, Tensor X, float vmin, float vmax) { return clip(inplace, 1.0f, X, 0.0f, vmin, vmax); }
    @Passed("CudaFloat32EngieBase")
    public Tensor clip(boolean inplace, float alpha, Tensor X, float beta, float vmin, float vmax) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.clip2D(Y.address,
                alpha, X.address, beta, vmin, vmax,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor clip_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y, float alpha, float vmin, float vmax) {
        return linear_bound_switch_mul(inplace, false, //y' = alpha, { vmin < Y < vmax } 
                1.0f, Y, vmin, vmax,                   //y' = 0, otherwise
                deltaY, 0, alpha, 0);//deltaX = deltaY * (vmin < Y < vmax ? alpha : 0)
    }
    
    public Tensor clip_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X, float alpha, float beta, float vmin, float vmax) {
        return linear_bound_switch_mul(inplace, false, //y' = alpha, { vmin < alpha*X + beta < vmax } -> { vmin - beta < alpha*X < vmax - beta } 
                alpha, X, vmin - beta, vmax - beta,    //y' = 0, otherwise
                deltaY, 0, alpha, 0);//deltaX = deltaY * ({ vmin - beta < alpha*X < vmax - beta }  ? alpha : 0)
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    //<editor-fold defaultstate="collapsed" desc="BP: exp">
    public Tensor exp(boolean inplace, Tensor X) { return exp(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor exp(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.exp2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor exp_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        return mul(inplace, alpha, deltaY, Y);//deltaX = alpha * deltaY * Y
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: log">
    public Tensor log(boolean inplace, Tensor X) { return log(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor log(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.log2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor log_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        if(check) { 
            require_dtype(Y, "Y"); require_dtype(deltaY, "deltaY");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.log2D_deltaX(deltaX.address,
                deltaY.address, 
                Y.address, alpha,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    @Passed("CudaFloat32EngieBase")
    public Tensor relu(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.relu2D(Y.address, X.address,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor relu_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.relu2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor relu_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.relu2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    public Tensor leakyRelu(boolean inplace, Tensor X) { return leakyRelu(inplace, X, 0.01f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor leakyRelu(boolean inplace, Tensor X, float negative_slope) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.leakyRelu2D(Y.address, 
                X.address, negative_slope,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor leakyRelu_deltaX_v1(boolean inplace, Tensor deltaY, 
            Tensor Y, float negative_slope) 
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.leakyRelu2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address, negative_slope, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor leakyRelu_deltaX_v2(boolean inplace, Tensor deltaY, 
            Tensor X, float negative_slope)
    {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X,"X");
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.leakyRelu2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address, negative_slope,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    public Tensor elu(boolean inplace, Tensor X) { return elu(inplace, X, 1.0f, 0.01f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor elu(boolean inplace, Tensor X, float alpha, float negative_slope) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.elu2D(Y.address,
                X.address, alpha, negative_slope, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor elu_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y, float alpha, float negative_slope)
    {
       if(check) { 
           require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
           equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
       }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.elu2D_deltaX_v1(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, negative_slope,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor elu_deltaX_v2(boolean inplace, Tensor deltaY, 
            Tensor X, float alpha, float negative_slope)
    {
        if(check) { 
            require_dtype(deltaY, "deltaU"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.elu2D_deltaX_v2(deltaX.address,
                deltaY.address, 
                X.address, alpha, negative_slope, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    @Passed("CudaFloat32EngieBase")
    public Tensor softplus(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.softPlus2D(Y.address,
                X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor softplus_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.softPlus2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor softplus_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.softPlus2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: gelu">
    @Passed("CudaFloat32EngieBase")
    public Tensor gelu(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.gelu2D(Y.address, X.address,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor gelu_deltaX(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.gelu2D_deltaX(deltaX.address,
                deltaY.address,
                X.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_leayRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward_propagation">
    public Tensor add_relu(boolean inplace, Tensor X1, Tensor X2) {//relu(X1 + X2)
        return linear2_relu(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor linear2_relu(boolean inplace,//default likeX1 
            Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return linear2_relu(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//relu(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_relu(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_relu2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor add_leakyRelu(boolean inplace, Tensor X1, Tensor X2, float k) { 
        return linear2_leakyRelu(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f, k);
    }
    public Tensor linear2_leakyRelu(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma, float k) {
        return linear2_leakyRelu(inplace, true, X1, X2, alpha, beta, gamma, k);
    }
    @Passed("CudaFloat32EngieBase")//LeakyRelu(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_leakyRelu(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma, float k)
    {
        if(check) {
            require_dtype(X1); require_dtype(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_leakyRelu2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma, k,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    public Tensor[] linear2_relu_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y, float alpha, float beta) {//V1: holdY(), Y is not changed
        return linear2_leakyRelu_deltaX_v1(inplace, deltaY, Y, alpha, beta, 0.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_leakyRelu_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y,//V1: holdY(), Y is not changed
            float alpha, float beta, float k)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(Y.dim));
        Tensor deltaX2 = this.empty(Y.dim);
        
        Syncer sc = core.linear2_leakyRelu2D_deltaX_v1(
                deltaX1.c().address,
                deltaX2.c().address, 
                deltaY.address, 
                Y.address, alpha, beta, k,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    
    public Tensor[] linear2_relu_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma) {
        return linear2_leakyRelu_deltaX_v2(inplace, deltaY, X1, X2, alpha, beta, gamma, 0.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_leakyRelu_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma, float k)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X1); require_dtype(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.linear2_leakyRelu2D_deltaX_v2(
                deltaX1.c().address, 
                deltaX2.c().address,
                deltaY.address, 
                X1.address, X2.address, 
                alpha, beta, gamma, k, 
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_elu">
    //<editor-fold defaultstate="collapsed" desc="forward_propagation">
    public Tensor add_elu(boolean inplace, Tensor X1, Tensor X2, float k) { 
        return linear2_elu(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f, 1.0f, k);
    }
    public Tensor linear2_elu(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, float alpha, float beta, float gamma, 
            float theta, float k) {
        return linear2_elu(inplace, true, X1, X2, alpha, beta, gamma, theta, k);
    }
    @Passed("CudaFloat32EngieBase")//elu(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_elu(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma, 
            float theta, float k) 
    {
        if(check) {
            require_dtype(X1); require_dtype(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_elu2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma, 
                theta, k,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_elu_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y,//V1: holdY(), Y is not changed
            float alpha, float beta,
            float theta, float k)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(Y.dim));
        Tensor deltaX2 = this.empty(Y.dim);
        
        Syncer sc = core.linear2_elu2D_deltaX_v1(
                deltaX1.c().address,
                deltaX2.c().address, 
                deltaY.address, 
                Y.address, alpha, beta,
                theta, k,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_elu_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma, 
            float theta, float k)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X1); require_dtype(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.linear2_elu2D_deltaX_v2(
                deltaX1.c().address, 
                deltaX2.c().address,
                deltaY.address, 
                X1.address, X2.address, 
                alpha, beta, gamma,
                theta, k, 
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward_propagation">
    public Tensor add_softplus(boolean inplace, Tensor X1, Tensor X2) { 
        return linear2_softplus(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor linear2_softplus(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return linear2_softplus(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//softplus(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_softplus(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2,
            float alpha, float beta, float gamma) 
    {
        if(check) {
            require_dtype(X1); require_dtype(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_softplus2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_softplus_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y,//V1: holdY(), Y is not changed
            float alpha, float beta)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(Y.dim));
        Tensor deltaX2 = this.empty(Y.dim);
        
        Syncer sc = core.linear2_softplus2D_deltaX_v1(
                deltaX1.c().address,
                deltaX2.c().address, 
                deltaY.address, 
                Y.address, alpha, beta,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_softplus_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X1); require_dtype(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.linear2_softplus2D_deltaX_v2(
                deltaX1.c().address, 
                deltaX2.c().address,
                deltaY.address, 
                X1.address, X2.address, 
                alpha, beta, gamma,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward_propagation">
    public Tensor add_gelu(boolean inplace, Tensor X1, Tensor X2) { 
        return linear2_gelu(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor linear2_gelu(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return linear2_gelu(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//gelu(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_gelu(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2,
            float alpha, float beta, float gamma)
    {
        if (check) {
            require_dtype(X1); require_dtype(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_gelu2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if (sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_gelu_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X1); require_dtype(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace ? deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.linear2_gelu2D_deltaX_v2(
                deltaX1.c().address, 
                deltaX2.c().address,
                deltaY.address, 
                X1.address, X2.address, 
                alpha, beta, gamma,
                deltaY.lengthv, deltaY.lastDim());
        if (sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    @Passed("CudaFloat32EngieBase")
    public Tensor tanh(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.tanh2D(Y.address,
                X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tanh_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.tanh2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tanh_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.tanh2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.sigmoid2D(Y.address, X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sigmoid2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sigmoid2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    public Tensor softmax(boolean inplace, Tensor X) { return softmax(inplace, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax(boolean inplace, Tensor X, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dtype(X, "X");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            if(X.length % features != 0) throw new IllegalArgumentException(String.format(
                    "X.length { got %d } %% features { got %d } != 0",
                    X.length, features));
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.softmax2D(Y.address,
                X.address, features,
                X.lengthv, X.lastDim());//X.lastDim = width = mem_width
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y) {
        return softmax_deltaX(inplace, deltaY, Y, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y, int features) {
        if(features == -1) features = Y.lastDim();
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            if(deltaY.length % features != 0) throw new IllegalArgumentException(String.format(
                    "deltaY.length { got %d } %% features { got %d } != 0", 
                    deltaY.length, features));
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.softmax2D_deltaX(deltaX.address, 
                deltaY.address,
                Y.address, features,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: log_softmax">
    public Tensor log_softmax(boolean inplace, Tensor X) { return log_softmax(inplace, X, - 1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor log_softmax(boolean inplace, Tensor X, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dtype(X, "X");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            if(X.length % features != 0) throw new IllegalArgumentException(String.format(
                    "X.length { got %d } %% features { got %d } != 0",
                    X.length, features));
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.logsoftmax2D(Y.address,
                X.address, features,
                X.lengthv, X.lastDim());//X.lastDim = width = mem_width
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor log_softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y) {
        return log_softmax_deltaX(inplace, deltaY, Y, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor log_softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y, int features) {
        if(features == -1) features = Y.lastDim();
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            if(deltaY.length % features != 0) throw new IllegalArgumentException(String.format(
                    "deltaY.length { got %d } %% features { got %d } != 0", 
                    deltaY.length, features));
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.logsoftmax2D_deltaX(deltaX.address, 
                deltaY.address,
                Y.address, features,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward_propagation">
    public Tensor add_sigmoid(boolean inplace, Tensor X1, Tensor X2) { 
        return linear2_sigmoid(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor linear2_sigmoid(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return linear2_sigmoid(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//sigmoid(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_sigmoid(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(X1); require_dtype(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_sigmoid2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_sigmoid_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y,//V1: holdY(), Y is not changed
            float alpha, float beta)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(Y.dim));
        Tensor deltaX2 = this.empty(Y.dim);
        
        Syncer sc = core.linear2_sigmoid2D_deltaX_v1(
                deltaX1.c().address,
                deltaX2.c().address, 
                deltaY.address, 
                Y.address, alpha, beta,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_sigmoid_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X1); require_dtype(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.linear2_sigmoid2D_deltaX_v2(
                deltaX1.c().address, 
                deltaX2.c().address,
                deltaY.address, 
                X1.address, X2.address, 
                alpha, beta, gamma,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: linear2_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward_propagation">
    public Tensor add_tanh(boolean inplace, Tensor X1, Tensor X2) { 
        return linear2_tanh(inplace, true, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor linear2_tanh(boolean inplace,//default likeX1
            Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return linear2_tanh(inplace, true, X1, X2, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//tanh(alpha*X1 + beta*X2 + gamma)
    public Tensor linear2_tanh(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2,
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(X1); require_dtype(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        Tensor Y = (inplace? (likeX1? X1 : X2) : this.empty(likeX1? X1.dim : X2.dim).c());
        Syncer sc = core.linear2_tanh2D(Y.address,
                X1.address, X2.address, 
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_tanh_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y,//V1: holdY(), Y is not changed
            float alpha, float beta)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(Y.dim));
        Tensor deltaX2 = this.empty(Y.dim);
        
        Syncer sc = core.linear2_tanh2D_deltaX_v1(
                deltaX1.c().address,
                deltaX2.c().address, 
                deltaY.address, 
                Y.address, alpha, beta,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear2_tanh_deltaX_v2(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,//V2: holdX(), {X1, X2} is not changed
            float alpha, float beta, float gamma)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X1); require_dtype(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
         
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.linear2_tanh2D_deltaX_v2(
                deltaX1.c().address, 
                deltaX2.c().address,
                deltaY.address, 
                X1.address, X2.address, 
                alpha, beta, gamma,
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else { deltaX1.setSyncer(sc); deltaX2.setSyncer(sc); }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin, cos">
    public Tensor cos(boolean inplace, Tensor X) { return sin(inplace, 1.0f, X, HALF_PI); }
    public Tensor cos(boolean inplace, float alpha, Tensor X, float beta) {
        return sin(inplace, alpha, X, beta + HALF_PI);
    }
    public Tensor cos_deltaX(boolean inplace, Tensor deltaY, Tensor X, float alpha, float beta)  {
        return sin_deltaX(inplace, deltaY, X, alpha, beta + HALF_PI);
    }
    
    public Tensor sin(boolean inplace, Tensor X) { return sin(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor sin(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.sin2D(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sin_deltaX(boolean inplace, Tensor deltaY, Tensor X, float alpha, float beta) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sin2D_deltaX(deltaX.address, 
                deltaY.address, 
                X.address, alpha, beta, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tan, cot">
    public Tensor cot(boolean inplace, Tensor X) { return tan(inplace, -1.0f, X, - HALF_PI); }
    public Tensor cot(boolean inplace, float alpha, Tensor X, float beta) {
        return tan(inplace, -alpha, X, -beta - HALF_PI);
    }
    public Tensor cot_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        return tan_deltaX(inplace, deltaY, Y, -alpha);
    }
    
    public Tensor tan(boolean inplace, Tensor X) { return tan(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tan(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.tan2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tan_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.tan2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: csc, sec">
    public Tensor sec(boolean inplace, Tensor X) { return csc(inplace, 1.0f, X, HALF_PI); }
    public Tensor sec(boolean inplace, float alpha, Tensor X, float beta) {
        return csc(inplace, alpha, X, beta + HALF_PI);
    }
    public Tensor sec_deltaX(boolean inplace, Tensor deltaY, Tensor X, float alpha, float beta)  {
        return csc_deltaX(inplace, deltaY, X, alpha, beta + HALF_PI);
    }
    
    public Tensor csc(boolean inplace, Tensor X) { return csc(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor csc(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.csc2D(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor csc_deltaX(boolean inplace, Tensor deltaY, Tensor X, float alpha, float beta) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.csc2D_deltaX(deltaX.address, 
                deltaY.address, 
                X.address, alpha, beta, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin">
    //arccos(X) + arcsin(X) = 0.5pi
    public Tensor arcsin(boolean inplace, Tensor X) { return arcsin(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor arcsin(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); } 
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.arcsin2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor arcsin_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        if(check) { 
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.arcsin2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arctan2D">
    //arctan(X) + arccot(X) = 0.5pi
    public Tensor arctan(boolean inplace, Tensor X) { return arctan(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor arctan(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X); } 
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.arctan2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor arctan_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        if(check) { 
            require_dtype(deltaY); require_dtype(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.arctan2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin">
    @Passed("CudaFloat32EngieBase")
    public Tensor halfSin(boolean inplace, float Amp, float alpha, Tensor X, float beta) {
        if(check) { require_dtype(X, "X"); } 
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.halfSin2D(Y.address,
                Amp, alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor halfSin_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float Amp, float alpha){
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.halfSin2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, Amp, alpha, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss functions">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    public Tensor L1(Tensor Yh, Tensor Y) { return L1(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L1(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) {
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L1_2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor L1_deltaYh(Tensor Yh, Tensor Y)  { return L1_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L1_deltaYh(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L1_2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    public Tensor L2(Tensor Yh, Tensor Y) { return L2(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L2(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) {
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L2_2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor L2_deltaYh(Tensor Yh, Tensor Y)  { return L2_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L2_deltaYh(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L2_2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: smoothL1">
    public Tensor smoothL1(Tensor Yh, Tensor Y) { return smoothL1(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor smoothL1(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.smoothL1_2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor smoothL1_deltaYh(Tensor Yh, Tensor Y)  { return smoothL1_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor smoothL1_deltaYh(boolean dimLikeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor deltaYh = this.empty(dimLikeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.smoothL1_2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    public Tensor binaryCrossEntropy(Tensor Yh, Tensor Y) { 
        return binaryCrossEntropy(true, Yh, Y, 1.0f, 1.0f); 
    }
    public Tensor binaryCrossEntropy(Tensor Yh, Tensor Y, float alpha, float beta) {
        return binaryCrossEntropy(true, Yh, Y, alpha, beta);
    }
    public Tensor binaryCrossEntropy(boolean likeYh, Tensor Yh, Tensor Y) {
        return binaryCrossEntropy(likeYh, Yh, Y, 1.0f, 1.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor binaryCrossEntropy(boolean likeYh, Tensor Yh, Tensor Y, float alpha, float beta) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.binaryCrossEntropy2D(L.address, 
                Y.address, Yh.address,
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor binaryCrossEntropy_deltaYh(Tensor Yh, Tensor Y)  {
        return binaryCrossEntropy_deltaYh(true, Yh, Y, 1.0f, 1.0f);
    }
    public Tensor binaryCrossEntropy_deltaYh(Tensor Yh, Tensor Y, float alpha, float beta) {
        return binaryCrossEntropy_deltaYh(true, Yh, Y, alpha, beta);
    }
    public Tensor binaryCrossEntropy_deltaYh(boolean likeYh, Tensor Yh, Tensor Y)  {
        return binaryCrossEntropy_deltaYh(likeYh, Yh, Y, 1.0f, 1.0f);
    }      
    @Passed("CudaFloat32EngieBase")
    public Tensor binaryCrossEntropy_deltaYh(boolean likeYh, Tensor Yh, Tensor Y, float alpha, float beta) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.binaryCrossEntropy2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address, 
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    public Tensor sigmoid_binaryCrossEntropy(Tensor X, Tensor Y) {
        return sigmoid_binaryCrossEntropy(true, X, Y, 1.0f, 1.0f);
    }
    public Tensor sigmoid_binaryCrossEntropy(Tensor X, Tensor Y, float alpha, float beta) {
        return sigmoid_binaryCrossEntropy(true, X, Y, alpha, beta);
    }
    public Tensor sigmoid_binaryCrossEntropy(boolean likeX, Tensor X, Tensor Y) {
        return sigmoid_binaryCrossEntropy(likeX, X, Y, 1.0f, 1.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_binaryCrossEntropy(boolean likeX, Tensor X, Tensor Y, float alpha, float beta) {
        if(check) {
            require_dtype(X, "X"); require_dtype(Y, "Y");
            equals_valueStructure(X, "X", Y, "Y"); 
        }
        Tensor L = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.sigmoid_binaryCrossEntropy2D(L.address, 
                Y.address, X.address, 
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor sigmoid_binaryCrossEntropy_deltaX(Tensor X, Tensor Y)  {
        return sigmoid_binaryCrossEntropy_deltaX(true, X, Y, 1.0f, 1.0f);
    }
    public Tensor sigmoid_binaryCrossEntropy_deltaX(Tensor X, Tensor Y, float alpha, float beta)  {
        return sigmoid_binaryCrossEntropy_deltaX(true, X, Y, alpha, beta);
    }
    public Tensor sigmoid_binaryCrossEntropy_deltaX(boolean likeX, Tensor X, Tensor Y) {
        return sigmoid_binaryCrossEntropy_deltaX(likeX, X, Y, 1.0f, 1.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_binaryCrossEntropy_deltaX(boolean likeX, Tensor X, Tensor Y, float alpha, float beta) {
        if(check) { 
            require_dtype(X, "X"); require_dtype(Y, "Y");
            equals_valueStructure(X, "X", Y, "Y"); 
        }
        Tensor deltaX = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.sigmoid_binaryCrossEntropy2D_deltaX(deltaX.address,
                Y.address, X.address,
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    public Tensor crossEntropy(Tensor Yh, Tensor Y) { return crossEntropy(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor crossEntropy(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y");
        }
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.crossEntropy2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor crossEntropy_deltaYh(Tensor Yh, Tensor Y) { return crossEntropy_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor crossEntropy_deltaYh(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dtype(Yh, "Yh"); require_dtype(Y, "Y");
            equals_valueStructure(Yh, "Yh", Y, "Y");
        }
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.crossEntropy2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax_crossEntropy">
    public Tensor softmax_crossEntropy(Tensor X, Tensor Y) { return softmax_crossEntropy(true, X, Y, -1); }
    public Tensor softmax_crossEntropy(Tensor X, Tensor Y, int features) { return softmax_crossEntropy(true, X, Y, features);}
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_crossEntropy(boolean likeX, Tensor X, Tensor Y, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dtype(X, "X"); require_dtype(Y, "Y");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            if(X.length % features != 0) throw new IllegalArgumentException(String.format(
                    "X.length { got %d } %% features { got %d } != 0", 
                    X.length, features));
            equals_valueStructure(X, "X", Y, "Y");
        }
        
        Tensor L = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.softmax_crossEntropy2D(L.address, 
                Y.address, X.address, features,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
  
    public Tensor softmax_crossEntropy_deltaX(Tensor X, Tensor Y, int features) {
        return softmax_crossEntropy_deltaX(true, X, Y, features);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_crossEntropy_deltaX(boolean likeX, Tensor X, Tensor Y, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dtype(X, "X"); require_dtype(Y, "Y");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            if(X.length % features != 0) throw new IllegalArgumentException(String.format(
                    "X.length { got %d } %% features { got %d } != 0", 
                    X.length, features));
            equals_valueStructure(X, "X", Y, "Y");
        }
        
        Tensor deltaX = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.softmax_crossEntropy2D_deltaX(deltaX.address,
                Y.address, X.address, features,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    //<editor-fold defaultstate="collapsed" desc="softmax_crossEntropy_deltaX_naive">
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_crossEntropy_deltaX_naive(boolean likeX, Tensor X, Tensor Y, int row_length)
    {
        if(check) { 
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must > 1");
            equals_valueStructure(X, "X", Y, "Y");
        }
        
        Tensor deltaX = this.softmax(false, X, row_length).c();
        deltaX = this.add(true, 1.0f, deltaX, -1.0f, Y);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="affine_param_check">
    protected final void check_affine(Tensor X, Tensor A, Tensor B) {
        require_dtype(X, "X"); require_dtype(A, "A"); require_dtype(B, "B");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
        equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
        equals(A.length, "A.length", B.length, "B.length");
    }
    
    protected final void check_affine_fuction_deltaX_v1(Tensor deltaY, Tensor Y, Tensor A) {
        require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); require_dtype(A, "A");
        equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        equals(Y.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
    }
    
    protected final void check_affine_function_deltaX_v2(Tensor deltaY, Tensor X, Tensor A, Tensor B) {
        require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
        require_dtype(A, "A"); require_dtype(B, "B");
        equals_valueStructure(deltaY, "deltaY", X, "X"); 
        equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
        equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
    }
    
    protected final void check_affine_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); 
        require_dtype(A, "A"); require_dtype(B, "B");
        must_greater_equal(Y.ndim(), "Y.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", Y, "Y");
        equals(Y.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
        equals(Y.lastDim(), "Y.lastDim", B.lastDim(), "B.lastDim");
        equals(A.length, "A.length", B.length, "B.length");
    }
    
    protected final void check_affine_funcion_deltaAB_v2(Tensor deltaY, Tensor X, Tensor A, Tensor B) {
        require_dtype(deltaY, "deltaY"); require_dtype(X, "X"); 
        require_dtype(A, "A"); require_dtype(B, "B");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", X, "X");
        equals(X.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
        equals(X.lastDim(), "Y.lastDim", B.lastDim(), "B.lastDim");
        equals(A.length, "A.length", B.length, "B.length");
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchNorm_param_check">
    protected final void check_batchNorm(Tensor X, Tensor X_mean, Tensor X_var) {
        require_dtype(X, "X"); require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
        equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim()");
        equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
    }
    
    protected final void check_batchNorm(Tensor X, Tensor X_mean, Tensor X_var, Tensor A, Tensor B) {
        require_dtype(X, "X"); require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var");
        require_dtype(A, "A"); require_dtype(B, "B");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
        equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim");
        equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
        equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
        equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        equals(X_mean.length, "X_mean.length", A.length, "A.length");
        equals(X_mean.length, "X_mean.length", B.length, "B.length");
    }
    
    protected final void check_batchNorm_deltaX_v1(Tensor deltaY, Tensor Y, Tensor X_var) {
        require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y"); require_dtype(X_var, "X_var"); 
        must_greater_equal(Y.ndim(), "Y.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", Y, "Y");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim()");
    }
    
    protected final void check_batchNorm_deltaX_v2(Tensor deltaY, Tensor X, Tensor X_mean, Tensor X_var) {
        require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
        require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", X, "X");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim()");
        equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
    }
    
    protected final void check_batchNorm_gradients_v1(Tensor deltaY, Tensor Y, Tensor X_var, Tensor A, Tensor B) {
        require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
        require_dtype(X_var, "X_var"); require_dtype(A, "A"); require_dtype(B, "B");
        must_greater_equal(Y.ndim(), "Y.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", Y, "Y");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim");
        equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
        equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
        equals(X_var.length, "X_var.length", A.length, "A.length");
        equals(X_var.length, "X_var.length", B.length, "B.length");
    }
    
    protected final void check_batchNorm_gradients_v2(Tensor deltaY, Tensor X, Tensor X_mean, Tensor X_var, Tensor A) {
        require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
        require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var"); require_dtype(A, "A");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", X, "X");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim");
        equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
        equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        equals(X_mean.length, "X_mean.length", A.length, "A.length");
    }
    
    protected final void check_batchNorm_gradients_v2(Tensor deltaY, Tensor X, Tensor X_mean, Tensor X_var, Tensor A, Tensor B) {
        require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
        require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var"); require_dtype(A, "A");
        must_greater_equal(X.ndim(), "X.ndim", 2);
        must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
        equals_valueStructure(deltaY, "deltaY", X, "X");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim");
        equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim");
        equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
        equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim()");
        equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        equals(X_mean.length, "X_mean.length", A.length, "A.length");
        equals(X_mean.length, "X_mean.length", B.length, "B.length");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: affine">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine(boolean inplace, Tensor X, Tensor A, Tensor B) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);
        Tensor deltaA = this.empty(A.dim).c();
        Syncer sc = core.affine2D_deltaA_v1(deltaA.address,
                deltaY.address,
                Y.address, 
                A.address, B.address, A.lengthv,//A.lengthv = row_lengthv
                deltaY.lengthv, deltaY.lastDim());//width = deltaY.lastDim
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);         
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result
                deltaY.address, Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    public Tensor affine_deltaA_v2(Tensor deltaY, Tensor X, int row_length) { return field_mul(deltaY, X, row_length); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_deltaAB_v2(Tensor deltaY, Tensor X, int row_length) {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        int width = X.lastDim(), height = row_length / width;
        int[] dimAB = (height > 1 ? new int[]{ height, width } : new int[]{ width });
        Tensor deltaA = this.empty(dimAB);
        Tensor deltaB = this.empty(dimAB);
        
        Syncer sc = core.affine2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address, 
                X.address,//deltaA.lengthv = row_lengthv
                deltaA.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_leakyRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_relu(boolean inplace, Tensor X, Tensor A, Tensor B) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_relu2D(Y.address,
                X.address, 
                A.address, B.address,
                A.lengthv, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_leakyRelu(boolean inplace, Tensor X, Tensor A, Tensor B, float k) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_leakyRelu2D(Y.address,
                X.address, 
                A.address, B.address,
                A.lengthv, k,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_leakyRelu_deltaX_v1(boolean inplace, 
            Tensor deltaY, float k, Tensor Y, Tensor A)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_fuction_deltaX_v1(deltaY, Y, A);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_leakyRelu2D_deltaX_v1(deltaX.address,
                deltaY.address, k,
                Y.address, 
                A.address, A.lengthv, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    public Tensor affine_relu_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X, Tensor A, Tensor B) {
        return affine_leakyRelu_deltaX_v2(inplace, deltaY, 0.0f, X, A, B);//V2: holdX(), X is not changed
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_leakyRelu_deltaX_v2(boolean inplace, 
            Tensor deltaY, float k, Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_function_deltaX_v2(deltaY, X, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_leakyRelu2D_deltaX_v2(deltaX.address, 
                deltaY.address, k,
                X.address,
                A.address, B.address, A.lengthv,
                deltaY.lengthv(), deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_leakyRelu_deltaAB_v1(Tensor deltaY, float k,
            Tensor Y, Tensor A, Tensor B)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_leakyRelu2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address, k, 
                Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    public Tensor[] affine_relu_deltaAB_v2(Tensor deltaY, Tensor X, Tensor A, Tensor B) {
        return affine_leakyRelu_deltaAB_v2(deltaY, 0.0f, X, A, B);//V2: holdX(), X is not changed
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_leakyRelu_deltaAB_v2(Tensor deltaY, float k,
            Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_funcion_deltaAB_v2(deltaY, X, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_leakyRelu2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address, k, 
                X.address, 
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_elu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_elu(boolean inplace, 
            Tensor X, Tensor A, Tensor B, float alpha, float k)
    {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_elu2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv, 
                alpha, k,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_elu_deltaX_v1(boolean inplace, 
            Tensor deltaY, float alpha, float k, Tensor Y, Tensor A)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_fuction_deltaX_v1(deltaY, Y, A);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_elu2D_deltaX_v1(deltaX.address,
                deltaY.address, alpha, k,
                Y.address, 
                A.address, A.lengthv, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_elu_deltaX_v2(boolean inplace, 
            Tensor deltaY, float alpha, float k, Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_function_deltaX_v2(deltaY, X, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_elu2D_deltaX_v2(deltaX.address, 
                deltaY.address, alpha, k,
                X.address,
                A.address, B.address, A.lengthv,
                deltaY.lengthv(), deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_elu_deltaAB_v1(Tensor deltaY, float alpha, float k,
            Tensor Y, Tensor A, Tensor B)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_elu2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address, alpha, k, 
                Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_elu_deltaAB_v2(Tensor deltaY, float alpha, float k,
            Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_funcion_deltaAB_v2(deltaY, X, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_elu2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address, alpha, k, 
                X.address, 
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_softplus(boolean inplace, Tensor X, Tensor A, Tensor B) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_softplus2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_softplus_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y, Tensor A)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_fuction_deltaX_v1(deltaY, Y, A);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_softplus2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                A.address, A.lengthv, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_softplus_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_function_deltaX_v2(deltaY, X, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_softplus2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address,
                A.address, B.address, A.lengthv,
                deltaY.lengthv(), deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_softplus_deltaAB_v1(Tensor deltaY,
            Tensor Y, Tensor A, Tensor B)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_softplus2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address,
                Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_softplus_deltaAB_v2(Tensor deltaY,
            Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_funcion_deltaAB_v2(deltaY, X, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_softplus2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address,
                X.address, 
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_gelu(boolean inplace, Tensor X, Tensor A, Tensor B) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_gelu2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_gelu_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_function_deltaX_v2(deltaY, X, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_gelu2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address,
                A.address, B.address, A.lengthv,
                deltaY.lengthv(), deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_gelu_deltaAB_v2(Tensor deltaY,
            Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_funcion_deltaAB_v2(deltaY, X, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_gelu2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address,
                X.address, 
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_sigmoid(boolean inplace, Tensor X, Tensor A, Tensor B) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_sigmoid2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_sigmoid_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y, Tensor A)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_fuction_deltaX_v1(deltaY, Y, A);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_sigmoid2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                A.address, A.lengthv, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_sigmoid_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_function_deltaX_v2(deltaY, X, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_sigmoid2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address,
                A.address, B.address, A.lengthv,
                deltaY.lengthv(), deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_sigmoid_deltaAB_v1(Tensor deltaY,
            Tensor Y, Tensor A, Tensor B)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_sigmoid2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address,
                Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_sigmoid_deltaAB_v2(Tensor deltaY,
            Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_funcion_deltaAB_v2(deltaY, X, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_sigmoid2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address,
                X.address, 
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: affine_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_tanh(boolean inplace, Tensor X, Tensor A, Tensor B) {
        if(check) check_affine(X, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine_tanh2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_tanh_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y, Tensor A)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_fuction_deltaX_v1(deltaY, Y, A);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_tanh2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                A.address, A.lengthv, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_tanh_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_function_deltaX_v2(deltaY, X, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.affine_tanh2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address,
                A.address, B.address, A.lengthv,
                deltaY.lengthv(), deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_tanh_deltaAB_v1(Tensor deltaY,
            Tensor Y, Tensor A, Tensor B)//V1: holdY(), Y is not changed
    {
        if(check) check_affine_deltaAB_v1(deltaY, Y, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_tanh2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address,
                Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_tanh_deltaAB_v2(Tensor deltaY,
            Tensor X, Tensor A, Tensor B)//V2: holdX(), X is not changed
    {
        if(check) check_affine_funcion_deltaAB_v2(deltaY, X, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine_tanh2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address,
                X.address, 
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(X, "X"); require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sqBatchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_sqmean.address, eps, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_sqmean, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dtype(X, "X"); require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim");
            equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
            equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sqBatchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_sqmean.address, eps,
                A.address, B.address, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaT"); require_dtype(Y, "Y");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean"); 
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v1(deltaX.address, 
                deltaY.address, 
                Y.address, 
                X_mean.address, X_sqmean.address, eps,
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address, 
                X_mean.address, X_sqmean.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A, Tensor B)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v1(deltaX.address,
                deltaY.address, 
                Y.address,
                X_mean.address, X_sqmean.address, eps, 
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address,
                X_mean.address, X_sqmean.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaA, deltaB }">
    public Tensor sqBatchNorm_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaA_v1(deltaY, Y, A, B);
    }
    public Tensor[] sqBatchNorm_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaAB_v1(deltaY, Y, A, B);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaA_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dimA = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor deltaA = this.empty(dimA).c();
        
        Syncer sc = core.sqBatchNorm2D_deltaA_v2(deltaA.address, 
                deltaY.address, 
                X.address,
                X_mean.address, X_sqmean.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sqBatchNorm_deltaAB_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dim = (height > 1 ? new int[]{ height, width }: new int[]{ width });
        Tensor deltaA = this.empty(dim);
        Tensor deltaB = this.empty(dim);
        
        Syncer sc = core.sqBatchNorm2D_deltaAB_v2(deltaA.c().address, deltaB.c().address,
                deltaY.address,
                X.address, 
                X_mean.address, X_sqmean.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sqBatchNorm_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A, Tensor B)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.sqBatchNorm2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address, 
                Y.address,
                X_mean.address, X_sqmean.address, eps, 
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sqBatchNorm_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            require_dtype(A, "A");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.sqBatchNorm2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address, 
                X.address, 
                X_mean.address, X_sqmean.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps,
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v1(deltaY, Y, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address, 
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v1(deltaX.address,
                deltaY.address, 
                Y.address,
                X_var.address, eps, 
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address,
                X_mean.address, X_var.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaA, deltaB }">
    public Tensor batchNorm_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaA_v1(deltaY, Y, A, B);
    }
    public Tensor[] batchNorm_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaAB_v1(deltaY, Y, A, B);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaA_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dimA = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor deltaA = this.empty(dimA).c();
        
        Syncer sc = core.batchNorm2D_deltaA_v2(deltaA.address, 
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_deltaAB_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_var, "X_var");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dim = (height > 1?  new int[]{ height, width }: new int[]{ width });
        Tensor deltaA = this.empty(dim);
        Tensor deltaB = this.empty(dim);
        
        Syncer sc = core.batchNorm2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address,
                X.address, 
                X_mean.address, X_var.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address, 
                Y.address,
                X_var.address, eps, 
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_leakyRelu (relu)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation (relu)">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_relu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_relu2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_relu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_relu2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward-propagation (leakyRelu)">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_leakyRelu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, float k)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_leakyRelu2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, k,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_leakyRelu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B, float k)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_leakyRelu2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, k,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_leakyRelu_deltaX_v1(boolean inplace,
            Tensor deltaY, float k, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v1(deltaY, Y, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_leakyRelu2D_deltaX_v1(deltaX.address, 
                deltaY.address, k,
                Y.address,
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    public Tensor batchNorm_relu_deltaX_v2(boolean inplace,//V2: holdX(), X is not changed
            Tensor deltaY, Tensor X, Tensor X_mean, Tensor X_var, float eps) {
        return batchNorm_leakyRelu_deltaX_v2(inplace, deltaY, 0.0f, X, X_mean, X_var, eps);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_leakyRelu_deltaX_v2(boolean inplace,
            Tensor deltaY, float k, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_leakyRelu2D_deltaX_v2(deltaX.address,
                deltaY.address, k, 
                X.address,
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_leakyRelu_gradients_v1(boolean inplace, 
            Tensor deltaY, float k, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if (check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_leakyRelu2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address, k, 
                Y.address, 
                X_var.address, eps,
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    public Tensor[] batchNorm_relu_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B) {
        return batchNorm_leakyRelu_gradients_v2(inplace, 
                deltaY, 0.0f, X, X_mean, X_var, eps, A, B);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_leakyRelu_gradients_v2(boolean inplace, 
            Tensor deltaY, float k, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_leakyRelu2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address, k, 
                X.address,
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_elu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_elu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            float alpha, float k)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_elu2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, alpha, k,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_elu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B,
            float alpha, float k)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_elu2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, alpha, k,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_elu_deltaX_v1(boolean inplace,
            Tensor deltaY, float alpha, float k, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v1(deltaY, Y, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_elu2D_deltaX_v1(deltaX.address, 
                deltaY.address, alpha, k,
                Y.address,
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_elu_deltaX_v2(boolean inplace,
            Tensor deltaY, float alpha, float k, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_elu2D_deltaX_v2(deltaX.address,
                deltaY.address, alpha, k, 
                X.address,
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_elu_gradients_v1(boolean inplace, 
            Tensor deltaY, float alpha, float k, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_elu2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address, alpha, k, 
                Y.address, 
                X_var.address, eps,
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_elu_gradients_v2(boolean inplace, 
            Tensor deltaY, float alpha, float k, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_elu2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address, alpha, k, 
                X.address,
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_softplus">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_softplus(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_softplus2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_softplus(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_softplus2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_softplus_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v1(deltaY, Y, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_softplus2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address,
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_softplus_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_softplus2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_softplus_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_softplus2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                Y.address, 
                X_var.address, eps,
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_softplus_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_softplus2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_gelu">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_gelu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_gelu2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_gelu(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_gelu2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_gelu_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_gelu2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_gelu_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_gelu2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_sigmoid">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_sigmoid(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_sigmoid2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_sigmoid(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_sigmoid2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_sigmoid_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v1(deltaY, Y, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_sigmoid2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address,
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_sigmoid_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_sigmoid2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_sigmoid_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_sigmoid2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                Y.address, 
                X_var.address, eps,
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_sigmoid_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_sigmoid2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm_tanh">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_tanh(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) this.check_batchNorm(X, X_mean, X_var);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_tanh2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_tanh(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm(X, X_mean, X_var, A, B);
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm_tanh2D(Y.address, 
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_tanh_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v1(deltaY, Y, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_tanh2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address,
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_tanh_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) check_batchNorm_deltaX_v2(deltaY, X, X_mean, X_var);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm_tanh2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation (affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_tanh_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v1(deltaY, Y, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_tanh2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                Y.address, 
                X_var.address, eps,
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_tanh_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) check_batchNorm_gradients_v2(deltaY, X, X_mean, X_var, A, B);
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm_tanh2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2 
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(X, "X"); require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmeann");     
            must_greater_equal(X.ndim(), "X.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean.ndim", 1);
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.layerNorm2D(Y.address,
                X.address, 
                X_mean.address, 
                X_sqmean.address, eps,
                X_mean.length,//X_row_mean.length = field_length
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_sqmean, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dtype(X); require_dtype(X_mean); require_dtype(X_sqmean);
            must_greater_equal(X.ndim(), "X.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean.ndim", 1);
            equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
            equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.layerNorm2D(Y.address,
                X.address, 
                X_mean.address, 
                X_sqmean.address, eps,
                A.address, B.address, X_mean.length,//X_row_mean.length = field_length
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean,ndim", 1);
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v1(deltaX.address, 
                deltaY.address, Y.address, 
                X_mean.address,
                X_sqmean.address, eps, X_mean.length,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "Y");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean.ndim", 1);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, X.address,
                X_mean.address, 
                X_sqmean.address, eps, X_mean.length,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(Y, "Y");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            require_dtype(A, "A"); require_dtype(B, "B");
            must_greater_equal(Y.ndim(), "Y.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean,ndim", 1);
            equals(Y.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
            equals(Y.lastDim(), "Y.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v1(deltaX.address,
                deltaY.address, Y.address, 
                X_mean.address,
                X_sqmean.address, eps,
                A.address, B.address, X_mean.length, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X);
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            require_dtype(A, "A");
            must_greater_equal(X.ndim(), "X.ndim", 2);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean,ndim", 1);
            equals(X.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, X.address,
                X_mean.address,
                X_sqmean.address, eps,
                A.address, X_mean.length,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public Tensor layerNorm_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) { 
        return affine_deltaA_v1(deltaY, Y, A, B);
    }
    public Tensor[] layerNorm_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaAB_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaA_v2(Tensor deltaY,
            Tensor X, Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY, "deltaY"); require_dtype(X, "X");
            require_dtype(X_mean, "X_mean"); require_dtype(X_sqmean, "X_sqmean");
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            must_greater_equal(X.ndim(), "X.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean,ndim", 1);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        int field_length = X_mean.length;
        int row_length = X.length / field_length;
        int width = deltaY.lastDim(), height = row_length / width;//X_mean.memstruc = A.mem_struc
        int[] dimA = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor deltaA = this.empty(dimA).c();
        
        Syncer sc = core.layerNorm2D_deltaA_v2(deltaA.address, 
                deltaY.address, X.address,
                X_mean.address, 
                X_sqmean.address, eps, field_length,//field_length = X_row_mean.length;
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] layerNorm_deltaAB_v2(Tensor deltaY,
            Tensor X, Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dtype(deltaY); require_dtype(X);
            require_dtype(X_mean); require_dtype(X_sqmean);
            must_greater_equal(deltaY.ndim(), "deltaY.ndim", 2);
            must_greater_equal(X.ndim(), "X.ndim", 2);
            equals(X_mean.ndim(), "X_mean.ndim", 1);
            equals(X_sqmean.ndim(), "X_sqmean,ndim", 1);
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        int field_length = X_mean.length;
        int row_length = X.length / field_length;
        int width = deltaY.lastDim(), height = row_length / width;//X_mean.memstruc = A.mem_struc
        int[] dim = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor deltaA = this.empty(dim);
        Tensor deltaB = this.empty(dim);
        
        Syncer sc = core.layerNorm2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address, X.address, 
                X_mean.address,
                X_sqmean.address, eps, field_length, //field_length = X_row_mean.length;
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[]{ deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot">
    public Tensor onehot(Tensor X, int num_class) { return onehot(X, 1.0f, 0.0f, num_class); }
    public Tensor onehot(Tensor X, float alpha, int num_class) {
        float beta = (1.0f - alpha) / (num_class - 1.0f);
        return onehot(X, alpha, beta, num_class);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor onehot(Tensor X, float alpha, float beta, int num_class) {
        if(check) { must_greater_equal(num_class, "num_class", 2); }
        
        int dimX[] = X.dim, ndimX = dimX.length;//[dimX, num_class]
        int[] dimY = new int[ndimX + 1]; dimY[ndimX] = num_class;
        for(int i=0; i<ndimX; i++) dimY[i] = dimX[i];
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = null; 
        if(X.dataType.equals(core.dataType_int32())) {
            sc = core.onehot2D_row_int32(Y.address, X.address,
                    alpha, beta, X.length,  //X.length = field_length
                    Y.lengthv, Y.lastDim());
        }
        else if(X.dataType.equals(core.dataType_int8())) {
            sc = core.onehot2D_row_int8(Y.address, X.address,
                    alpha, beta, X.length,
                    Y.lengthv, Y.lastDim());
        }
        else throw new RuntimeException(String.format(
                "X.dataType { got %s } != %s or %s", X.dataType, 
                core.dataType_int32(), 
                core.dataType_int8()));
        
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pixel_to_tensor">
    @Passed("CudaFloat32EngieBase")
    public Tensor pixel_to_tensor(boolean inplace, Tensor X) {
        if(check) { require_int8(X, "X"); }
        Tensor Y = this.empty(X.dim).c();
        Syncer sc1 = core.pix2tensor2D(Y.address, X.address, Y.lengthv, Y.lastDim());
        
        //======[inplace = false, return the new Tensor Y]======================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //======[inplace = true, return the old Tensor X]=======================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<dtype> -> X<uint8>
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor_to_pixel(boolean inplace, Tensor X) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty_int8(X.dim).c();
        Syncer sc1 = core.tensor2pix2D(Y.address, X.address, Y.lengthv, Y.lastDim());
        
        //======[inplace = false, return the new Tensor Y]======================
        if(!inplace) { if(sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //======[inplace = true, return the old Tensor X]=======================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int8> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free(old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="SGD">
    public Tensor sgd(Tensor W, Tensor deltaW, float lr) { return add(true, 1.0f, W, -lr, deltaW); }
    public Tensor sgd(Tensor W, Collection<Tensor> grads, float lr) {
        if(check) { 
            require_dtype(W, "W"); require_dtype(grads, "grads");
            equals_valueStructure(W, "W", grads, "grads"); 
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.sgd(W.address, gradsAddr, lr, W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nesterov, 
            Tensor deltaW, float lr)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(deltaW, "deltaW");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.sgdmn(W.address, 
                V.address, momentum, dampen, nesterov, 
                deltaW.address, lr,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nestrov, 
            Collection<Tensor> grads, float lr)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(grads, "grads");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.sgdmn(W.address,
                V.address, momentum, dampen, nestrov,
                gradsAddr, lr, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SGDMN(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nesterov, 
            Tensor deltaW, float lr, 
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(deltaW, "deltaW");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.sgdmn_decay(W.address, 
                V.address, momentum, dampen, nesterov, 
                deltaW.address, lr,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nesterov, 
            Collection<Tensor> grads, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(grads, "grads");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.sgdmn_decay(W.address,
                V.address, momentum, dampen, nesterov,
                gradsAddr, lr, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Momentum">
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dtype(W, "W");  require_dtype(V, "V"); require_dtype(deltaW, "deltaW");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.momentum2D(W.address, 
                V.address, a1, a2,
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Collection<Tensor> grads, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(grads, "grads");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.momentum2D(W.address, 
                V.address, a1, a2,
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Momentum(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(deltaW, "deltaW");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.momentum2D_decay(W.address, 
                V.address, a1, a2,
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Collection<Tensor> grads, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(V, "V"); require_dtype(grads, "grads");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.momentum2D_decay(W.address, 
                V.address, a1, a2,
                gradsAddr, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW"); require_dtype(S, "S"); 
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.rmsprop2D(W.address, 
                S.address, a1, a2, eps_t, 
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Collection<Tensor> grads, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads"); require_dtype(S, "S"); 
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.rmsprop2D(W.address, 
                S.address, a1, a2, eps_t, 
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW"); require_dtype(S, "S");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.rmsprop2D_decay(W.address,
                S.address, a1, a2, eps_t,
                deltaW.address, lr_t, L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Collection<Tensor> grads, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads"); require_dtype(S, "S");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc  = core.rmsprop2D_decay(W.address, 
                S.address, a1, a2, eps_t,
                gradsAddr, lr_t, 
                L1coef, L2coef, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adam">
    //<editor-fold defaultstate="collapsed" desc="adam_type2">
    public Tensor adam_type2(Tensor W,
            Tensor V, float a1, float a2, float Uv,
            Tensor S, float b1, float b2, float eps, float Us,
            Tensor deltaW, float lr)
    {
         if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adam2D_type2(W.address, 
                V.address, a1, a2, Uv,
                S.address, b1, b2, eps, Us,
                deltaW.address, lr,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adam2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                deltaW.address, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> grads, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adam2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                gradsAddr, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adam2D_decay(W.address,
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> grads, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        Syncer sc = core.adam2D_decay(W.address,
                V.address, a1, a2,
                S.address, b1, b2, eps_t,
                gradsAddr, lr_t, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam(amsgrad)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adam_amsgrad(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor Smax,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(S, "Smax");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", Smax, "Smax");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adam_amsgrad2D(W.address,
                V.address, a1, a2,
                S.address, b1, b2, eps_t,
                Smax.address,
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam_amsgrad(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor Smax,
            Collection<Tensor> grads, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(S, "Smax");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", Smax, "Smax");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adam_amsgrad2D(W.address, 
                V.address, a1, a2,
                S.address, b1, b2, eps_t,
                Smax.address, 
                gradsAddr, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam(amsgrad, L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adam_amsgrad(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor Smax,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(S, "Smax");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", Smax, "Smax");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adam_amsgrad2D_decay(W.address,
                V.address, a1, a2,
                S.address, b1, b2, eps_t,
                Smax.address, 
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam_amsgrad(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor Smax,
            Collection<Tensor> grads, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(S, "Smax");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", Smax, "Smax");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adam_amsgrad2D_decay(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                Smax.address, 
                gradsAddr, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adamax2D(W.address,
                V.address, a1, a2,
                S.address, b1, eps, 
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Collection<Tensor> grads, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adamax2D(W.address, 
                V.address, a1, a2,
                S.address, b1, eps,
                gradsAddr, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
       
        Syncer sc = core.adamax2D_decay(W.address,
                V.address, a1, a2, 
                S.address, b1, eps,
                deltaW.address, lr_t, 
                L1coef, L2coef, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Collection<Tensor> grads, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads);
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adamax2D_decay(W.address,
                V.address, a1, a2,
                S.address, b1, eps,
                gradsAddr, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamW(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float lr_t, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adamW2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                deltaW.address, lr_t, lr,
                L1coef, L2coef, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamW(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> grads, float lr_t, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adamW2D(W.address,
                V.address, a1, a2, 
                S.address, b1, b2, eps_t,
                gradsAddr, lr_t, lr,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamW_amsgrad(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor Smax,
            Tensor deltaW, float lr_t, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(Smax, "Smax");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", Smax, "Smax");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adamW_amsgrad2D(W.address, 
                V.address, a1, a2,
                S.address, b1, b2, eps_t, 
                Smax.address, 
                deltaW.address, lr_t, lr, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamW_amsgrad(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor Smax,
            Collection<Tensor> grads, float lr_t, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(Smax, "Smax");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", Smax, "Smaax");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.adamW_amsgrad2D(W.address,
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                Smax.address,
                gradsAddr, lr_t, lr, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RAdam">
    @Passed("CudaFloat32EngieBase")
    public Tensor radam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float pcur, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        //======[lr * r * correct_beta2/correct_beta1 * V/sqrt(S + eps)]========
        if(pcur > 5.0f) {
            Syncer sc = core.adam2D(W.address,
                    V.address, a1, a2, 
                    S.address, b1, b2, eps_t, 
                    deltaW.address, lr_t, 
                    W.lengthv, W.lastDim());
            if(sync) sc.sync(); else W.setSyncer(sc);
            return W;
        }
        
        //======[lr / correct_beta1 * V]========================================
        Syncer sc1 = core.momentum2D(W.address,
                V.address, a1, a2, 
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        
        Syncer sc2 = core.quadratic2_2D(S.address,//S = b1*S + dW^2
                S.address, deltaW.address, 
                0, 0, b2, b1, 0, 0,
                S.lengthv, S.lastDim());
        
        Syncer sc = Syncer.dual(sc1, sc2);
        if(sync) sc.sync(); else { W.setSyncer(sc); S.setSyncer(sc); }
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor radam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> grads, float pcur, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        //======[lr * r * correct_beta2/correct_beta1 * V/sqrt(S + eps)]========
        if(pcur > 5.0f) {
            Syncer sc = core.adam2D(W.address,
                    V.address, a1, a2, 
                    S.address, b1, b2, eps_t, 
                    gradsAddr, lr_t, 
                    W.lengthv, W.lastDim());
            if(sync) sc.sync(); else W.setSyncer(sc);
            return W;
        }
        
        //======[lr / correct_beta1 * V]========================================
        Syncer sc1 = core.momentum2D(W.address,
                V.address, a1, a2, 
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        
        Syncer sc2 = core.quadratic2_iteration2D(S.address,//S = b1*S + dW^2
                S.address, gradsAddr, 
                0, 0, b2, b1, 0, 0,
                S.lengthv, S.lastDim());
        
        Syncer sc = Syncer.dual(sc1, sc2);
        if(sync) sc.sync(); else { W.setSyncer(sc); S.setSyncer(sc); }
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RAdam(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor radam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float pcur, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        //======[lr * r * correct_beta2/correct_beta1 * V/sqrt(S + eps)]========
        if(pcur > 5.0f) {
            Syncer sc = core.adam2D_decay(W.address,
                    V.address, a1, a2, 
                    S.address, b1, b2, eps_t, 
                    deltaW.address, lr_t,
                    L1coef, L2coef,
                    W.lengthv, W.lastDim());
            if(sync) sc.sync(); else W.setSyncer(sc);
            return W;
        }
        
        //======[lr / correct_beta1 * V]========================================
        Syncer sc1 = core.momentum2D_decay(W.address,
                V.address, a1, a2, 
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        
        Syncer sc2 = core.quadratic2_2D(S.address,//S = b1*S + dW^2
                S.address, deltaW.address, 
                0, 0, b2, b1, 0, 0,
                S.lengthv, S.lastDim());
        
        Syncer sc = Syncer.dual(sc1, sc2);
        if(sync) sc.sync(); else { W.setSyncer(sc); S.setSyncer(sc); }
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor radam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> grads, float pcur, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;
        
        //======[lr * r * correct_beta2/correct_beta1 * V/sqrt(S + eps)]========
        if(pcur > 5.0f) {
            Syncer sc = core.adam2D_decay(W.address,
                    V.address, a1, a2, 
                    S.address, b1, b2, eps_t, 
                    gradsAddr, lr_t, 
                    L1coef, L2coef,
                    W.lengthv, W.lastDim());
            if(sync) sc.sync(); else W.setSyncer(sc);
            return W;
        }
        
        //======[lr / correct_beta1 * V]========================================
        Syncer sc1 = core.momentum2D_decay(W.address,
                V.address, a1, a2, 
                gradsAddr, lr_t, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        
        Syncer sc2 = core.quadratic2_iteration2D(S.address,//S = b1*S + dW^2
                S.address, gradsAddr, 
                0, 0, b2, b1, 0, 0,
                S.lengthv, S.lastDim());
        
        Syncer sc = Syncer.dual(sc1, sc2);
        if(sync) sc.sync(); else { W.setSyncer(sc); S.setSyncer(sc); }
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(G, "G");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adamod2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t,
                G.address, c1, c2, 
                deltaW.address, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
     @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Collection<Tensor> grads, float lr_t)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(G, "G");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;

        Syncer sc = core.adamod2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                G.address, c1, c2,
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(deltaW, "deltaW");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(G, "G");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adamod2D_decay(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t,
                G.address, c1, c2, 
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Collection<Tensor> grads, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dtype(W, "W"); require_dtype(grads, "grads");
            require_dtype(V, "V"); require_dtype(S, "S"); require_dtype(G, "G");
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", grads, "grads");
        }
        
        long[] gradsAddr = new long[grads.size()]; int index = 0;
        for(Tensor grad : grads) gradsAddr[index++] = grad.address;

        Syncer sc = core.adamod2D_decay(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                G.address, c1, c2,
                gradsAddr, lr_t, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    public Engine set_seed(long seed) { core.set_seed(seed); return this; }
    
    //<editor-fold defaultstate="collapsed" desc="uniform">
    public Tensor Uniform(int... dim) { return uniform(empty(dim).c(), 0.0f, 1.0f); }
    public Tensor Uniform(Tensor X) { return uniform(X, 0.0f, 1.0f); }
     
    public Tensor uniform(float vmin, float vmax, int... dim) { return uniform(empty(dim).c(), vmin, vmax); }
    @Passed("CudaFloat32EngieBase")
    public Tensor uniform(Tensor X, float vmin, float vmax) {
        if(check) { require_dtype(X, "X"); }
        Syncer sc = core.uniform2D(X.address, vmin, vmax, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sparse uniform">
    public Tensor Sparse_Uniform(Tensor X, float p) { return sparse_uniform(X, p, 0.0f, 1.0f); }
    public Tensor Sparse_Uniform(float p, int... dim) {
        return sparse_uniform(empty(dim).c(), p, 0.0f, 1.0f);
    }
    
    public Tensor sparse_uniform(float p, float vmin, float vmax, int... dim)  {
        return sparse_uniform(empty(dim).c(), p, vmin, vmax);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sparse_uniform(Tensor X, float p, float vmin, float vmax)  {
        if(check) { require_dtype(X, "X"); }
        Syncer sc = core.sparse_uniform2D(X.address, p, vmin, vmax, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="bernouli">
    public Tensor Bernouli(float p, int... dim) { return bernouli(empty(dim).c(), p, 1.0f , 0.0f); }
    public Tensor Bernouli(float p, float v1, float v2, int... dim)  {
        return bernouli(empty(dim).c(), p, v1, v2);
    }
    
    public Tensor bernouli(Tensor X, float p) { return bernouli(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor bernouli(Tensor X, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Syncer sc = core.bernouli2D(X.address, p, v1, v2, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="bernouli_mul">
    public Tensor[] dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return bernouli_mul(X, p, pr, 0);//X * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] bernouli_mul(Tensor X, float p) { return bernouli_mul(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] bernouli_mul(Tensor X, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                p, v1, v2, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="leakyRelu_bernouli_mul">
    public Tensor[] relu_dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return leakyRelu_bernouli_mul(X, 0.0f, p, pr, 0);//leakyRelu(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] relu_bernouli_mul(Tensor X, float p) { return leakyRelu_bernouli_mul(X, 0.0f, p, 1.0f , 0.0f); }
    public Tensor[] relu_bernouli_mul(Tensor X, float p, float v1, float v2) {
        return leakyRelu_bernouli_mul(X, 0.0f, p, v1, v2);
    }
    
    public Tensor[] leakyRelu_dropout(Tensor X, float k, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return leakyRelu_bernouli_mul(X, k, p, pr, 0);//leakyRelu(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] leakyRelu_bernouli_mul(Tensor X, float k, float p) { return leakyRelu_bernouli_mul(X, k, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] leakyRelu_bernouli_mul(Tensor X, float k, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.leakyRelu_bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                k, p, v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="elu_bernouli_mul">
    public Tensor[] elu_dropout(Tensor X, float alpha, float k, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return elu_bernouli_mul(X, alpha, k, p, pr, 0);//elu(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] elu_bernouli_mul(Tensor X, float alpha, float k, float p) { return elu_bernouli_mul(X, alpha, k, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] elu_bernouli_mul(Tensor X, float alpha, float k, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.elu_bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                alpha, k, p, v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="softplus_bernouli_mul">
    public Tensor[] softplus_dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return softplus_bernouli_mul(X, p, pr, 0);//softplus(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] softplus_bernouli_mul(Tensor X, float p) { return softplus_bernouli_mul(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] softplus_bernouli_mul(Tensor X, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.softplus_bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                p, v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="gelu_bernouli_mul">
    public Tensor[] gelu_dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return gelu_bernouli_mul(X, p, pr, 0);//gelu(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] gelu_bernouli_mul(Tensor X, float p) { return gelu_bernouli_mul(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] gelu_bernouli_mul(Tensor X, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.gelu_bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                p, v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sigmoid_bernouli_mul">
    public Tensor[] sigmoid_dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return sigmoid_bernouli_mul(X, p, pr, 0);//sigmoid(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] sigmoid_bernouli_mul(Tensor X, float p) { return sigmoid_bernouli_mul(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sigmoid_bernouli_mul(Tensor X, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.sigmoid_bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                p, v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tanh_bernouli_mul">
    public Tensor[] tanh_dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent, pr = 1.0f / p;
        return tanh_bernouli_mul(X, p, pr, 0);//tanh(X) * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] tanh_bernouli_mul(Tensor X, float p) { return tanh_bernouli_mul(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] tanh_bernouli_mul(Tensor X, float p, float v1, float v2) {
        if(check) { require_dtype(X, "X"); }
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.tanh_bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                p, v1, v2,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="gaussian">
    public Tensor Gaussian(int... dim) { return gaussian(empty(dim).c(), 0.0f, 1.0f); }
    public Tensor Gaussian(Tensor X) { return gaussian(X, 0.0f, 1.0f); }
    
    public Tensor gaussian(float mu, float sigma, int... dim) { return gaussian(empty(dim).c(), mu, sigma); } 
    @Passed("CudaFloat32EngieBase")
    public Tensor gaussian(Tensor X, float mu, float sigma) {
        if(check) { require_dtype(X, "X"); }
        Syncer sc = core.gaussian2D(X.address, mu, sigma, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sparse gaussian">
    public Tensor Sparse_Gaussian(float p, int... dim) {  return sparse_gaussian(empty(dim).c(), p, 0.0f, 1.0f); }
    public Tensor Sparse_Gaussian(Tensor X, float p) { return sparse_gaussian(X, p, 0.0f, 1.0f); }
    
    public Tensor sparse_gaussian(float p, float mu, float sigma, int... dim)  {
        return sparse_gaussian(empty(dim).c(), p, mu, sigma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sparse_gaussian(Tensor X, float p, float mu, float sigma) {
        if(check) { require_dtype(X, "X"); }
        Syncer sc = core.sparse_gaussian2D(X.address, p, mu, sigma,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: FanMode">
    public static class FanMode {
        public static final int fan_in = 0;
        public static final int fan_out = 1;
        public static final int fan_in_out = 2;
        public static final int default_value = fan_in;
    }
    
    public int fan_mode(String type) {
        type = type.toLowerCase();
        if("fan_in".equals(type)) return FanMode.fan_in;
        if("fan_out".equals(type)) return FanMode.fan_out;
        if("fan_in_out".equals(type)) return FanMode.fan_in_out;
        return FanMode.default_value;
    }
    
    public float fan(int fan_mode, int[] fans) {
        if(fan_mode == FanMode.fan_in) return fans[0];
        if(fan_mode == FanMode.fan_out) return fans[1];
        return (fans[0] + fans[1]) * 0.5f; 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: Nonlinearity">
    public static class Nonlinearity  {
        public static final int sigmoid = 0;
        public static final int tanh = 1;
        public static final int relu = 2;
        public static final int leaky_relu = 3;
        public static final int elu = 4;
        public static final int default_value = leaky_relu;
    }
    
    public int nonlinearity(String type) {
        type = type.toLowerCase();
        if("sigmoid".equals(type)) return Nonlinearity.sigmoid;
        if("tanh".equals(type)) return Nonlinearity.tanh;
        if("relu".equals(type)) return Nonlinearity.relu;
        if("leaky_relu".equals(type)) return Nonlinearity.leaky_relu;
        if("elu".equals(type)) return Nonlinearity.elu;
        return Nonlinearity.default_value;
    }
    
    protected float gain(int nonlinearity, float... params) {
        if(nonlinearity == Nonlinearity.sigmoid) return 1.0f;
        if(nonlinearity == Nonlinearity.tanh) return 1.666667f;// 5/3
        if(nonlinearity == Nonlinearity.relu) return 1.414214f;//sqrt(2.0)
        if(nonlinearity == Nonlinearity.leaky_relu) {
            float k = 0.01f;
            if(params != null && params.length != 0) k = params[0];
            return (float) Math.sqrt(2.0 / (1.0 + k*k));
        }
        if(nonlinearity == Nonlinearity.elu) return 0.75f;//3.0 / 4
        return 1.0f;
    }
    //</editor-fold>
    
    public int xavier_fan_mode = FanMode.default_value;
    //<editor-fold defaultstate="collapsed" desc="xavier_uniform">
    public Tensor xavier_uniform(Tensor X, int[] fans) { return xavier_uniform(X, 1.0f, xavier_fan_mode, fans); }
    public Tensor xavier_uniform(Tensor X, int fan_mode, int[] fans) { return xavier_uniform(X, 1.0f, fan_mode, fans); }
    public Tensor xavier_uniform(Tensor X, float alpha, int fan_mode, int[] fans) {
        float fan = fan(fan_mode, fans);
        float std = (float) (1.0 / Math.sqrt(fan));
        float bound = alpha * 1.732051f * std;//sqrt(3) = 1.732051 
        return this.uniform(X, -bound, bound);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="xavier_gaussian">
    public Tensor xavier_gaussian(Tensor X, int[] fans) { return xavier_gaussian(X, 1.0f, xavier_fan_mode, fans); }
    public Tensor xavier_gaussian(Tensor X, int fan_mode, int[] fans) { return xavier_gaussian(X, 1.0f, fan_mode, fans); }
    public Tensor xavier_gaussian(Tensor X, float alpha, int fan_mode, int[] fans) {
        float fan = fan(fan_mode, fans);
        float std = (float) (1.0 / Math.sqrt(fan));
        float sigma = alpha * std;
        return this.gaussian(X, 0, sigma);
    }
    //</editor-fold>
    
    public int kaiming_fan_mode = FanMode.default_value;
    public int kaiming_no_linearity = Nonlinearity.default_value;
    //<editor-fold defaultstate="collapsed" desc="kaiming_uniform">
    public Tensor kaiming_uniform(Tensor X, int[] fans) {
        return kaiming_uniform(X, 1.0f, 
               kaiming_fan_mode, fans,
               kaiming_no_linearity,  null);
    }
    
    public Tensor kaiming_uniform(Tensor X, float alpha, int[] fans) {
        return kaiming_uniform(X, alpha, 
               kaiming_fan_mode, fans,
               kaiming_no_linearity,  null);
    }
    
    public Tensor kaiming_uniform(Tensor X, 
            int fan_mode, int[] fans,
            int nonlinearity, float... params) {
        return kaiming_uniform(X, 1.0f, 
                fan_mode, fans,
                nonlinearity, params);
    }
    
    public Tensor kaiming_uniform(Tensor X, int[] fans, float... params) {
        return kaiming_uniform(X, 1.0f, 
                kaiming_fan_mode, fans,
                kaiming_no_linearity, params);
    }
    
    public Tensor kaiming_uniform(Tensor X, float alpha,
            int fan_mode, int[] fans,
            int nonlinearity, float... params)
    {
        float fan = fan(fan_mode, fans);
        float gain = gain(nonlinearity, params);
        float std = (float) (gain / Math.sqrt(fan));
        float bound = alpha * 1.732051f * std;//sqrt(3) = 1.732051
        return this.uniform(X, -bound, bound);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="kaiming_gaussian">
    public Tensor kaiming_gaussian(Tensor X, int[] fans) {
        return kaiming_gaussian(X, 1.0f, 
                kaiming_fan_mode, fans,
                kaiming_no_linearity, null);
    }
    
    public Tensor kaiming_gaussian(Tensor X, float alpha, int[] fans) {
        return kaiming_gaussian(X, alpha, 
                kaiming_fan_mode, fans,
                kaiming_no_linearity, null);
    }
    
    public Tensor kaiming_gaussian(Tensor X, 
            int fan_mode, int[] fans, 
            int nonlinearity, float... params) {
        return kaiming_gaussian(X, 1.0f, 
                fan_mode, fans,
                nonlinearity, params);
    }
    
    public Tensor kaiming_gaussian(Tensor X, float alpha, 
            int fan_mode, int[] fans,
            int nonlinearity, float... params) 
    {
        float fan = fan(fan_mode, fans);
        float gain = gain(nonlinearity, params);
        float std = alpha * (float) (gain / Math.sqrt(fan));
        float sigma = alpha*std;
        return this.gaussian(X, 0, sigma);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    //<editor-fold defaultstate="collapsed" desc="straight linear">
    public Result<Boolean> hasNan(Tensor X) {
        return new Result<Boolean>() {
            @Override
            protected Boolean waitResult() {
               return Float.isNaN(straight_sum(X).get());
            }
        };
    }
    
    public Result<Float> straight_mean(Tensor X) {
        float alpha = 1.0f / X.length; //(1 / N) * sum(X) = sum(X / N)
        return straight_linear(X, alpha, 0.0f);
    }
    
    public Result<Float> straight_sum(Tensor X) { return straight_linear(X, 1.0f, 0.0f); }
    public Result<Float> straight_sum(Tensor X, float alpha) { return straight_linear(X, alpha, 0.0f);  }
    @Passed("CudaFloat32EngieBase")//sum(alpha*X + beta)
    public Result<Float> straight_linear(Tensor X, float alpha, float beta) {
        if(check) { require_dtype(X, "X"); }
        Result<Float> result = core.straight_linear(X.address, 
                alpha, beta, X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight quadratic">
    public Result<Float> straight_sqmean(Tensor X) {//(1/ N) * sum(X^2) = sum(X^2 / N)
        float alpha = 1.0f / X.length;
        return straight_quadratic(X, alpha, 0.0f, 0.0f);
    }
    
    public Result<Float> straight_sqsum(Tensor X) { return straight_quadratic(X, 1.0f, 0.0f, 0.0f); }
    public Result<Float> straight_sqsum(Tensor X, float alpha) {
        return straight_quadratic(X, alpha, 0.0f, 0.0f); //alpha * sum(X^2)
    }
   
    @Passed("CudaFloat32EngieBase")//sum(alpha*X^2 + beta*X + gamma)
    public Result<Float> straight_quadratic(Tensor X, float alpha, float beta, float gamma)  {
        if(check) { require_dtype(X, "X"); }
        Result<Float> result = core.straight_quadratic(X.address, 
                alpha, beta, gamma, X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight minValue & max">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_max(Tensor X)  {
        if(check) { require_dtype(X, "X"); }
        Result<Float> result = core.straight_max(X.address,
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
      
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_min(Tensor X)  {
        if(check) { require_dtype(X, "X"); }
        Result<Float> result = core.straight_min(X.address,
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight minValue & max indexed">
    @Passed("CudaFloat32EngieBase")
    public IndexedResult<Float> straight_max_indexed(Tensor X)  {
        if(check) { require_dtype(X, "X"); }
        IndexedResult<Float> result = core.straight_max_indexed(X.address,
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
      
   @Passed("CudaFloat32EngieBase")
    public IndexedResult<Float> straight_min_indexed(Tensor X)  {
        if(check) { require_dtype(X, "X"); }
        IndexedResult<Float> result = core.straight_min_indexed(X.address, 
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="non_zero_percent">
    public Result<Float> zero_percent(Tensor X) {
        Result<Float> nz = this.nonzero_percent(X);
        return new Result<Float>() {
            @Override
            protected Float waitResult() {
                return 1.0f - nz.get();
            }
        };
    }
    
    public Result<Float> nonzero_percent(Tensor X) {
        Tensor Y = this.gt(true, square(false, X).c(), 0);//Y^2 > 0
        
        float alpha = 1.0f / Y.length;
        Result<Float> result = core.straight_linear(
                Y.c().address, alpha, 0.0f, 
                Y.lengthv, Y.lastDim());
        
        if(sync) { result.get(); Y.delete(); return result; }
        return Result.dual(result, ()->{ Y.delete(); });
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight equal">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_equal(Tensor X1, Tensor X2) {
        Tensor Y = equal(X1, X2);

        float alpha = 1.0f / Y.length;
        Result<Float> result = core.straight_linear(
                Y.c().address, alpha, 0.0f, 
                Y.lengthv, Y.lastDim());
        
        if(sync) { result.get(); Y.delete(); return result; }
        return Result.dual(result, ()->{ Y.delete(); });
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="straight_var_mean">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_var(Tensor X) {
        if(check) { require_dtype(X, "X"); }
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f,
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f,
                X.lengthv, X.lastDim());
        
        return new Result<Float>() {
            @Override
            protected Float waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float var = Math.max(0.0f, sm - m*m);//E(X^2) - E(X)*E(X)
                return var;
            }
        };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Result<float[]> straight_var_mean(Tensor X) {
        if(check) { require_dtype(X, "X"); }
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f, 
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f,
                X.lengthv, X.lastDim());
        
        return new Result<float[]>() {
            @Override
            protected float[] waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float var = Math.max(0.0f, sm - m*m);//E(X^2) - E(X)*E(X)
                return new float[] { var, m };
            }
        };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight_std_mean">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_std(Tensor X) {
        if(check) { require_dtype(X, "X"); }
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f, 
                X.lengthv, X.lastDim());
        Result<Float> squmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f, 
                X.lengthv, X.lastDim());
        
        Result<Float> result =  new Result<Float>() {
            @Override
            protected Float waitResult() {
                float m = mean.get();
                float sm = squmean.get();
                float std = (float) Math.sqrt(Math.max(0.0f, sm - m*m));//sqrt(E(X^2) - E(X)*E(X))
                return std;
            }
        };
        if(sync) result.get();
        return result;
    }   
    
    @Passed("CudaFloat32EngieBase")
    public Result<float[]> straight_std_mean(Tensor X) {
        if(check) { require_dtype(X, "X"); }
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f, 
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f, 
                X.lengthv, X.lastDim());
        
        Result<float[]> result =  new Result<float[]>() {
            @Override
            protected float[] waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float std = (float) Math.sqrt(Math.max(0.0f, sm - m*m));//sqrt(E(X^2) - E(X)*E(X))
                return new float[]{ std, m };
            }
        };
        if(sync) result.get();
        return result;
    }   
    //</editor-fold>  
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field [field, row] reduce function">
    //<editor-fold defaultstate="collapsed" desc="field linear">
    public Tensor field_mean(Tensor X) { return field_mean(X, -1); }
    public Tensor field_mean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) ((1.0 * row_length) / X.length);//(1 / field_length) 
        return field_linear(X, row_length, alpha, 0.0f);
    }
    
    public Tensor field_sum(Tensor X) { return field_sum(X, -1); } 
    public Tensor field_sum(Tensor X, int row_length) { return field_linear(X, row_length, 1.0f, 0.0f); }//sum(X)
    public Tensor field_sum(Tensor X, float alpha, int row_length) {//sum(alpha*X)
        return field_linear(X, row_length, alpha, 0.0f);
    }
   
    public Tensor field_linear(Tensor X, float alpha, float beta) { return field_linear(X, -1, alpha, beta); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_linear(Tensor X, int row_length, float alpha, float beta) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? //this 2D mem_structure can save memory in padding0-cases
                new int[]{ height, width } :
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_linear(Y.address, 
                X.address, alpha, beta,
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field linear2">
    public Tensor field_add(Tensor X1, Tensor X2) { return field_linear2(X1, X2, -1, 1.0f, 1.0f, 0.0f); }
    public Tensor field_add(Tensor X1, Tensor X2, int row_length) {
        return field_linear2(X1, X2, row_length, 1.0f, 1.0f, 0.0f);
    } 
    
    public Tensor field_sub(Tensor X1, Tensor X2) { return field_linear2(X1, X2, -1, 1.0f, -1.0f, 0.0f); }
    public Tensor field_sub(Tensor X1, Tensor X2, int row_length) {
        return field_linear2(X1, X2, row_length, 1.0f, -1.0f, 0.0f);
    } 
    
    public Tensor field_linear2(Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return field_linear2(X1, X2, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_linear2(Tensor X1, Tensor X2, int row_length, 
            float alpha, float beta, float gamma)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            must_greater_equal(X1.ndim(), "X1.ndim", 2);
            must_greater_equal(X2.ndim(), "X2.ndim", 2);
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X1.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_linear2(Y.address, 
                X1.address, X2.address,
                alpha, beta, gamma,
                X1.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field quadratic"> 
    public Tensor field_sqmean(Tensor X) { return field_sqmean(X, -1); }
    public Tensor field_sqmean(Tensor X, int row_length) { 
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) ((1.0 * row_length) / X.length);//(1 / field_length)
        return field_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor field_sqsum(Tensor X) { return field_sqsum(X, -1); }
    public Tensor field_sqsum(Tensor X, int row_length) { return field_quadratic(X, row_length, 1.0f, 0.0f, 0.0f); }//sum(X^2)
    public Tensor field_sqsum(Tensor X, int row_length, float alpha) {//sum(alpha*X^2)
        return field_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor field_quadratic(Tensor X, float alpha, float beta, float gamma) {
        return field_quadratic(X, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_quadratic(Tensor X, int row_length, float alpha, float beta, float gamma) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1 ? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_quadratic(Y.address,
                X.address, alpha, beta, gamma, 
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field quadratic2">
    public Tensor field_mulmean(Tensor X1, Tensor X2) { return field_mulmean(X1, X2, -1); }
    public Tensor field_mulmean(Tensor X1, Tensor X2, int row_length) {
        if(row_length == -1) row_length = X1.lastDim();
        float alpha = (float) ((1.0 * row_length) / X1.length);//(1/ field_length)
        return field_quadratic2(X1, X2, row_length, 0.0f, alpha, 0.0f, 0.0f,  0.0f, 0.0f);
    }
    
    public Tensor field_mul(Tensor X1, Tensor X2) { return field_mul(X1, X2, -1); }
    public Tensor field_mul(Tensor X1, Tensor X2, int row_length) {//sum(X1*X2)
        return field_quadratic2(X1, X2, row_length, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    public Tensor field_mul(Tensor X1, Tensor X2, int row_length, float alpha) {//sum(alpha * X1*X2)
        return field_quadratic2(X1, X2, row_length, 0.0f, alpha, 0.0f, 0.0f,  0.0f, 0.0f);
    }
    
    public Tensor field_quadratic2(Tensor X1, Tensor X2,
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return field_quadratic2(X1, X2, -1, k11, k12, k22, k1, k2, C);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_quadratic2(Tensor X1, Tensor X2, int row_length, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) { 
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            must_greater_equal(X1.ndim(), "X1.ndim", 2);
            must_greater_equal(X2.ndim(), "X2.ndim", 2);
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X1.lastDim(), height = row_length / width;
        int[] dimY = (height > 1 ? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_quadratic2(Y.address,
                X1.address, X2.address,
                k11, k12, k22, 
                k1, k2, C, 
                X1.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field linear_quadratic">
    public Tensor[] field_mean_sqmean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) ((1.0 * row_length) / X.length);//(1 / field_length)
        return field_linear_quadratic(X, row_length,
                alpha, 0.0f,//mean = Y1 = field_sum: X / field_length
                alpha, 0.0f, 0.0f);//squareMean = Y2 = field_sum: X^2 / field_length
    }
    
    public Tensor[] field_sum_sqsum(Tensor X) { return field_sum_sqsum(X, -1); }
    public Tensor[] field_sum_sqsum(Tensor X, int row_length) {
        return field_linear_quadratic(X, row_length, 
                1.0f, 0.0f,//Y1 = field_sum: X
                1.0f, 0.0f, 0.0f);//Y2 = field_sum: X^2
    }
    
    public Tensor[] field_linear_quadratic(Tensor X, 
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2) {
        return field_linear_quadratic(X, -1,
                alpha1, beta1, 
                alpha2, beta2, gamma2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_linear_quadratic(Tensor X, int row_length,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X2"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor Y1 = this.empty(dimY);
        Tensor Y2 = this.empty(dimY);
        
        Syncer sc = core.field_linear_quadratic(
                Y1.c().address,//result0
                Y2.c().address,//result1
                X.address, 
                alpha1, beta1,
                alpha2, beta2, gamma2,
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { Y1.setSyncer(sc); Y2.setSyncer(sc); }
        return new Tensor[]{ Y1, Y2 };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field max\minValue">
    public Tensor field_max(Tensor X) { return field_max(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_max(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }

        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_max(Y.address,
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor field_min(Tensor X) { return field_min(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_min(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_min(Y.address,
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field max\minValue indexed">
    public Tensor[] field_max_indexed(Tensor X) { return field_max_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_max_indexed(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY);
        Tensor Index = this.empty_int32(dimY);
        
        Syncer sc = core.field_max_indexed(
                Y.c().address,//result0
                Index.c().address,//result1
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[]{ Y, Index };
    }
    
    public Tensor[] field_min_indexed(Tensor X) { return field_min_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_min_indexed(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor Y = this.empty(dimY);
        Tensor Index = this.empty_int32(dimY);
        
        Syncer sc = core.field_min_indexed(
                Y.c().address,//result0
                Index.c().address,//result1
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[] { Y, Index };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field_var_mean">  
    public Tensor field_var(boolean unbiased, Tensor X) { return field_var(unbiased, X, - 1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_var(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width }: new int[]{ width });
        Tensor var  = this.empty(dimY);
        Tensor mean = this.empty(dimY);
        
        Syncer sc = core.field_var_mean(unbiased,
                var.c().address, //result0
                mean.c().address,//result1
                X.address, 
                X.length, row_length, width);
        
        if(sync) { sc.sync(); delete(mean); }
        else var.setSyncer(Syncer.dual(sc, ()->{ delete(mean); }));
        return var;
    }
    
    public Tensor[] field_var_mean(boolean unbiased, Tensor X) { return field_var_mean(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_var_mean(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor var  = this.empty(dimY);
        Tensor mean = this.empty(dimY);
        
        Syncer sc = core.field_var_mean(unbiased,
                var.c().address, //result0
                mean.c().address,//result1
                X.address, 
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { var.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { var, mean };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_std_mean">  
    public Tensor field_std(boolean unbiased, Tensor X) { return field_std(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_std(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width }: new int[]{ width });
        Tensor std  = this.empty(dimY);
        Tensor mean = this.empty(dimY);
        
        Syncer sc = core.field_std_mean(unbiased,
                std.c().address, //result0
                mean.c().address,//result1
                X.address,
                X.length, row_length, width);
        
        if(sync) { sc.sync(); delete(mean); }
        else std.setSyncer(Syncer.dual(sc, ()->{ delete(mean); })); 
        return std;
    }
    
    public Tensor[] field_std_mean(boolean unbiased, Tensor X) { return field_std_mean(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_std_mean(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); must_greater_equal(X.ndim(), "X.ndim", 2); }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? new int[]{ height, width } : new int[]{ width });
        Tensor std  = this.empty(dimY);
        Tensor mean = this.empty(dimY);
        
        Syncer sc = core.field_std_mean(unbiased,
                std.c().address, //result0
                mean.c().address,//result1
                X.address,
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { std.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { std, mean };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="center [dim0, dim1, dim2] reduce function">
    //<editor-fold defaultstate="collapsed" desc="center_reduce_param_check">
    private void center_reduce_param_check(Tensor X, String name, int dim0, int dim2) {
        if(X.ndim() < 3) throw new RuntimeException(String.format(
                "%s.ndim { got %d } must >= 3", name, X.ndim())); 
        if(X.length % dim0 != 0) throw new RuntimeException(String.format(
                "%s.length { got %d } %% dim0 { got %d } != 0", name, X.length, dim0));
        if(X.length % dim2 != 0) throw new RuntimeException(String.format(
                "%s.length { got %d } %% dim2 { got %d } != 0", name, X.length, dim2));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="center_linear">
    public Tensor center_mean(Tensor X) { return center_mean(X, -1, -1); }
    public Tensor center_mean(Tensor X, int dim0, int dim2) { 
        if(dim0 == -1) dim0 = X.firstDim();
        if(dim2 == -1) dim2 = X.lastDim();
        int n = X.length / (dim0 * dim2);
        float alpha = (float) (1.0 / n);
        return center_linear(X, dim0, dim2, alpha, 0);
    }
    
    public Tensor center_sum(Tensor X) { return center_sum(X, -1, -1); }
    public Tensor center_sum(Tensor X, int dim0, int dim2) { return center_linear(X, dim0, dim2, 1.0f, 0.0f); }//sum(X)
    public Tensor center_sum(Tensor X, int dim0, int dim2, float alpha) {//sum(alpha*X)
        return center_linear(X, dim0, dim2, alpha, 0.0f);
    }
    
    public Tensor center_linear(Tensor X, float alpha, float beta) { return center_linear(X, -1, -1, alpha, beta); }
    @Passed("CudaFloat32EngieBase")//sum(alpha*X + beta)
    public Tensor center_linear(Tensor X, int dim0, int dim2,
            float alpha, float beta) 
    {
        if(dim0 == -1) dim0 = X.firstDim();
        if(dim2 == -1) dim2 = X.lastDim();
        if(check) {//X[dim0, dim1, dim2] -> Y[dim0, dim2]
            require_dtype(X, "X");
            center_reduce_param_check(X, "X", dim0, dim2);
        }
       
        int length = dim0 * dim2;
        int width = X.lastDim(), height = length / width;
        Tensor Y = this.empty(height, width);
        
        int dim1 = X.length / length;
        Syncer sc = core.center_linear(Y.c().address, 
                X.address, 
                alpha, beta, 
                dim0, dim1, dim2, 
                width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="center_quadratic">
    public Tensor center_sqmean(Tensor X) { return center_sqmean(X, -1, -1); }
    public Tensor center_sqmean(Tensor X, int dim0, int dim2) { 
        if(dim0 == -1) dim0 = X.firstDim();
        if(dim2 == -1) dim2 = X.lastDim();
        int n = X.length / (dim0 * dim2);
        float alpha = (float) (1.0 / n);
        return center_quadratic(X, dim0, dim2, alpha, 0, 0);
    }
    
    public Tensor center_sqsum(Tensor X) { return center_sqsum(X, -1, -1); }
    public Tensor center_sqsum(Tensor X, int dim0, int dim2) { return center_quadratic(X, dim0, dim2, 1.0f, 0.0f, 0.0f); }//sum(X^2)
    public Tensor center_sqsum(Tensor X, int dim0, int dim2, float alpha) {//sum(alpha*X^2)
        return center_quadratic(X, dim0, dim2, alpha, 0.0f, 0.0f);
    }
    
    public Tensor center_quadratic(Tensor X, float alpha, float beta, float gamma) {
        return center_quadratic(X, -1, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")//sum(alpha*X^2 + beta*X + gamma)
    public Tensor center_quadratic(Tensor X, int dim0, int dim2,
            float alpha, float beta, float gamma) 
    {
        if(dim0 == -1) dim0 = X.firstDim();
        if(dim2 == -1) dim2 = X.lastDim();
        if(check) {//X[dim0, dim1, dim2] -> Y [dim0, dim2]
            require_dtype(X, "X");
            center_reduce_param_check(X, "X", dim0, dim2);
        }
       
        int length = dim0 * dim2;
        int width = X.lastDim(), height = length / width;
        Tensor Y = this.empty(height, width);
        
        int dim1 = X.length / length;
        Syncer sc = core.center_quadratic(Y.c().address, 
                X.address, 
                alpha, beta, gamma, 
                dim0, dim1, dim2, 
                width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="center_quadratic2">
    public Tensor center_mul(Tensor X1, Tensor X2) { return center_quadratic2(X1, X2, -1, -1, 0, 1.0f, 0, 0, 0, 0); }
    public Tensor center_mul(Tensor X1, Tensor X2, int dim0, int dim2) {//X1 * X2
        return center_quadratic2(X1, X2, dim0, dim2, 0, 1.0f, 0, 0, 0, 0);
    }
    
    public Tensor center_sqadd(Tensor X1, Tensor X2) { return center_quadratic2(X1, X2, -1, -1, 1.0f, 0, 1.0f, 0, 0, 0); }
    public Tensor center_sqadd(Tensor X1, Tensor X2, int dim0, int dim2) {//X1^2 + X2^2
        return center_quadratic2(X1, X2, dim0, dim2, 1.0f, 0, 1.0f, 0, 0, 0);
    }
    
    public Tensor center_sqsub(Tensor X1, Tensor X2) { return center_quadratic2(X1, X2, -1, -1, 1.0f, 0, -1.0f, 0, 0, 0); }
    public Tensor center_sqsub(Tensor X1, Tensor X2, int dim0, int dim2) {//X1^2 - X2^2
        return center_quadratic2(X1, X2, dim0, dim2, 1.0f, 0, -1.0f, 0, 0, 0);
    }
    
    public Tensor center_quadratic2(Tensor X1, Tensor X2,
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return center_quadratic2(X1, X2, -1, -1, k11, k12, k22, k1, k2, C);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor center_quadratic2(Tensor X1, Tensor X2, int dim0, int dim2,
            float k11, float k12, float k22,
            float k1, float k2, float C) 
    {
        if(dim0 == -1) dim0 = X1.firstDim();
        if(dim2 == -1) dim2 = X1.lastDim();
        if(check) {//X[dim0, dim1, dim2] -> Y [dim0, dim2]
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            center_reduce_param_check(X1, "X1", dim0, dim2);
            center_reduce_param_check(X2, "X2", dim0, dim2);
            equals_valueStructure(X1, "X1", X2, "X2");
        }
       
        int length = dim0 * dim2;
        int width = X1.lastDim(), height = length / width;
        Tensor Y = this.empty(height, width);
        
        int dim1 = X1.length / length;
        Syncer sc = core.center_quadratic2(Y.c().address,
                X1.address, X2.address, 
                k11, k12, k22, 
                k1, k2, C, 
                dim0, dim1, dim2, 
                width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row [field, row] reduce function">
    //<editor-fold defaultstate="collapsed" desc="row_reduce_param_check">
    protected void row_reduce_param_check(Tensor X, int row_length) {
        if(X.ndim() <= 1) throw new IllegalArgumentException(String.format(
                "X.ndim { got %d } must > 1", X.ndim()));
        if(X.length % row_length != 0) throw new IllegalArgumentException(String.format(
                "X.length { got %d } %% row_length { got %d } != 0", X.length, row_length));
    }
    
    private void row_reduce_param_check(Tensor X, String name, int row_length) {
        if(X.ndim() <= 1) throw new IllegalArgumentException(String.format(
                "%s.ndim { got %d } must > 1", name, X.ndim()));
        if(X.length % row_length != 0) throw new IllegalArgumentException(String.format(
                "%s.length { got %d } %% row_length { got %d } != 0", name, X.length, row_length));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row linear">
    public Tensor row_mean(Tensor X) { return row_mean(X, -1); }
    public Tensor row_mean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) (1.0 / row_length);
        return row_linear(X, row_length, alpha, 0.0f);//sum(X / row_length)
    }
    
    public Tensor row_sum(Tensor X) { return row_sum(X, -1); }
    public Tensor row_sum(Tensor X, int row_length) { return row_linear(X, row_length, 1.0f, 0.0f); }
    public Tensor row_sum(Tensor X, int row_length, float alpha) {
        return row_linear(X, row_length, alpha, 0.0f);
    }
    
    public Tensor row_linear(Tensor X, float alpha, float beta) { return row_linear(X, -1, alpha, beta); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_linear(Tensor X, int row_length, float alpha, float beta) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_linear(Y.address,
                X.address, alpha, beta,
                field_length, row_length,
                X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear2">
    public Tensor row_add(Tensor X1, Tensor X2) { return row_linear2(X1, X2, -1, 1.0f, 1.0f); }
    public Tensor row_add(Tensor X1, Tensor X2, int row_length) {
        return row_linear2(X1, X2, row_length, 1.0f, 1.0f);
    }
    
    public Tensor row_sub(Tensor X1, Tensor X2) { return row_linear2(X1, X2, -1, 1.0f, -1.0f); }
    public Tensor row_sub(Tensor X1, Tensor X2, int row_length) {
        return row_linear2(X1, X2, row_length, 1.0f, -1.0f);
    }
    
    public Tensor row_linear2(Tensor X1, Tensor X2, float alpha, float beta, float gamma) {
        return row_linear2(X1, X2, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_linear2(Tensor X1, Tensor X2, int row_length,
            float alpha, float beta, float gamma)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2");
            row_reduce_param_check(X1, "X1", row_length);
            row_reduce_param_check(X2, "X2", row_length);
        }
        
        int field_length = X1.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_linear2(Y.address, 
                X1.address, X2.address,
                alpha, beta, gamma,
                field_length, row_length,
                X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row quadratic">
    public Tensor row_sqmean(Tensor X) { return row_sqmean(X, -1); }
    public Tensor row_sqmean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) (1.0 / row_length);//sum(X^2)/row_length = sum(X^2 / row_length)
        return row_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor row_square(Tensor X) { return row_square(X, -1); }
    public Tensor row_square(Tensor X, int row_length) { return row_quadratic(X, row_length, 1.0f, 0.0f, 0.0f); }
    public Tensor row_square(Tensor X, int row_length, float alpha) {
        return row_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor row_quadratic(Tensor X, float alpha, float beta, float gamma) {
        return row_quadratic(X, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_quadratic(Tensor X, int row_length, float alpha, float beta, float gamma) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_quadratic(Y.address,
                X.address, alpha, beta, gamma,
                field_length, row_length,
                X.lastDim()); 
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row quadratic2">
    public Tensor row_mulmean(Tensor X1, Tensor X2) { return row_mulmean(X1, X2, -1); }
    public Tensor row_mulmean(Tensor X1, Tensor X2, int row_length) {//sum(X) / row_length
        if(row_length == -1) row_length = X1.lastDim();
        float alpha = (float) (1.0 / row_length);
        return row_quadratic2(X1, X2, row_length,
                0, alpha, 0,
                0, 0, 0);
    }
    
    public Tensor row_mul(Tensor X1, Tensor X2) { return row_mul(X1, X2, -1); }
    public Tensor row_mul(Tensor X1, Tensor X2, int row_length) {//sum(X1 * X2)
        return row_quadratic2(X1, X2, row_length, 0, 1.0f, 0, 0, 0, 0);
    }
    public Tensor row_mul(Tensor X1, Tensor X2, int row_length, float alpha) {//sum(alpha * X1 * X2)
        return row_quadratic2(X1, X2, row_length, 0, alpha, 0, 0, 0, 0);
    }
    
    public Tensor row_quadratic2(Tensor X1, Tensor X2,
            float k11, float k12, float k22, 
            float k1, float k2, float C) {
        return row_quadratic2(X1, X2, -1, k11, k12, k22, k1, k2, C);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_quadratic2(Tensor X1, Tensor X2, int row_length,
            float k11, float k12, float k22, 
            float k1, float k2, float C)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) {
            require_dtype(X1, "X1"); require_dtype(X2, "X2");
            equals_valueStructure(X1, "X1", X2, "X2");
            row_reduce_param_check(X1, "X1", row_length);
            row_reduce_param_check(X2, "X2", row_length);
        }
        
        int field_length = X1.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_quadratic2(Y.address, 
                X1.address, X2.address, 
                k11, k12, k22,
                k1, k2, C, 
                field_length, row_length, 
                X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row linear_quadratic">
    public Tensor[] row_mean_sqmean(Tensor X) { return row_mean_sqmean(X, -1); }
    public Tensor[] row_mean_sqmean(Tensor X, int row_length) {//sum(X / row_length)
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) (1.0 / row_length);
        return row_linear_quadratic(X, row_length,
                alpha, 0.0f,//mean = Y1 = row_sum: X / row_length
                alpha, 0.0f, 0.0f);//squareMean = Y2 = row_sum: X^2 / row_length
    }
    
    public Tensor[] row_sum_sqsum(Tensor X) { return row_sum_sqsum(X, -1); }
    public Tensor[] row_sum_sqsum(Tensor X, int row_length) {
        return row_linear_quadratic(X, row_length,
                1.0f, 0.0f,//Y1 = row_sum: X
                1.0f, 0.0f, 0.0f);//Y2 = row_sum: X^2
    }
    
    public Tensor[] row_linear_quadratic(Tensor X,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2) {
        return row_linear_quadratic(X, -1, 
                alpha1, beta1, 
                alpha2, beta2, gamma2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_linear_quadratic(Tensor X, int row_length,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y1 = this.empty(field_length);//Y = Tensor1D[field_length]
        Tensor Y2 = this.empty(field_length);//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_linear_quadratic(Y1.c().address, Y2.c().address,
                X.address, 
                alpha1, beta1,//Y1 = alpha1*X + beta1
                alpha2, beta2, gamma2,//Y2 = alpha2*X^2 + beta2*X + gamma2
                field_length, row_length,
                X.lastDim());
        
        if(sync) sc.sync(); else { Y1.setSyncer(sc); Y2.setSyncer(sc); }
        return new Tensor[]{ Y1, Y2 };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row_var_mean">
    public Tensor row_var(boolean unbiased, Tensor X) { return row_var(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_var(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length);  }
        
        int field_length = X.length / row_length;
        Tensor var  = this.empty(field_length);
        Tensor mean = this.empty(field_length);
        
        Syncer sc = core.row_var_mean(unbiased,
                var.c().address, //result0
                mean.c().address,//result1
                X.address, 
                field_length, row_length,
                X.lastDim());
        if(sync) { sc.sync(); delete(mean); }
        else var.setSyncer(Syncer.dual(sc, ()-> { delete(mean); }));
        return var;
    }    
    
    public Tensor[] row_var_mean(boolean unbiased, Tensor X) { return row_var_mean(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_var_mean(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor var  = this.empty(field_length);
        Tensor mean = this.empty(field_length);
        
        Syncer sc = core.row_var_mean(unbiased,
                var.c().address, //result0
                mean.c().address,//result1
                X.address, 
                field_length, row_length,
                X.lastDim());
        if(sync) sc.sync(); else { var.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { var, mean };
    }    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_std_mean">
    public Tensor row_std(boolean unbiased, Tensor X) { return row_std(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_std(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor std  = this.empty(field_length);
        Tensor mean = this.empty(field_length);
        
        Syncer sc = core.row_std_mean(unbiased,
                std.c().address, //result0
                mean.c().address,//result1
                X.address, 
                field_length, row_length,
                X.lastDim());
        if(sync) { sc.sync(); delete(mean); }
        else std.setSyncer(Syncer.dual(sc, ()-> { delete(mean); }));
        return std;
    }    
    
    public Tensor[] row_std_mean(boolean unbiased,Tensor X)  { return row_std_mean(unbiased, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_std_mean(boolean unbiased, Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor std  = this.empty(field_length);
        Tensor mean = this.empty(field_length);
        
        Syncer sc = core.row_std_mean(unbiased,
                std.c().address, //result0
                mean.c().address,//result1
                X.address, 
                field_length, row_length,
                X.lastDim());
        if(sync) sc.sync(); else { std.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { std, mean };
    }    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row max\minValue">
    public Tensor row_max(Tensor X) { return row_max(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_max(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length);  }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_max(Y.address, 
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor row_min(Tensor X) { return row_min(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_min(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length);  }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_min(Y.address, 
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row max\minValue indexed">
    public Tensor row_max_index(Tensor X) { return row_max_index(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_max_index(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length);//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length);
        
        Syncer sc = core.row_max_indexed(
                Y.c().address,//result0
                Index.c().address,//result1
                X.address,
                field_length, row_length, 
                X.lastDim());
        
        if(sync) { sc.sync(); delete(Y); }
        else Index.setSyncer(Syncer.dual(sc, ()-> {delete(Y); }));
        return Index;
    }
    
    public Tensor[] row_max_indexed(Tensor X) { return row_max_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_max_indexed(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length);//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length);
        
        Syncer sc = core.row_max_indexed(
                Y.c().address,//result0
                Index.c().address,//tesult1
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[] { Y, Index };
    }
    
    public Tensor row_min_index(Tensor X) { return row_min_index(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_min_index(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length);//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length);
        
        Syncer sc = core.row_min_indexed(
                Y.c().address,//result0
                Index.c().address,//result1
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) { sc.sync(); delete(Y); }
        else Y.setSyncer(Syncer.dual(sc, ()-> { delete(Y); }));
        return Index;
    }
    
    public Tensor[] row_min_indexed(Tensor X) { return row_min_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_min_indexed(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { require_dtype(X, "X"); row_reduce_param_check(X, row_length); }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length);//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length);
        
        Syncer sc = core.row_min_indexed(
                Y.c().address, 
                Index.c().address,
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[] { Y, Index };
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Neural: Extended">
    public void garbage_collect(Unit unit) { unit.gc(); }
    public void delete(Unit unit) { unit.delete(); }
    //</editor-fold>
}
