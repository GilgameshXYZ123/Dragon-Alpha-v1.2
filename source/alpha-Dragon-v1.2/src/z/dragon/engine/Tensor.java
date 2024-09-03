/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.io.Serializable;
import z.dragon.nn.core.Trace;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import z.util.math.vector.Vector;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.ImageEngine.ImageAffiner;
import z.dragon.engine.ImageEngine.RandomImageAffiner;
import z.dragon.engine.Syncer.DualSyncer;
import z.dragon.engine.Syncer.FollowSyncer;
import z.dragon.engine.Syncer.RemoteSync;
import z.dragon.nn.core.UnitCore;

/**
 * @author Gilgamesh
 * {4, 3, 2, 1}, firstDim -> lastDim
 */
@SuppressWarnings("unchecked")
public class Tensor implements StateValue, Serializable {
    private static final long serialVersionUID = 11231231234666L;
    //<editor-fold defaultstate="collapsed" desc="static class: TensorList">
    public static class TensorList extends ArrayList<Tensor> {
        private static final long serialVersionUID = 615120712446L;
        
        public TensorList(int init_capacity) { super(init_capacity);}
        public TensorList() { super(); }
        
        @Override public synchronized void clear() { super.clear(); }
        
        @Override
        public synchronized final boolean add(Tensor ts) {
            return ((ts == null || ts.is_null()) ? false : super.add(ts));
        }

        public synchronized  final boolean addAll(Tensor[] arr) {
            boolean flag = false;
            for(Tensor ts : arr) {
                if(ts == null || ts.is_null()) continue;
                flag |= super.add(ts);
            }
            return flag;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: TenorSet">
    public static class TensorSet extends HashSet<Tensor>  {
        private static final long serialVersionUID = 1L;
        
        public TensorSet() { super(); }
        public TensorSet(int init_capacity) { super(init_capacity); }

        @Override public synchronized void clear() { super.clear(); }
        
        @Override
        public synchronized final boolean add(Tensor ts) {
            return ((ts == null || ts.is_null()) ? false : super.add(ts));
        }
        
        public synchronized final boolean add(Tensor...ts) {
            if(ts == null || ts.length == 0) return false;
            boolean result = true;
            for(Tensor t : ts) {
                if(t == null || t.is_null()) result &= false;
                else result &= super.add(t);
            }
            return result;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: TensorMap">
    public static class TensorMap<K> extends HashMap<K, Tensor> {
        private static final long serialVersionUID = 1L;
        
        public TensorMap() { super(); }
        public TensorMap(int initialCapacity) { super(initialCapacity); }

        @Override public synchronized void clear() { super.clear(); }
        
        @Override
        public synchronized final Tensor put(K key, Tensor value) {
            return ((value == null || value.is_null()) ? null :  super.put(key, value)); 
        }
    }
    //</editor-fold>
    
    public static final int MAX_NDIM = 4;
    public static final int MAX_LENGTH_4X = Integer.MAX_VALUE;

    //<editor-fold defaultstate="collapsed" desc="member params">
    transient protected final Engine eg;
    protected String dataType;
    protected long address;
    protected int[] dim;//dimensions, from the first to last, the highest dimension is the first
    
    protected long mem_size;//mem_size alloced by Engine
    protected int length_4x;//length_algined <= mem_length, as least used mem_length
    protected int lengthv;//mem_stride * mem_height, mem_stride = (mem_width + 3) >> 2 << 2
    protected int length;//mem_width * mem_height
    //</editor-fold>
    
    Tensor(Engine engine, String dataType, int[] dim) { this(true, engine, dataType, dim); }
    Tensor(boolean check, Engine engine, String dataType, int[] dim) {
        this.eg = engine;
        this.dataType = dataType;
        this.setDim(check, dim);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Baisc-Functions: Engine & Address"> 
    public final Engine engine() {return eg;}
    public final long address() {return address; }
    
    public final String dataType() { return dataType; }
    public final boolean is_dtype() { return eg.is_dtype(this); }
    public final boolean is_int32() { return eg.is_int32(this); }
    public final boolean is_int8() { return eg.is_int8(this); }
   
    transient protected String msg = null;
    public final String message() { return msg; }
    public final Tensor message(String message) { this.msg = message; return this; }
    
    public static boolean isNull(Tensor ts) { return ts == null || ts.address == 0L; }
    public final boolean is_null() { return address == 0L; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { dataType = ").append(dataType);
        sb.append(", dim = ").append(Arrays.toString(dim));
        sb.append(", [length, mem_size, address] = [")
                .append(length).append(", ")
                .append(mem_size).append(", ")
                .append(address).append(']');
        sb.append(", need_gards = ").append(need_grad);
        if(msg != null) sb.append(", messge = ").append(msg);
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        this.append(sb);
        return sb.toString();
    }
    
    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        eg.delete_core(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Dimension - Memory Structure"> 
    public final int ndim() { return dim.length; }
    public final int[] dim() { return dim; }
    
    public final int dim(int index) {//index = -1: the last dim, dim.length - 1
        if(index < 0) index = dim.length + index;//index = -2: the second last dim, dim.length - 2
        return dim[index];
    }
    
    public final int firstDim() { return dim[0]; }
    public final int lastDim() { return dim[dim.length - 1]; }
    
    public final int length() { return length; }
    public final int lengthv() { return lengthv; }
    public final int length_4x() { return length_4x; }
    public final long memory_size() { return mem_size; }
    
    protected void bindMemory(long[] block) {
        this.mem_size = block[0];
        this.address = block[1];
    }
      
    protected void copy_memoryMetaData_and_deleteSrc(Tensor X) {
        synchronized(X) {//X.{carrier, mod_count} remain  unchanged
            this.address = X.address;
            this.dim = X.dim;
            this.mem_size = X.mem_size;
            this.length_4x = X.length_4x;
            this.lengthv = X.lengthv;
            this.length = X.length;
            X.address = 0;//As X.finalize may be called
        }
    }   
    
    protected void copy_memoryMetaData(Tensor X) {
        synchronized(X) {
            this.address = X.address;
            this.dim = X.dim;
            this.mem_size = X.mem_size;
            this.length_4x = X.length_4x;
            this.lengthv = X.lengthv;
            this.length = X.length;
        }
    }
    
    protected final void setDim(boolean check, int... dim) {
        if(check) {
            if(dim == null) throw new NullPointerException("dim for Tensor can not be null");
            if(dim.length == 0) throw new IllegalArgumentException("Tensor.ndim == 0");
            if(dim.length > MAX_NDIM) throw new IllegalArgumentException(String.format(
                    "Tensor.ndim { got %d } > MAX_NDIM { got %d }", dim.length, MAX_NDIM ));
            for(int i=0; i<dim.length; i++) if(dim[i] <= 0) throw new IllegalArgumentException(String.format(
                    "dim[%d] { got %d } must be positive", i, dim[i]));
        }
        
        int firstDim = dim[0], firstDim_4x = ((firstDim + 3) >> 2) << 2;
        int Length_4x, Lengthv, Length;
        if(dim.length == 1) {
            Length_4x = firstDim_4x;
            Lengthv   = firstDim_4x;
            Length    = firstDim;
        }
        else {
            int lastDim = dim[dim.length-1], lastDim_4x = ((lastDim + 3) >> 2) << 2;
            int midlen = 1;
            for(int i = 1; i< dim.length - 1; i++) midlen *= dim[i];
            Length_4x = firstDim_4x * midlen * lastDim_4x;
            Lengthv   = firstDim    * midlen * lastDim_4x;
            Length    = firstDim    * midlen * lastDim;
        }
        
        if(Length_4x <= 0 || Length_4x > MAX_LENGTH_4X) 
            throw new IllegalArgumentException("the length of Tensor is exceed the upper limit");
        
        this.length_4x = Length_4x;
        this.lengthv = Lengthv;
        this.length = Length;
        this.dim = Vector.arrayCopy(dim);
    }
    
    public boolean dimEquals(int... dim) { return Arrays.equals(this.dim, dim); }
    public boolean dimEquals(Tensor ts) { return Arrays.equals(dim, ts.dim); }
    
    public boolean isMemAligned() {
        return (dim[0] & 3) != 0 || (dim[dim.length - 1] & 3) != 0;
    }
    
    public boolean memSturcEquals(Tensor ts)  {return memStrucEquals(ts.length, ts.dim);}
    boolean memStrucEquals(int length2, int... dim2) {
        if(length2 != this.length) return false;//ts.length != this.length
        
        int ndim1 = dim.length, ndim2 = dim2.length;
        int firstDim1 = dim[0], lastDim1 = dim[dim.length - 1]; 
        int firstDim2 = dim2[0], lastDim2 = dim2[dim2.length - 1];
        
        //if: this and another is not memalgined, and this.length = another.length,
        if((firstDim1 & 3) == 0 && (lastDim1 & 3) == 0 &&
           (firstDim2 & 3) == 0 && (lastDim2 & 3) == 0) return true;
        
        if(ndim1>1 && ndim2>1) {//ND ND
            //the mem alignment only effects the first and the last dim
            return firstDim2 == firstDim1 && lastDim2 == lastDim1;
        }
        if(ndim1>1) {//ND 1D
            return (firstDim1 & 3) == 0 && (lastDim1 & 3) == 0 &&
                   (length2 & 3) == 0;
        }
        if(ndim2>1) {//1D ND
            return (firstDim2 & 3) == 0 && (lastDim2 & 3) == 0 &&
                   (this.length & 3) == 0;
        }
        return true;//1D 1D
    }
    
    boolean valueStrucEquals(int length2, int... dim2)  {
        if(length2 != this.length) return false;//ts.length != this.length
        return dim[dim.length - 1] == dim2[dim2.length - 1];
        //we have: lengthv2 % mem_width2 == 0
        //As: ts.length != this.length
        //    ts.mem_width = this.mem_width
        //So: ts.lengthv = ts.length/ts.mem_width * ts.mem_stride
        //    this.lengthv = this.length/this.mem_width * this.mem_stride
        //we have ts.lengthv = this.lengthv
    }
    
    public boolean valueStrucEquals(Tensor ts) {
        if(ts.lengthv != this.lengthv) return false;
        if(ts.lastDim() == this.lastDim()) return true;
        return !ts.isMemAligned() && !this.isMemAligned();
    }
    
    public void requireValueStrucEquals(Tensor ts, String name1, String name2) {
        if(!valueStrucEquals(ts)) throw new IllegalArgumentException(
                name1 + ".valueStructure is different from that of" + name2);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Syncer: extended">
    transient protected volatile Syncer syncer;//to get the result of the last computation
    public final Syncer syncer() { return syncer; }
    public final synchronized void setSyncer(Syncer sc) { this.syncer = sc; }
    
    public final synchronized Tensor c() {
        if (syncer != null) {
            Syncer sc = syncer;
            syncer = null;
            sc.sync();
        }
        return this;
    }
    
    public Tensor remote_sync() { RemoteSync.sync(this); return this; }
    public Tensor remote_delete() { RemoteSync.delete(this); return this; }
    
    public Tensor dual(Syncer after) {
        synchronized(this) { Syncer before = syncer; syncer = new DualSyncer(before, after); }
        return this;
    }
    
    public Tensor follow(Tensor ts) {
        synchronized(this) { syncer = new FollowSyncer(ts); }
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Engine: extended">
    //<editor-fold defaultstate="collapsed" desc="value-auxilary">
    public float[][] value2D()  { return value2D(this.lastDim());}
    public float[][] value2D(int lastDim) {
        if(this.ndim() < 2) throw new IllegalArgumentException("ndim must >= 2");
        float[] value = eg.valueOf(this);
        int width = lastDim; if(width == -1) width =  this.lastDim();
        int height = this.length / width;
        return Vector.to2D(value, height, width);
    }
    
    public Tensor vprintln() { Vector.println(eg.valueOf(this)); return this; }
    public Tensor vprintln(int start, int end) {
        float[] value = eg.valueOf(this);
        Vector.println(value, start, end);
        return this;
    }
    //</editor-fold>
    
    @Override public float[] value() { return eg.valueOf(this); }
    public int[] value_int32() { return eg.valueOf_int32(this); }
    public byte[] value_int8() { return eg.valueOf_int8(this); }
    public byte[] pixel() { return eg.valueOf_int8(this); }
    
    public <T> T raw_data() { return eg.raw_data(this); }
    public <T> T data() { return eg.data(this); }
    
    public static final void sync(Tensor... list) { for (Tensor t : list) if(t != null) t.c(); }
    public static final void sync(Collection<Tensor> arr) { for (Tensor t : arr) if(t != null) t.c(); }

    public static final void delete(Tensor... arr) {
        for (Tensor t : arr) if (t != null && !t.is_null()) t.delete(); 
    }
    public static final void delete(Collection<Tensor> list) {
        for (Tensor t : list) if (t != null && !t.is_null()) t.delete();
    }
    
    public static Tensor[] zero_like(Tensor... arr) {
        Tensor[] zeros = new Tensor[arr.length];
        for(int i=0; i<arr.length; i++) zeros[i] = arr[i].zeros_like();
        return zeros;
    }
    public static Tensor[] zero_like(Parameter... params) {
        Tensor[] zeros = new Tensor[params.length];
        for(int i=0; i<params.length; i++) zeros[i] = params[i].tensor.zeros_like();
        return zeros;
    }
   
    public final boolean check() { return eg.check; }
    public void delete() { eg.delete_core(this); }
    
    public Tensor to_int8(boolean inplace) { return eg.dtype_to_int8(inplace, this); }
    public Tensor to_int32(boolean inplace) { return eg.dtype_to_int32(inplace, this); }
    public Tensor to_dtype(boolean inplace) { return eg.to_dtype(inplace, this); }
    
    public Tensor copy() { return eg.copy(this); }
    public Tensor zero() { return eg.zero(this); }
    public Tensor constant(float value) { return eg.constant(this, value); }
    public Tensor zero_nan(){ return eg.zero_nan(true, this); }
    
    public Tensor empty_like() { return eg.empty_like(this); }
    public Tensor zeros_like() { return eg.zeros_like(this); }
    public Tensor ones_like() { return eg.ones_like(this); }
    public Tensor constant_like(float value) { return eg.constants_like(value, this); }
    
    public Tensor set(Tensor ts) { eg.set(this, ts); return this; }
    public Tensor set(float[] value) { eg.set(this, value); return this; }
    public Tensor set(byte[]  value) { eg.set(this, value); return this; }
    public Tensor set(String line) { eg.set(this, line); return this; }
    public Tensor set(ArrayList<String> lines) { eg.set(this, lines); return this; }
    public Tensor set(StateValue value, boolean partial, String msg) {
        eg.set(this, value, partial, msg); 
        return this;
    }
    
    public Result<Boolean> hasNan() { return eg.hasNan(this); }
    public Result<Float> max() { return eg.straight_max(this); }
    public Result<Float> min() { return eg.straight_min(this); }
    public Result<Float> sum() { return eg.straight_sum(this); }
    public Result<Float> sqsum() { return eg.straight_sqsum(this); }
    public Result<Float> mean() { return eg.straight_mean(this); }
    public Result<Float> var() { return eg.straight_var(this); }
    public Result<Float> std() { return eg.straight_std(this); }
    public Result<Float> equal(Tensor X) { return eg.straight_equal(X, this); }
    public Result<Float> nonzero_percent() { return eg.nonzero_percent(this); }
    public Result<Float> zero_percent() { return eg.zero_percent(this); }
    
    public Result<float[]> std_mean() { return eg.straight_std_mean(this); }
    public Result<float[]> var_mean() { return eg.straight_var_mean(this); }
    
    public Tensor sadd(boolean inplace, float C) { return eg.sadd(inplace, this, C); }
    public Tensor ssub(boolean inplace, float C) { return eg.ssub(inplace, this, C); }
    public Tensor smul(boolean inplace, float C) { return eg.smul(inplace, this, C); }
    public Tensor sdiv(boolean inplace, float C) { return eg.sdiv(inplace, this, C); }
    public Tensor linear(boolean inplace, float alpha, float beta) { 
        return eg.linear(inplace, alpha, this, beta); 
    }
   
    public Tensor square(boolean inplace) { return eg.square(inplace, this); }
    public Tensor quadratic(boolean inplace, float alpha, float beta, float gamma) { 
        return eg.quadratic(inplace, this, alpha, beta, gamma);
    }
   
    public Tensor add(boolean inplace, Tensor X) { return eg.add(inplace, this, X); }
    public Tensor sub(boolean inplace, Tensor X) { return eg.sub(inplace, this, X); }
    public Tensor linear2(boolean inplace, Tensor X, float alpha, float beta, float gamma) { 
        return eg.linear2(inplace, this, X, alpha, beta, gamma); 
    }
  
    public Tensor mul(boolean inplace, Tensor X) { return eg.mul(inplace, this, X); }
    public Tensor mul(boolean inplace, float alpha, Tensor X) { return eg.mul(inplace, alpha, this, X); }
    public Tensor squareAdd(boolean inplace, Tensor X) { return eg.sqadd(inplace, this, X); } 
    public Tensor squareAdd(boolean inplace, Tensor X, float alpha, float beta) { return eg.sqadd(inplace, this, X, alpha, beta); }
    public Tensor quadraitic2(boolean inplace, Tensor X,
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return eg.quadratic2(inplace, this, X,
                k11, k12, k22,
                k1, k2, C);
    }
    
    public Tensor rpl(boolean inplace) { return eg.rpl(inplace, this); }
    public Tensor rpl(boolean inplace, float alpha) { return eg.rpl(inplace, alpha, this); }
    public Tensor div(boolean inplace, Tensor X) { return eg.div(inplace, this, X); }
    public Tensor div(boolean inplace, float alpha, Tensor X) { return eg.div(inplace, alpha, this, X); }
    public Tensor div(boolean inplace, Tensor X, 
            float alpha1, float beta1,
            float alpha2, float beta2, 
            float gamma) {
        return eg.div(inplace, 
                alpha1, this, beta1, 
                alpha2, X, beta2, 
                gamma);
    }
    
    public Tensor min(boolean inplace, float vmin) { return eg.min(inplace, this, vmin); }
    public Tensor max(boolean inplace, float vmax) { return eg.max(inplace, this, vmax); }
    public Tensor min2(boolean inplace, Tensor X) { return eg.min2(inplace, this, X); }
    public Tensor max2(boolean inplace, Tensor X) { return eg.max2(inplace, this, X); }
    public Tensor clip(boolean inplace, float vmin, float vmax) { return eg.clip(inplace, this, vmin, vmax); }

    public Tensor log(boolean inplace) { return eg.log(inplace, this); }
    public Tensor log(boolean inplace, float alpha, float beta) { return eg.log(inplace, alpha, this, beta); }
    public Tensor exp(boolean inplace) { return eg.exp(inplace, this); }
    public Tensor exp(boolean inplace, float alpha, float beta) { return eg.exp(inplace, alpha, this, beta); } 
    
    public Tensor relu(boolean inplace) { return eg.relu(inplace, this); }
    public Tensor leakyRelu(boolean inplace) { return eg.leakyRelu(inplace, this); }
    public Tensor leakyRelu(boolean inplace, float negative_slope) { return eg.leakyRelu(inplace, this, negative_slope); }
    public Tensor softplus(boolean inplace) { return eg.softplus(inplace, this); }

    public Tensor sigmoid(boolean inplace) { return eg.sigmoid(inplace, this); }
    public Tensor tanh(boolean inplace) { return eg.tanh(inplace, this); }
    public Tensor softmax(boolean inplace) { return eg.softmax(inplace, this); }
    public Tensor log_softmax(boolean inplace) { return eg.log_softmax(inplace, this); }
    
    public Tensor sin(boolean inplace) { return eg.sin(inplace, this); }
    public Tensor sin(boolean inplace, float alpha, float beta) { return eg.sin(inplace, alpha, this, beta); }
    public Tensor cos(boolean inplace) { return eg.cos(inplace, this); }
    public Tensor cos(boolean inplace, float alpha, float beta) { return eg.cos(inplace, alpha, this, beta); }

    public Tensor tan(boolean inplace) { return eg.tan(inplace, this); }
    public Tensor tan(boolean inplace, float alpha, float beta) { return eg.tan(inplace, alpha, this, beta); }
    public Tensor cot(boolean inplace) { return eg.cot(inplace, this); }
    public Tensor cot(boolean inplace, float alpha, float beta) { return eg.cot(inplace, alpha, this, beta); }
    
    public Tensor sec(boolean inplace) { return eg.sec(inplace, this); }
    public Tensor sec(boolean inplace, float alpha, float beta) { return eg.sec(inplace, alpha, this, beta); }
    public Tensor csc(boolean inplace) { return eg.csc(inplace, this); }
    public Tensor csc(boolean inplace, float alpha, float beta) { return eg.csc(inplace, alpha, this, beta); }
    
    public Tensor pixel_to_tensor(boolean inplace) { return eg.pixel_to_tensor(inplace, this); } 
    public Tensor tensor_to_pixel(boolean inplace) { return eg.tensor_to_pixel(inplace, this); }
    
    protected Tensor view = null, root = null;
    public Tensor view(boolean inplace, int... dim) { return eg.view(inplace, this, dim); }
    public Tensor view_copy() { return eg.view_copy(this); }
    
    public Tensor reshape(boolean inplace, int... dim) { return eg.reshape(inplace, this, dim); }
    public Tensor transpose(boolean inplace, int dimIdx1, int dimIdx2) { return eg.transpose(inplace, this, dimIdx1, dimIdx2); }
    
    public Tensor pad(boolean inplace, int... p) { return eg.pad(inplace, this, p); }
    public Tensor pad(boolean inplace, int[] p1, int[] p2) { return eg.pad(inplace, this, p2, p1); }
    public Tensor trim(boolean inplace, int... t) { return eg.trim(inplace, this, t); }
    public Tensor trim(boolean inplace, int[] t0, int[] t1) { return eg.trim(inplace, this, t0, t1); }
    
    public Tensor pad2D(boolean inplace, int... p) { return eg.pad2D(inplace, this, p); }
    public Tensor pad2D(boolean inplace, int[] p1, int[] p2) { return eg.pad2D(inplace, this, p2, p1); }
    public Tensor trim2D(boolean inplace, int... t) { return eg.trim2D(inplace, this, t); }
    public Tensor trim2D(boolean inplace, int[] t0, int[] t1) { return eg.trim2D(inplace, this, t0, t1); }
    
    public Tensor expand(boolean inplace, int... out_dim) { return eg.expand(inplace, this, out_dim); }
    public Tensor expand(boolean inplace, int[] start, int[] out_dim) { return eg.expand(inplace, this, start, out_dim);}
    public Tensor crop(boolean inplace, int... out_dim) { return eg.crop(inplace, this, out_dim); }
    public Tensor crop(boolean inplace, int[] start, int[] out_dim) { return eg.crop(inplace, this, start, out_dim); }
   
    public Tensor expand2D(boolean inplace, int... out_dim) { return eg.expand2D(inplace, this, out_dim); }
    public Tensor expand2D(boolean inplace, int[] start, int[] out_dim) { return eg.expand2D(inplace, this, start, out_dim);}
    public Tensor crop2D(boolean inplace, int... out_dim) { return eg.crop2D(inplace, this, out_dim); }
    public Tensor crop2D(boolean inplace, int[] start, int[] out_dim) { return eg.crop2D(inplace, this, start, out_dim); }
    
    public Tensor normalize_row(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
        return eg.normalize_row(inplace, X, X_mean, X_std, eps);
    }
    public Tensor normalize_field(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
        return eg.normalize_field(inplace, X, X_mean, X_std, eps);
    }
    
    public Tensor gt(boolean inplace, float v) { return eg.gt(inplace, this, v); }
    public Tensor lt(boolean inplace, float v) { return eg.lt(inplace, this, v); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="State: extended">
    @Override public Class<?> type() { return float[].class; }

    private static final int element_each_line = 4096;
    
    @Override
    public ArrayList<String> toStringLines() {
        float[] value = eg.valueOf(this);
        ArrayList<String> lines = new ArrayList<>(4);
        int sb_size = Math.min(element_each_line, length) << 4;//length = 4096 * 16, 16K mem
        StringBuilder sb = new StringBuilder(sb_size);
        
        for(int i=0; i<value.length; i++) {
            sb.append(value[i]).append(',');
            if((i + 1) % element_each_line == 0) {
                String line = sb.deleteCharAt(sb.length() - 1).toString();
                lines.add(line);
                sb.setLength(0);//reset the StringBuilder
            }
        }
        
        if(value.length % element_each_line != 0) {
            String line = sb.deleteCharAt(sb.length() - 1).toString();
            lines.add(line);
        }
        return lines;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Neural: extended">
    //<editor-fold defaultstate="collapsed" desc="mod_count: if the data of Tensor is changed">
    public static class Mod_Count implements Serializable { private int value = 0; }
    protected Mod_Count mod_counter = new Mod_Count();
    
    public int mod_count() { synchronized(mod_counter) { return mod_counter.value; } }
    public Tensor modify(boolean flag) { 
        if(flag) synchronized(mod_counter) { mod_counter.value++; }
        return this; 
    }
    
    public boolean is_hold(int mod_count) {
        synchronized(mod_counter) { 
            return mod_count == mod_counter.value; 
        } 
    }
    
    public Tensor hold(int mod_count, String msg) {
        synchronized(mod_counter) {
            if(mod_count != mod_counter.value) throw new RuntimeException(String.format(
                    "%s : Tensor.mod_count { got %d } != % d, it may cause errors in compute graph",
                    msg, mod_counter.value, mod_count));
        }
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="gradient: find the gradient of a tensor">
    protected boolean is_grad = false;
    public final boolean is_grad() { return is_grad; }
    
    protected boolean need_grad = false;
    public final boolean need_grad() { return need_grad; }
    public final Tensor need_grad(boolean flag) { this.need_grad = flag; return this; }
    
    protected Tensor grad = null;//for variables
    public final Tensor grad() { return grad; }
    public final synchronized Tensor grad(Tensor grad) { this.grad = grad; return this; }
    
    public final synchronized void clear_grad() {
        if (grad == null) return;
        grad.delete(); grad = null;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="carrier: for OneOff Unit">
    //pay attention: the gradients is only used form OneOffScale
    //As sime intermediate variables causes memory leaks and need to be brought out by carriers
    transient protected boolean need_carry = false;//only the output of OneOffScale needCarry
    public final boolean need_carry() { return need_carry; }
    public final Tensor need_carry(boolean flag) { this.need_carry = flag; return this; }
    
    transient protected TensorSet carrier;
    public final synchronized TensorSet carrier() {return carrier;}
    
    public void carry(Tensor ts) {
        if (ts == null || ts == this || !ts.need_carry) return;
        if (carrier == null) carrier = new TensorSet(4);
        synchronized(this) {
            carrier.add(ts);
            if(ts.carrier != null && !ts.carrier.isEmpty()) {//hitch = union(ts.hitch, ts)
                carrier.addAll(ts.carrier);
                ts.carrier = null;//clear ts.carrier
            }
        }
    }
    
   public void carry(Tensor... arr) {
       if(arr == null || arr.length == 0) return;
       for(Tensor ts : arr) carry(ts);
   }
    
    public void clear_carrier() {
        if(carrier == null) return;
        carrier.forEach((fare) -> { eg.delete(fare); });
        carrier.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trace & forward_gc: Used to connect two Units">
    transient protected Trace trace;
    
    public final Trace trace() { return trace; }
    public void setTrace(Trace trace) { this.trace = trace; }
    public void setTrace(UnitCore last, int last_out_index, boolean need_grads) { 
        trace = new Trace(last, last_out_index, need_grads);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ImageEngine<pixel: uint8>: extended">
    public final ImageTool img() { return new ImageTool(this); }
    
    public static class ImageTool
    {
        protected final Tensor ts;
        
        public ImageTool(Tensor ts) { this.ts = ts; }
        
        public byte[] pixel() { return ts.eg.valueOf_int8(ts); }
        
        public Tensor pixel_to_dtype(boolean inplace) { return ts.eg.img.pixel_to_dtype(inplace, ts); }
        public Tensor linear_pixel_to_dtype(boolean inplace, float alpha, float beta) {
            return ts.eg.img.linear_pixel_to_dtype(inplace, alpha, ts, beta);
        }
        
        public Tensor dtype_to_pixel(boolean inplace) { return ts.eg.img.dtype_to_pixel(inplace, ts); }
        public Tensor linear_dtype_to_pixel(boolean inplace, float alpha, float beta) {
            return ts.eg.img.linear_dtype_to_pixel(inplace, alpha, ts, beta);
        }
        
        public Tensor whites_like() { return ts.eg.img.whites_like(ts); }
        public Tensor blacks_like() { return ts.eg.img.blacks_like(ts); }
        public Tensor constants(int value) { return ts.eg.img.constants_like(value, ts); }
        
        public Tensor white() { return ts.eg.img.white(ts); }
        public Tensor black() { return ts.eg.img.black(ts); }
        public Tensor constant(int value) { return ts.eg.img.constant(ts, value); }
        
        public Tensor sadd(boolean inplace, float C) { return ts.eg.img.sadd(inplace, ts, C); }
        public Tensor ssub(boolean inplace, float C) { return ts.eg.img.ssub(inplace, ts, C); }
        public Tensor smul(boolean inplace, float C) { return ts.eg.img.smul(inplace, ts, C); }
        public Tensor sdiv(boolean inplace, float C) { return ts.eg.img.sdiv(inplace, ts, C); }
        public Tensor invert_color(boolean inplace, float C) { return ts.eg.img.invert_color(inplace, ts); }
        public Tensor linear(boolean inplace, float alpha, float beta) { return ts.eg.img.linear(inplace, alpha, ts, beta); }
        
        public Tensor square(boolean inplace) { return ts.eg.img.square(inplace, ts); }
        public Tensor quadraitc(boolean inplace, float alpha, float beta, float gamma) { return ts.eg.img.quadratic(inplace, ts, alpha, beta, gamma); }
        
        public Tensor normalize_row(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
            return ts.eg.img.normalize_row(inplace, X, X_mean, X_std, eps);
        }
        public Tensor normalize_field(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
            return ts.eg.img.normalize_field(inplace, X, X_mean, X_std, eps);
        }
        
        public Tensor pad(boolean inplace, int... p) { return ts.eg.img.pad(inplace, ts, p); }
        public Tensor pad(boolean inplace, int[] p0, int[] p1) { return ts.eg.img.pad(inplace, ts, p0, p1); }
        public Tensor trim(boolean inplace, int... t) { return ts.eg.img.trim(inplace, ts, t); }
        public Tensor trim(boolean inplace, int[] p0, int[] p1) { return ts.eg.trim(inplace, ts, p0, p1); }
        
        public Tensor pad2D(boolean inplace, int... p) { return ts.eg.img.pad2D(inplace, ts, p); }
        public Tensor pad2D(boolean inplace, int[] p0, int[] p1) { return ts.eg.img.pad2D(inplace, ts, p0, p1); }
        public Tensor trim2D(boolean inplace, int... t) { return ts.eg.img.trim2D(inplace, ts, t); }
        public Tensor trim2D(boolean inplace, int[] p0, int[] p1) { return ts.eg.trim2D(inplace, ts, p0, p1); }
        
        public Tensor expand(boolean inplace, int... out_dim) { return ts.eg.img.expand(inplace, ts, out_dim); }
        public Tensor expand(boolean inplace, int[] start, int[] out_dim) { return ts.eg.img.expand(inplace, ts, start, out_dim); }
        public Tensor crop(boolean inplace, int... out_dim) { return ts.eg.img.crop(inplace, ts, out_dim); }
        public Tensor crop(boolean inplace, int[] start, int[] out_dim) { return ts.eg.img.crop(inplace, ts, start, out_dim); }
        
        public Tensor expand2D(boolean inplace, int... out_dim) { return ts.eg.img.expand2D(inplace, ts, out_dim); }
        public Tensor expand2D(boolean inplace, int[] start, int[] out_dim) { return ts.eg.img.expand2D(inplace, ts, start, out_dim); }
        public Tensor crop2D(boolean inplace, int... out_dim) { return ts.eg.img.crop2D(inplace, ts, out_dim); }
        public Tensor crop2D(boolean inplace, int[] start, int[] out_dim) { return ts.eg.img.crop2D(inplace, ts, start, out_dim); }
        
        public Tensor transpose(boolean inplace, int dimIdx1, int dimIdx2) { return ts.eg.img.transpose(inplace, ts, dimIdx1, dimIdx2); }
        
        public Tensor resize(boolean inplace, int out_size) { return ts.eg.img.resize(inplace, ts, out_size); }
        public Tensor resize(boolean inplace, int OH, int OW) { return ts.eg.img.resize(inplace, ts, OH, OW); }
        
        public ImageAffiner affine() { return ts.eg.img.affine(); }
        
        public Tensor translate(boolean inplace, float ty, float tx) { return ts.eg.img.translate(inplace, ts, ty, tx); }
        public Tensor translate(boolean inplace, int OH, int OW, float ty, float tx) { return ts.eg.img.translate(inplace, ts, OH, OW, ty, tx); } 
        
        public Tensor scale(boolean inplace, float sy, float sx) { return ts.eg.img.scale(inplace, ts, sy, sx); }
        public Tensor scale(boolean inplace, int OH, int OW, float sy, float sx) { return ts.eg.img.scale(inplace, ts, OH, OW, sy, sx); }
        
        public Tensor horizontal_flip(boolean inplace) { return ts.eg.img.horizontal_flip(inplace, ts); }
        public Tensor horizontal_flip(boolean inplace, int OH, int OW) { return ts.eg.img.horizontal_flip(inplace, ts, OH, OW); }
        public Tensor vertical_flip(boolean inplace) { return ts.eg.img.vertical_flip(inplace, ts); }
        public Tensor vertical_flip(boolean inplace, int OH, int OW) { return ts.eg.img.vertical_flip(inplace, ts, OH, OW); }
        
        public Tensor shear(boolean inplace, float shy, float shx) { return ts.eg.img.shear(inplace, ts, shy, shx); } 
        public Tensor shear(boolean inplace, int OH, int OW, float shy, float shx) { return ts.eg.img.shear(inplace, ts, OH, OW, shy, shx); }
        
        public Tensor rotate(boolean inplace, float theta) { return ts.eg.img.rotate(inplace, ts, theta); }
        public Tensor rotate(boolean inplace, int OH, int OW, float theta) { return ts.eg.img.rotate(inplace, ts, OH, OW, theta); }
        public Tensor rotate(boolean inplace, float theta, float cy, float cx) { return ts.eg.img.rotate(inplace, ts, theta, cy, cx); }
        public Tensor rotate(boolean inplace, int OH, int OW, float theta, float cy, float cx)  { return ts.eg.img.rotate(inplace, ts, OH, OW, theta, cy, cx); }
    
        public Tensor extract_BGR(boolean inplace) { return ts.eg.img.extract_BGR(inplace, ts); }
        public Tensor extract_RGB(boolean inplace) { return ts.eg.img.extract_RGB(inplace, ts); }
        public Tensor extract_3channels(boolean inplace, int c0, int c1, int c2) { return ts.eg.img.extract_3channels(inplace, ts, c0, c1, c2); }
        
        public Tensor adjust_brightness(boolean inplace, float factor) { return ts.eg.img.adjust_brightness(inplace, ts, factor); }
        public Tensor adjust_saturation(boolean inplace, float factor) {  return ts.eg.img.adjust_saturation(inplace, ts, factor); }
        public Tensor adjust_constrast(boolean inplace, float factor) { return ts.eg.img.adjust_constrast(inplace, ts, factor); }
        public Tensor adjust_color(boolean inplace, float brightness, float saturation, float contrast) {
            return ts.eg.img.adjust_color(inplace, ts, brightness, saturation, contrast);
        } 
        
        public Tensor random_expand(boolean inplace, int... out_dim) { return ts.eg.img.random_expand(inplace, ts, out_dim); }
        public Tensor random_crop(boolean inplace, int... out_dim) { return ts.eg.img.random_crop(inplace, ts, out_dim); }
        
        public Tensor random_expand2D(boolean inplace, int... out_dim) { return ts.eg.img.random_expand2D(inplace, ts, out_dim); }
        public Tensor random_crop2D(boolean inplace, int... out_dim) { return ts.eg.img.random_crop2D(inplace, ts, out_dim); }
        
        public Tensor jit_brightness(float amp, boolean inplace, float factor) { return ts.eg.img.jit_brightness(amp, inplace, ts, factor); }
        public Tensor jit_saturation(float amp, boolean inplace, float factor) {  return ts.eg.img.jit_saturation(amp, inplace, ts, factor); }
        public Tensor jit_contrast(float amp, boolean inplace, float factor) { return ts.eg.img.jit_constrast(amp, inplace, ts, factor); }
        public Tensor jit_color(float amp, boolean inplace, float brightness, float saturation, float contrast) { 
            return ts.eg.img.jit_color(amp, inplace, ts, brightness, saturation, contrast);
        }
        public Tensor jit_color(float[] amp, boolean inplace, float brightness, float saturation, float contrast) {
            return ts.eg.img.jit_color(amp, inplace, ts, brightness, saturation, contrast);
        }
        
        public RandomImageAffiner random_affine() { return ts.eg.img.random_affine(); }
    }
    //</editor-fold>
}
