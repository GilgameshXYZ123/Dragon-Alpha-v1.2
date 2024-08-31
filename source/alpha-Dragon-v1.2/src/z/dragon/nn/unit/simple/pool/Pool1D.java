/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool;

import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public abstract class Pool1D extends SimpleFunction {
    private static final long serialVersionUID = 1L;
    
    protected int FW;
    protected int sw;
    protected int pw;
    protected int OW;
    
    protected Pool1D(
            int kernel_width, 
            int stride_width,
            int padding_width,
            int output_width)
    {
        this.FW = kernel_width;
        this.sw = stride_width;
        this.pw = padding_width;
        this.OW = output_width;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">  
    public final int[] kernel() { return new int[]{ FW }; }
    public Pool1D kernel(int... kernel) {
        if(kernel == null || kernel.length != 1) throw new IllegalArgumentException(
                "kernel == null || kernel.length != 1");
        FW = kernel[1];
        return this;
    } 
    
    public final int[] stride() { return new int[]{ sw }; } 
    public Pool1D stride(int...stride) {
        if(stride == null || stride.length != 1) throw new IllegalArgumentException(
                "stride == null || stride.length != 1");
        sw = stride[0];
        return this;
    }
    
    public final int[] padding() { return new int[]{ pw }; }
    public Pool1D padding(int... padding) {
        if(padding == null ||  padding.length != 1) throw new IllegalArgumentException(
                "padding == null || padding.length != 1");
        pw = padding[0];
        return this;
    }
    
    public final int[] out_size() { return new int[] { OW }; }
    public Pool1D out_size(int... out_size) {
        if(out_size == null || out_size.length != 1) throw new IllegalArgumentException(
                "out_size == null || out_size.length != 1");
        OW = out_size[0];
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { kernel = [").append(FW).append("], ");
        sb.append(", stride = [").append(sw).append("], ");
        sb.append(", padding = [").append(pw).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlinePool1D"> 
    public static abstract class InlinePool1D<T extends Pool1D> extends SimpleCore<T> {
        public InlinePool1D(T unit) { super(unit); }

        public final int[] kernel()   { return new int[]{ ut.FW }; }
        public final int[] stride()   { return new int[]{ ut.sw }; } 
        public final int[] padding()  { return new int[]{ ut.pw }; }
        public final int[] out_size() { return new int[]{ ut.OW }; }
    }
    //</editor-fold>
}
