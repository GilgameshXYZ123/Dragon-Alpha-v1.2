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
public abstract class Pool2D extends SimpleFunction {
    private static final long serialVersionUID = 1L;
    
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int OH, OW;
    
    protected Pool2D(
            int kernel_height,  int kernel_width, 
            int stride_height,  int stride_width,
            int padding_height, int padding_width,
            int output_height,  int output_width)
    {
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.OH = output_height;  this.OW = output_width;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">  
    public final int[] kernel() { return new int[]{ FH, FW }; }
    public Pool2D kernel(int... kernel) {
        if(kernel == null || kernel.length != 2) throw new IllegalArgumentException(
                "kernel == null || kernel.length != 2");
        FH = kernel[0]; FW = kernel[1];
        return this;
    } 
    
    public final int[] stride() { return new int[]{ sh, sw }; } 
    public Pool2D stride(int...stride) {
        if(stride == null || stride.length != 2) throw new IllegalArgumentException(
                "stride == null || stride.length != 2");
        sh = stride[0]; sw = stride[1];
        return this;
    }
    
    public final int[] padding() { return new int[]{ ph, pw }; }
    public Pool2D padding(int... padding) {
        if(padding == null ||  padding.length != 2) throw new IllegalArgumentException(
                "padding == null || padding.length != 1");
        ph = padding[0]; pw = padding[1];
        return this;
    }
    
    public final int[] out_size() { return new int[] { OH, OW }; }
    public Pool2D out_size(int... out_size) {
        if(out_size == null || out_size.length != 2) throw new IllegalArgumentException(
                "out_size == null || out_size.length != 2");
        OH = out_size[0]; OW = out_size[1];
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { kernel = [").append(FH).append(", ").append(FW).append("], ");
        sb.append(", stride = [").append(sh).append(", ").append(sw).append("], ");
        sb.append(", padding = [").append(ph).append(", ").append(pw).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlinePool2D"> 
    public static abstract class InlinePool2D<T extends Pool2D> extends SimpleCore<T> {
        public InlinePool2D(T unit) { super(unit); }

        public final int[] kernel()   { return new int[]{ ut.FH, ut.FW }; }
        public final int[] stride()   { return new int[]{ ut.sh, ut.sw }; } 
        public final int[] padding()  { return new int[]{ ut.ph, ut.pw }; }
        public final int[] out_size() { return new int[]{ ut.OH, ut.OW }; }
    }
    //</editor-fold>
}
