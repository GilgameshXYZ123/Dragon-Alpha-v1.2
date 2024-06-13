/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public abstract class Unpool2D extends SimpleFunction
{
    private static final long serialVersionUID = 1L;
    
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int OH, OW;
    
    public Unpool2D(
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width, 
            int padding_height, int padding_width,
            int output_height, int output_width) 
    {
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.OH = output_height;  this.OW = output_width;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final int[] kernel() { return new int[]{ FH, FW }; }
    public Unpool2D kernel(int... kernel) {
        if(kernel == null || kernel.length != 2) throw new IllegalArgumentException();
        FH = kernel[0]; FW = kernel[1];
        return this;
    }
    
    public final int[] stride() { return new int[]{ sh, sw }; }
    public Unpool2D stride(int...stride) {
        if(stride == null || stride.length != 2) throw new IllegalArgumentException();
        sh = stride[0]; sw = stride[1];
        return this;
    }
    
    public final int[] padding() { return new int[]{ ph, pw };}
    public Unpool2D padding(int... padding) {
        if(padding == null ||  padding.length != 2) throw new IllegalArgumentException();
        ph = padding[0]; pw = padding[1];
        return this;
    }
    
    public final int[] out_size() { return new int[] { OH, OW }; }
    public final Unpool2D out_size(int... out_size) {
        if(out_size == null || out_size.length != 2) throw new IllegalArgumentException();
        OH = out_size[0]; OW = out_size[1];
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append(" kernel = [").append(FH).append(", ").append(FW).append(']');
        sb.append(", stride = [").append(sh).append(", ").append(sw).append(']');
        sb.append(", padding = [").append(ph).append(", ").append(pw).append(']');
        sb.append(", output_size = [").append(OH).append(", ").append(OW).append("] }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineUnpool2D"> 
    public static abstract class InlineUnpool2D<T extends Unpool2D> extends SimpleCore<T>
    {
        public InlineUnpool2D(T unit) { super(unit); }
        
        public final int[] kernel() { return new int[]{ ut.FH, ut.FW }; }
        public final int[] stride() { return new int[]{ ut.sh, ut.sw }; }
        public final int[] padding() { return new int[]{ ut.ph, ut.pw };}
        public final int[] out_size() { return new int[] { ut.OH, ut.OW }; }
    }
    //</editor-fold>
}
