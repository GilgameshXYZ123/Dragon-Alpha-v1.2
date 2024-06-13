/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public abstract class AdaptivePool2D extends SimpleFunction
{
    private static final long serialVersionUID = 1L;
    
    protected int OH, OW;
    
    protected AdaptivePool2D(int out_height, int out_width) {
        out_size(out_height, out_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public int[] out_size() { return new int[]{ OH, OW }; }
    public AdaptivePool2D out_size(int... out_size) {
        if(out_size == null || out_size.length != 2) throw new IllegalArgumentException(
                "out_size == null || out_size.length != 2");
        if(out_size[0] <= 0) throw new IllegalArgumentException(String.format(
                "out_height { got %d } must > 0", out_size[0]));
        if(out_size[1]  <= 0) throw new IllegalArgumentException(String.format(
                "out_width { got %d } must > 0", out_size[1]));
        this.OH = out_size[0];
        this.OW = out_size[1];
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ out_size = [").append(OH).append(", ").append(OW).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptivePool2D">
    public static abstract class InlineAdaptivePool2D<T extends AdaptivePool2D> extends SimpleCore<T>
    {
        transient protected int FH, FW;
        transient protected int sh, sw;
        transient protected int IH, IW;
        
        public InlineAdaptivePool2D(T unit) { super(unit); }
        
        public final int[] out_size() {return new int[]{ ut.OH, ut.OW };}
        
        protected final void __adaptive__(Tensor X) {//X[N, IH, IW, IC]
            IH = X.dim(-3); sh = Math.floorDiv(IH, ut.OH); FH = IH - (ut.OH - 1)*sh;
            IW = X.dim(-2); sw = Math.floorDiv(IW, ut.OW); FW = IW - (ut.OW - 1)*sw;
        }
    }
    //</editor-fold>
}
