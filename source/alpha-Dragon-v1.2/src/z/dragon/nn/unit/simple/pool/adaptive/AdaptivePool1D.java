/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool.adaptive;

import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public abstract class AdaptivePool1D extends SimpleFunction {
    private static final long serialVersionUID = 1L;
    
    protected int OW;
    
    protected AdaptivePool1D(int out_width) { out_size(out_width); }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public int[] out_size() { return new int[]{ OW }; }
    public AdaptivePool1D out_size(int... out_size) {
        if(out_size == null || out_size.length != 1) throw new IllegalArgumentException(
                "out_size == null || out_size.length != 1");
        if(out_size[0] <= 0) throw new IllegalArgumentException(String.format(
                "out_height { got %d } must > 0", out_size[0]));
        this.OW = out_size[0];
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ out_size = [").append(OW).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptivePool1D">
    public static abstract class InlineAdaptivePool1D
            <T extends AdaptivePool1D> extends SimpleCore<T> {
        transient protected int FW;
        transient protected int sw;
        transient protected int IW;
        
        public InlineAdaptivePool1D(T unit) { super(unit); }
        
        public final int[] out_size() {return new int[]{ ut.OW };}
        
        protected final void __adaptive__(Tensor X) {//X[N, IW, IC]
            IW = X.dim(-2);
            sw = Math.floorDiv(IW, ut.OW); 
            FW = IW - (ut.OW - 1)*sw;
        }
    }
    //</editor-fold>
}
