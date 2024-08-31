/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool.adaptive;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class AdaptiveAvgPool1D extends AdaptivePool1D {
    private static final long serialVersionUID = 1L;
    
    protected boolean ignore_padding;
    
    public AdaptiveAvgPool1D(boolean ignore_padding, int out_width) {
        super(out_width);
        this.ignore_padding = ignore_padding;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public boolean ignore_padding() { return ignore_padding; }
    public AdaptiveAvgPool1D ignore_padding(boolean flag) { this.ignore_padding = flag; return this;}
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public AdaptiveAvgPool1D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { ignore_padding = ").append(ignore_padding);
        sb.append(", out_size = [").append(OW).append("] }");
    }
    
    @Override
    protected InlineAdaptiveAvgPool1D<?> create_unit_core() {
        return new InlineAdaptiveAvgPool1D<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptiveAvgPool1D">
    public static class InlineAdaptiveAvgPool1D
            <T extends AdaptiveAvgPool1D> extends InlineAdaptivePool1D<T> {
        transient protected Tensor y;
        
        public InlineAdaptiveAvgPool1D(T unit) { super(unit); }
        
        public final boolean ignore_padding() { return ut.ignore_padding; }
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">

        @Override
        protected void __before_forward__(Engine eg, Tensor X) {
            __adaptive__(X);
            y = (ut.pre_alloc_forward ? 
                    eg.alloc.pool1D(X, FW, ut.OW, sw, 0) : 
                    null);
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return eg.pool1D_avg(ut.ignore_padding, out, X, FW, sw, 0);
            }
            return eg.pool1D_avg(ut.ignore_padding, X, FW, ut.OW, sw, 0);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            
            Tensor deltaX = eg.unpool1D_avg(ut.ignore_padding, deltaY, FW, IW, sw, 0);
            return (grad_inplace?
                    deltaX.dual(()-> { deltaY.delete(); }):
                    deltaX);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
