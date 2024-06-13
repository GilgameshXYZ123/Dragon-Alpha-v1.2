/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class AdaptiveAvgPool2D extends AdaptivePool2D
{
    private static final long serialVersionUID = 1L;
    
    protected boolean ignore_padding;
    
    public AdaptiveAvgPool2D(boolean ignore_padding, int out_height, int out_width) {
        super(out_height, out_width);
        this.ignore_padding = ignore_padding;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public boolean ignore_padding() { return ignore_padding; }
    public AdaptiveAvgPool2D ignore_padding(boolean flag) { this.ignore_padding = flag; return this;}
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public AdaptiveAvgPool2D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { ignore_padding = ").append(ignore_padding);
        sb.append(", out_size = [").append(OH).append(", ").append(OW).append("] }");
    }
    
    @Override
    protected InlineAdaptiveAvgPool2D<?> create_unit_core() {
        return new InlineAdaptiveAvgPool2D<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptiveAvgPool2D">
    public static class InlineAdaptiveAvgPool2D<T extends AdaptiveAvgPool2D> extends InlineAdaptivePool2D<T>
    {
        transient protected Tensor y;
        
        public InlineAdaptiveAvgPool2D(T unit) { super(unit); }
        
        public final boolean ignore_padding() { return ut.ignore_padding; }
        
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">

        @Override
        protected void __before_forward__(Engine eg, Tensor X) {
            __adaptive__(X);
            y = (ut.pre_alloc_forward ? 
                    eg.alloc.pool2D(X, FH, FW, ut.OH, ut.OW, sh, sw, 0, 0) : 
                    null);
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return eg.pool2D_avg(ut.ignore_padding, out, X, FH, FW, 
                        sh, sw, 0, 0);
            }
            
            return eg.pool2D_avg(ut.ignore_padding, X, FH, FW, ut.OH, ut.OW, 
                    sh, sw, 0, 0);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            
            Tensor deltaX = eg.unpool2D_avg(ut.ignore_padding, 
                    deltaY, FH, FW, IH, IW,
                    sh, sw, 0, 0);
            return (grad_inplace?
                    deltaX.dual(()-> { deltaY.delete(); }):
                    deltaX);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
