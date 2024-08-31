/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class AvgPool1D extends Pool1D {
    private static final long serialVersionUID = 1L;
    
    protected boolean ignore_padding;

    public AvgPool1D(boolean ignore_padding,
            int kernel_height,
            int stride_height,
            int padding_height,
            int output_height)
    {
        super(kernel_height,
              stride_height,
              padding_height,
              output_height);
        this.ignore_padding = ignore_padding;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public boolean ignore_padding() { return ignore_padding; }
    public AvgPool1D ignore_padding(boolean flag) { this.ignore_padding = flag; return this;}
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public AvgPool1D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { ignore_padding = ").append(ignore_padding);
        sb.append(", kernel = [").append(FW).append("]");
        sb.append(", stride = [").append(sw).append("]");
        sb.append(", padding = [").append(pw).append("]");
        sb.append(", out_size = [").append(OW).append("] }");
    }
    
    @Override
    protected InlineAvgPool1D<?> create_unit_core() {
        return new InlineAvgPool1D<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAvgPool1D">
    public static class InlineAvgPool1D<T extends AvgPool1D> extends InlinePool1D<T> {
        transient protected int IW;
        transient protected Tensor y;

        public InlineAvgPool1D(T unit) { super(unit); }
        
        public boolean ignore_padding() { return ut.ignore_padding; }
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) {
            IW = X.dim(-2);//[N, W, C]
            y = (ut.pre_alloc_forward ? 
                    eg.alloc.pool1D(X, ut.FW, ut.OW, ut.sw, ut.pw) : 
                    null);
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return eg.pool1D_avg(ut.ignore_padding, out, X, ut.FW, ut.sw, ut.pw);
            }
            return eg.pool1D_avg(ut.ignore_padding, X, ut.FW, ut.OW, ut.sw, ut.pw);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            
            Tensor deltaX = eg.unpool1D_avg(ut.ignore_padding, deltaY, ut.FW, IW, ut.sw, ut.pw);
            return (grad_inplace?
                    deltaX.dual(()-> { deltaY.delete(); }) : 
                    deltaX);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
