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
public class AvgPool2D extends Pool2D {
    private static final long serialVersionUID = 1L;
    
    protected boolean ignore_padding;

    public AvgPool2D(boolean ignore_padding,
            int kernel_height,  int kernel_width, 
            int stride_height,  int stride_width, 
            int padding_height, int padding_width,
            int output_height,  int output_width) 
    {
        super(kernel_height,  kernel_width,
              stride_height,  stride_width,
              padding_height, padding_width,
              output_height,  output_width);
        this.ignore_padding = ignore_padding;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public boolean ignore_padding() { return ignore_padding; }
    public AvgPool2D ignore_padding(boolean flag) { this.ignore_padding = flag; return this;}
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public AvgPool2D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { ignore_padding = ").append(ignore_padding);
        sb.append(", kernel = [").append(FH).append(", ").append(FW).append("]");
        sb.append(", stride = [").append(sh).append(", ").append(sw).append("]");
        sb.append(", padding = [").append(ph).append(", ").append(pw).append("]");
        sb.append(", out_size = [").append(OH).append(", ").append(OW).append("] }");
    }
    
    @Override
    protected InlineAvgPool2D<?> create_unit_core() {
        return new InlineAvgPool2D<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAvgPool2D">
    public static class InlineAvgPool2D<T extends AvgPool2D> extends InlinePool2D<T> {
        transient protected int IH, IW;
        transient protected Tensor y;

        public InlineAvgPool2D(T unit) { super(unit); }
        
        public boolean ignore_padding() { return ut.ignore_padding; }
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) {
            IH = X.dim(-3); IW = X.dim(-2);//[N, H, W, C]
            y = (ut.pre_alloc_forward ? 
                    eg.alloc.pool2D(X, ut.FH, ut.FW, ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw) : 
                    null);
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return eg.pool2D_avg(ut.ignore_padding, out, X, ut.FH, ut.FW, 
                        ut.sh, ut.sw, ut.ph, ut.pw);
            }
            
            return eg.pool2D_avg(ut.ignore_padding, X, ut.FH, ut.FW, ut.OH, ut.OW,
                    ut.sh, ut.sw, ut.ph, ut.pw);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            
            Tensor deltaX = eg.unpool2D_avg(ut.ignore_padding, 
                    deltaY, ut.FH, ut.FW, IH, IW, 
                    ut.sh, ut.sw, ut.ph, ut.pw);
            return (grad_inplace?
                    deltaX.dual(()-> { deltaY.delete(); }) : 
                    deltaX);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
