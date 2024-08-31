/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Train2Eval;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class MaxPool2D extends Pool2D implements Train2Eval {
    private static final long serialVersionUID = 1L;
    
    public MaxPool2D(
            int kernel_height,  int kernel_width,
            int stride_height,  int stride_width,
            int padding_height, int padding_width,
            int output_height,  int output_width) 
    {
        super(kernel_height,  kernel_width, 
              stride_height,  stride_width,
              padding_height, padding_width,
              output_height,  output_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public MaxPool2D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { training = ").append(training);
        sb.append(", kernel = [").append(FH).append(", ").append(FW).append("], ");
        sb.append(", stride = [").append(sh).append(", ").append(sw).append("], ");
        sb.append(", padding = [").append(ph).append(", ").append(pw).append("] }");
    }
    
    @Override
    protected InlineMaxPool2D<?> create_unit_core() {
        return new InlineMaxPool2D<>(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; } 
    @Override public MaxPool2D train() { this.training = true; return this; }
    @Override public MaxPool2D eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineMaxPool2D">
    public static class InlineMaxPool2D<T extends MaxPool2D> extends InlinePool2D<T> {
        transient protected int IH, IW;
        transient protected Tensor y;
        transient protected Tensor Index;//Tensor<int32>
        
        public InlineMaxPool2D(T unit) { super(unit); }
        
        public boolean training() { return ut.training; }
        public Tensor Index() { return Index; }
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: others">
        @Override
        public void variables(TensorSet set) {
            super.variables(set);
            set.add(Index);
        }

        @Override
        public void gc() { 
            super.gc(); 
            if(Index != null) { Index.delete(); Index = null; }
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) { 
            if(!ut.training) return;
            
            IH = X.dim(-3); IW = X.dim(-2);//[N, H, W, C]
            if(ut.pre_alloc_forward) {
                Tensor[] outs = eg.alloc.pool2D_indexed(X, ut.FH, ut.FW, ut.OH, ut.OW, 
                        ut.sh, ut.sw, ut.ph, ut.pw);
                y = outs[0]; Index = outs[1];
            }
            else { y = null; Index = null; }
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(!ut.training) return eg.pool2D_max(X, ut.FH, ut.FW, ut.OH, ut.OW, 
                    ut.sh, ut.sw, ut.ph, ut.pw);
            
            if(ut.pre_alloc_forward) {//training: compute Index for: X -> Y
               Tensor out = y.c(); y = null;
               return eg.pool2D_max_indexed(out, Index.c(), X, ut.FH, ut.FW, 
                       ut.sh, ut.sw, ut.ph, ut.pw);
            }
            
            Tensor[] result = eg.pool2D_max_indexed(X, ut.FH, ut.FW, ut.OH, ut.OW, 
                    ut.sh, ut.sw, ut.ph, ut.pw);//training: compute Index for: X -> Y
            Index = result[1];
            return result[0];//deltaY = result[0]
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.unpool2D_max_indexed(deltaY, Index, IH, IW, ut.FH, ut.FW, 
                    ut.sh, ut.sw, ut.ph, ut.pw);
            return (grad_inplace?
                    deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                    deltaX.dual(()-> { Index.delete(); }));
        }
        //</editor-fold>
    }
    //</editor-fold>
}
