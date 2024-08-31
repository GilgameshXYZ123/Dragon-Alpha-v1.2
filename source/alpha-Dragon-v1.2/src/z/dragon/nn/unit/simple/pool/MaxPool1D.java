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
public class MaxPool1D extends Pool1D implements Train2Eval {
    private static final long serialVersionUID = 1L;
    
    public MaxPool1D(
            int kernel_height, 
            int step_height,
            int padding_height,
            int output_height) 
    {
        super(kernel_height,
              step_height,
              padding_height,
              output_height);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public MaxPool1D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { training = ").append(training);
        sb.append(", kernel = [").append(FW).append("], ");
        sb.append(", stride = [").append(sw).append("], ");
        sb.append(", padding = [").append(pw).append("] }");
    }
    
    @Override
    protected InlineMaxPool1D<?> create_unit_core() {
        return new InlineMaxPool1D<>(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; } 
    @Override public MaxPool1D train() { this.training = true; return this; }
    @Override public MaxPool1D eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineMaxPool1D">
    public static class InlineMaxPool1D<T extends MaxPool1D> extends InlinePool1D<T> {
        transient protected int IW;
        transient protected Tensor y;
        transient protected Tensor Index;//Tensor<int32>
        
        public InlineMaxPool1D(T unit) { super(unit); }
        
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
            
            IW = X.dim(-2);//[N, W, C]
            if(ut.pre_alloc_forward) {
                Tensor[] outs = eg.alloc.pool1D_indexed(X, ut.FW, ut.OW, ut.sw, ut.pw);
                y = outs[0]; Index = outs[1];
            }
            else { y = null; Index = null; }
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(!ut.training) return eg.pool1D_max(X, ut.FW, ut.OW, ut.sw, ut.pw);
            
            if(ut.pre_alloc_forward) {//training: compute Index for: X -> Y
               Tensor out = y.c(); y = null;
               return eg.pool1D_max_indexed(out, Index.c(), X, ut.FW, ut.sw, ut.pw);
            }
            
            Tensor[] result = eg.pool1D_max_indexed(X, ut.FW, ut.OW, ut.sw, ut.pw);
            Index = result[1];//training: compute Index for: X -> Y
            return result[0];//deltaY = result[0]
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.unpool1D_max_indexed(deltaY, Index, IW, ut.FW, ut.sw, ut.pw);
            return (grad_inplace?
                    deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                    deltaX.dual(()-> { Index.delete(); }));
        }
        //</editor-fold>
    }
    //</editor-fold>
}
