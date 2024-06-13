/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

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
public class AdaptiveMaxPool2D extends AdaptivePool2D implements Train2Eval
{
    private static final long serialVersionUID = 1L;
    
    public AdaptiveMaxPool2D(int out_height, int out_width) { 
        super(out_height, out_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public AdaptiveMaxPool2D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { training = ").append(training);
        sb.append(", out_size = [").append(OH).append(", ").append(OW).append("] }");
    }
     
    @Override
    protected InlineAdaptiveMaxPool2D<?> create_unit_core() {
        return new InlineAdaptiveMaxPool2D<>(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; } 
    @Override public AdaptiveMaxPool2D train() { this.training = true; return this; }
    @Override public AdaptiveMaxPool2D eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptiveMaxPool2D">
    public static class InlineAdaptiveMaxPool2D<T extends AdaptiveMaxPool2D> extends InlineAdaptivePool2D<T>
    {
        transient protected Tensor Index;//Tensor<int32>
        transient protected Tensor y;
        
        public InlineAdaptiveMaxPool2D(T unit) { super(unit); }
        
        public final boolean training() { return ut.training; } 
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
            __adaptive__(X); if(!ut.training) return;            
            
            if(ut.pre_alloc_forward) {
                Tensor[] outs = eg.alloc.pool2D_indexed(X, FH, FW, ut.OH, ut.OW, sh, sw, 0, 0);
                y = outs[0]; Index = outs[1];
            }
            else { y = null; Index = null; }
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(!ut.training) return eg.pool2D_max(X, FH, FW, ut.OH, ut.OW,  sh, sw, 0, 0);
            
            if(ut.pre_alloc_forward) {//training: compute Index for: X -> Y
                Tensor out = y.c(); y = null;
                return eg.pool2D_max_indexed(out, Index.c(), X, FH, FW, sh, sw, 0, 0);
            }

            Tensor[] result = eg.pool2D_max_indexed(X, FH, FW, ut.OH, ut.OW, sh, sw, 0, 0);
            Index = result[1];//training: compute Index for: X -> Y
            return result[0];//deltaY = result[0]
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.unpool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, 
                    sh, sw, 0, 0);
            return (grad_inplace? 
                    deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                    deltaX.dual(()-> { Index.delete(); }));
        }
        //</editor-fold>
    }
    //</editor-fold>
}
