/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool.adaptive;

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
public class AdaptiveMaxPool1D extends AdaptivePool1D implements Train2Eval {
    private static final long serialVersionUID = 1L;
    
    public AdaptiveMaxPool1D(int out_width) { 
        super(out_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public AdaptiveMaxPool1D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { training = ").append(training);
        sb.append(", out_size = [").append(", ").append(OW).append("] }");
    }
     
    @Override
    protected InlineAdaptiveMaxPool1D<?> create_unit_core() {
        return new InlineAdaptiveMaxPool1D<>(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; } 
    @Override public AdaptiveMaxPool1D train() { this.training = true; return this; }
    @Override public AdaptiveMaxPool1D eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptiveMaxPool1D">
    public static class InlineAdaptiveMaxPool1D
            <T extends AdaptiveMaxPool1D> extends InlineAdaptivePool1D<T> {
        transient protected Tensor Index;//Tensor<int32>
        transient protected Tensor y;
        
        public InlineAdaptiveMaxPool1D(T unit) { super(unit); }
        
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
                Tensor[] outs = eg.alloc.pool1D_indexed(X, FW, ut.OW, sw, 0);
                y = outs[0]; Index = outs[1];
            }
            else { y = null; Index = null; }
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(!ut.training) return eg.pool1D_max(X, FW, ut.OW, sw, 0);
            
            if(ut.pre_alloc_forward) {//training: compute Index for: X -> Y
                Tensor out = y.c(); y = null;
                return eg.pool1D_max_indexed(out, Index.c(), X, FW, sw, 0);
            }

            Tensor[] result = eg.pool1D_max_indexed(X, FW, ut.OW, sw, 0);
            Index = result[1];//training: compute Index for: X -> Y
            return result[0];//deltaY = result[0]
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.unpool1D_max_indexed(deltaY, Index, IW, FW, sw, 0);
            return (grad_inplace? 
                    deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                    deltaX.dual(()-> { Index.delete(); }));
        }
        //</editor-fold>
    }
    //</editor-fold>
}
