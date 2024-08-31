/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LeakyRelu_Dropout extends SimpleInplaceFunction implements Train2Eval {
    private static final long serialVersionUID = 562781240350001L;
    
    protected float k;
    protected float p;//posibility of 1 instead of 0
    
    public LeakyRelu_Dropout(boolean inplace,
            float negative_slope, 
            float nonzero_p) 
    {
        super(inplace);
        nonzero_percent(nonzero_p);
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float negative_slop() { return k; }
    public LeakyRelu_Dropout negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    public final float nonzero_percent() { return p; }
    public final LeakyRelu_Dropout nonzero_percent(float nonzero_p) {
        if(nonzero_p == 0) throw new IllegalArgumentException("nonzero_p can't be zero");
        this.p = nonzero_p;
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", negative_slope = ").append(k);
        sb.append(", nonzero_percent = ").append(p).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineLeakyRelu_Dropout create_unit_core() {
        return new InlineLeakyRelu_Dropout(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; }
    @Override public LeakyRelu_Dropout train() { this.training = true; return this; }
    @Override public LeakyRelu_Dropout eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLeakyRelu_Dropout">
    public static class InlineLeakyRelu_Dropout extends SimpleInplaceInline<LeakyRelu_Dropout>
    {
        transient protected Tensor R;

        public InlineLeakyRelu_Dropout(LeakyRelu_Dropout unit) { super(unit); }
        
        public final float negative_slop() { return ut.k; }
        public final float nonzero_percent() { return ut.p; }
        public final boolean training() { return ut.training; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: others">
        @Override
        public void gc() {
            super.gc(); 
            if(R != null) { R.delete(); R = null; }
        }
    
        @Override
        public void variables(TensorSet set) {
            super.variables(set);
            set.add(R);
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) return eg.leakyRelu(inplace, X, ut.k);//exp = (1/p)*p + 0*(1-p) = 1

            Tensor[] outs = eg.leakyRelu_dropout(X, ut.k, ut.p);
            R = outs[1];
            return outs[0];
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads ? //when deltaX is cauculated, rber is not needed
                    eg.mul(grad_inplace, deltaY, R).dual(()->{ R.delete(); }):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
