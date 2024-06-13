/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LeakyRelu_BernouliMul extends SimpleInplaceFunction
        implements Train2Eval
{
    private static final long serialVersionUID = 562781240330001L;
    
    protected float k;
    protected float p;//possibility of positive case
    protected float v1;//value of positive case
    protected float v2;//value of negative case
    
    public LeakyRelu_BernouliMul(boolean inplace, 
            float negative_slope,
            float p, float v1, float v2) 
    {
        super(inplace);
        this.k = negative_slope;
        this.p = p;
        this.v1 = v1;
        this.v2 = v2;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float negative_slop() { return k; }
    public LeakyRelu_BernouliMul negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    public final float expect() { return v1*p + v2*(1.0f - p); } 
    
    public final float p() { return p; }
    public LeakyRelu_BernouliMul p(float p) { this.p = p; return this; }
    
    public final float v1() { return v1; }
    public LeakyRelu_BernouliMul v1(float v1) { this.v1 = v1; return this; }
    
    public final float v2() { return v2; }
    public LeakyRelu_BernouliMul v2(float v2) { this.v2 = v2; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", expect = ").append(expect());
        sb.append(", negative_slope = ").append(k);
        sb.append(", p = ").append(p);
        sb.append(", [v1, v2] = [").append(v1).append(", ").append(v2).append(" ]");
        sb.append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineLeakyRelu_BernouliMul create_unit_core() { 
        return new InlineLeakyRelu_BernouliMul(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; }
    @Override public LeakyRelu_BernouliMul train() { this.training = true; return this; }
    @Override public LeakyRelu_BernouliMul eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineBernouliMul">
    public static class InlineLeakyRelu_BernouliMul extends SimpleInplaceInline<LeakyRelu_BernouliMul>
    {
        transient protected Tensor R;

        public InlineLeakyRelu_BernouliMul(LeakyRelu_BernouliMul unit) { super(unit); }
    
        public final float negative_slop() { return ut.k; }
        public final float expect() { return ut.v1*ut.p + ut.v2*(1.0f - ut.p); } 
        public final float p() { return ut.p; }
        public final float v1() { return ut.v1; }
        public final float v2() { return ut.v2; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: others">
        @Override
        public void gc() {
            super.gc(); 
            if(R != null) { R.delete(); R = null; }
        }
    
        @Override
        public void variables(Tensor.TensorSet set) {
            super.variables(set);
            set.add(R);
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) {
                X = eg.leakyRelu(inplace, X, ut.k);
                float exp = expect();//exp = (v1*p + v2*(1.0f - p))
                return (exp == 1.0f ? X : eg.linear(true, expect(), X.c(), 0.0f));
            }
        
            Tensor[] outs = eg.leakyRelu_bernouli_mul(X, ut.k, ut.p, ut.v1, ut.v2);
            R = outs[1];
            return outs[0];
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY,
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads ? //when deltaX is cauculated, R is not needed
                    eg.mul(grad_inplace, deltaY, R).dual(()->{ R.delete(); }):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
