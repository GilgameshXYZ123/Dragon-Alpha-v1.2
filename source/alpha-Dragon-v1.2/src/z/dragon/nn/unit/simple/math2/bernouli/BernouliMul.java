/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2.bernouli;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 * bernouli(X > p, v1, v2).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class BernouliMul extends SimpleInplaceFunction implements Train2Eval {
    private static final long serialVersionUID = 562781240330001L;
    
    protected float p;//possibility of positive case
    protected float v1;//value of positive case
    protected float v2;//value of negative case
    
    public BernouliMul(boolean inplace, float p, float v1, float v2) {
        super(inplace);
        this.p = p;
        this.v1 = v1;
        this.v2 = v2;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float expect() { return v1*p + v2*(1.0f - p); } 
    
    public final float p() { return p; }
    public BernouliMul p(float p) { this.p = p; return this; }
    
    public final float v1() { return v1; }
    public BernouliMul v1(float v1) { this.v1 = v1; return this; }
    
    public final float v2() { return v2; }
    public BernouliMul v2(float v2) { this.v2 = v2; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", expect = ").append(expect());
        sb.append(", p = ").append(p);
        sb.append(", [v1, v2] = [").append(v1).append(", ").append(v2).append(" ]");
        sb.append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineBernouliMul create_unit_core() { 
        return new InlineBernouliMul(this);
    }
    
    protected boolean training = true;
    @Override public final boolean training() { return training; }
    @Override public BernouliMul train() { this.training = true; return this; }
    @Override public BernouliMul eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineBernouliMul">
    public static class InlineBernouliMul<T extends BernouliMul> extends SimpleInplaceInline<T>{
        transient protected Tensor R;

        public InlineBernouliMul(T unit) { super(unit); }
    
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
            if(!ut.training) return eg.linear(inplace, expect(), X, 0);//exp = (v1*p + v2*(1.0f - p))
        
            Tensor[] outs = eg.bernouli_mul(X, ut.p, ut.v1, ut.v2);
            R = outs[1];
            return outs[0];//Y = outs[0]
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
