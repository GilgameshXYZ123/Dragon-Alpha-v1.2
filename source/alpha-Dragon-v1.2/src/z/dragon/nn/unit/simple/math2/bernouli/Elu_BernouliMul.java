/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2.bernouli;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 * bernouli(Elu(X) > p, v1, v2).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Elu_BernouliMul extends BernouliMul {
    private static final long serialVersionUID = 562781240330661L;
    
    protected float alpha, k;
    
    public Elu_BernouliMul(boolean inplace, 
            float alpha, float negative_slope,
            float p, float v1, float v2) 
    {
        super(inplace, p, v1, v2);
        this.alpha = alpha;
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float alpha() { return alpha; }
    public Elu_BernouliMul alpha(float alpha) { this.alpha = alpha ; return this; }
    
    public final float negative_slop() { return k; }
    public Elu_BernouliMul negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", expect = ").append(expect());
        sb.append(", alpha = ").append(alpha);
        sb.append(", negative_slope = ").append(k);
        sb.append(", p = ").append(p);
        sb.append(", [v1, v2] = [").append(v1).append(", ").append(v2).append(" ]");
        sb.append(" }");
    }
    
    @Override
    protected InlineElu_BernouliMul create_unit_core() { 
        return new InlineElu_BernouliMul(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineElu_BernouliMul">
    public static class InlineElu_BernouliMul extends InlineBernouliMul<Elu_BernouliMul> {
        public InlineElu_BernouliMul(Elu_BernouliMul unit) { super(unit); }
    
        public final float alpha() { return ut.alpha; }
        public final float negative_slop() { return ut.k; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) {
                X = eg.elu(inplace, X, ut.alpha, ut.k);
                float exp = expect();//exp = (v1*p + v2*(1.0f - p))
                return (exp == 1.0f ? X : eg.linear(true, exp, X.c(), 0.0f));
            }
        
            Tensor[] outs = eg.elu_bernouli_mul(X, ut.alpha, ut.k, ut.p, ut.v1, ut.v2);
            R = outs[1];
            return outs[0];
        }
        //</editor-fold>
    }
    //</editor-fold>
}
