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
 * bernouli(leakyRelu(X) > p, v1, v2).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LeakyRelu_BernouliMul extends BernouliMul {
    private static final long serialVersionUID = 562781240330661L;
    
    protected float k;
    
    public LeakyRelu_BernouliMul(boolean inplace, 
            float negative_slope,
            float p, float v1, float v2) 
    {
        super(inplace, p, v1, v2);
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float negative_slop() { return k; }
    public LeakyRelu_BernouliMul negative_slop(float negative_slope) { k = negative_slope; return this; }
    
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
    
    @Override
    protected InlineLeakyRelu_BernouliMul create_unit_core() { 
        return new InlineLeakyRelu_BernouliMul(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLeakyRelu_BernouliMul">
    public static class InlineLeakyRelu_BernouliMul extends InlineBernouliMul<LeakyRelu_BernouliMul> {
        public InlineLeakyRelu_BernouliMul(LeakyRelu_BernouliMul unit) { super(unit); }
    
        public final float negative_slop() { return ut.k; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) {
                X = eg.leakyRelu(inplace, X, ut.k);
                float exp = expect();//exp = (v1*p + v2*(1.0f - p))
                return (exp == 1.0f ? X : eg.linear(true, exp, X.c(), 0.0f));
            }
        
            Tensor[] outs = eg.leakyRelu_bernouli_mul(X, ut.k, ut.p, ut.v1, ut.v2);
            R = outs[1];
            return outs[0];
        }
        //</editor-fold>
    }
    //</editor-fold>
}
