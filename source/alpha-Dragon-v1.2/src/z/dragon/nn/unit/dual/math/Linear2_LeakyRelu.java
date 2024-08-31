/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualFunction;

/**
 *
 * @author Gilgamesh
 */
public class Linear2_LeakyRelu extends DualFunction  {
    private static final long serialVersionUID = 1L;
    
    protected boolean likeX1;
    protected float alpha, beta, gamma;
    protected float k;
    
    public Linear2_LeakyRelu(boolean likeX1, 
            float alpha, float beta, float gamma,
            float negative_slope)
    {
        this.likeX1 = likeX1;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final boolean likeX1() { return likeX1; }
    public Linear2_LeakyRelu likeX1(boolean flag) { likeX1 = flag; return this; }
    
    public final float alpha() { return alpha; }
    public Linear2_LeakyRelu alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public Linear2_LeakyRelu beta(float beta) { this.beta = beta; return this; }
    
    public final float gamma() { return gamma; }
    public Linear2_LeakyRelu gamma(float gamma) { this.gamma = gamma; return this; }
    
    public final float negative_slope() { return k; }
    public Linear2_LeakyRelu negative_slope(float negative_slope) { this.k = negative_slope; return this; }
    
    @Override
    protected InlineLinear2_LeakyRelu create_unit_core() {
        return new InlineLinear2_LeakyRelu(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinear2_LeakyRelu">
    public static class InlineLinear2_LeakyRelu extends DualCore<Linear2_LeakyRelu>
    {
        public InlineLinear2_LeakyRelu(Linear2_LeakyRelu unit) { super(unit); }

        public final boolean likeX1() { return ut.likeX1; }
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
        public final float gamma() { return ut.gamma; }
        public final float negative_slope() { return ut.k; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            return eg.linear2_leakyRelu(false, ut.likeX1, X1, X2, 
                    ut.alpha, ut.beta, ut.gamma, ut.k);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads, 
                boolean backward_grads1, boolean backward_grads2) {
            if(!backward_grads) return null;
            
            Tensor[] grads = (is_holdY()? //V1: holdY / V2: hold{X1, X2}
                    eg.linear2_leakyRelu_deltaX_v1(grad_inplace, deltaY, holdY(), ut.alpha, ut.beta, ut.k) :
                    eg.linear2_leakyRelu_deltaX_v2(grad_inplace, deltaY, holdX1(), holdX2(), ut.alpha, ut.beta, ut.gamma, ut.k));
            
            if(!backward_grads1) { grads[0].remote_delete(); grads[0] = null; }//deltaX1
            if(!backward_grads2) { grads[1].remote_delete(); grads[1] = null; }//deltaX2
            return grads;//{deltaX1, deltaX2}
        }
        //</editor-fold>
    }
    //</editor-fold>
}
