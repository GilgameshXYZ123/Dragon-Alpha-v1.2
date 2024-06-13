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
public class Linear2_Relu extends DualFunction 
{
    protected boolean likeX1;
    protected float alpha, beta, gamma;
    
    public Linear2_Relu(boolean likeX1, 
            float alpha, float beta, float gamma)
    {
        this.likeX1 = likeX1;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final boolean likeX1() { return likeX1; }
    public Linear2_Relu likeX1(boolean flag) { likeX1 = flag; return this; }
    
    public final float alpha() { return alpha; }
    public Linear2_Relu alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public Linear2_Relu beta(float beta) { this.beta = beta; return this; }
    
    public final float gamma() { return gamma; }
    public Linear2_Relu gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    protected InlineLinear2_Relu create_unit_core() {
        return new InlineLinear2_Relu(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinear2_LeakyRelu">
    public static class InlineLinear2_Relu extends DualCore<Linear2_Relu>
    {
        public InlineLinear2_Relu(Linear2_Relu unit) { super(unit); }

        public final boolean likeX1() { return ut.likeX1; }
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
        public final float gamma() { return ut.gamma; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            return eg.linear2_relu(false, ut.likeX1, X1, X2, 
                    ut.alpha, ut.beta, ut.gamma);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads, 
                boolean backward_grads1, boolean backward_grads2) {
            if(!backward_grads) return null;
            
            Tensor[] grads = (is_holdY()? //V1: holdY / V2: hold{X1, X2}
                    eg.linear2_relu_deltaX_v1(grad_inplace, deltaY, holdY(), ut.alpha, ut.beta) :
                    eg.linear2_relu_deltaX_v2(grad_inplace, deltaY, holdX1(), holdX2(), ut.alpha, ut.beta, ut.gamma));
            
            if(!backward_grads1) { grads[0].remote_delete(); grads[0] = null; }//deltaX1
            if(!backward_grads2) { grads[1].remote_delete(); grads[1] = null; }//deltaX2
            return grads;//{deltaX1, deltaX2}
        }
        //</editor-fold>
    }
    //</editor-fold>
}
