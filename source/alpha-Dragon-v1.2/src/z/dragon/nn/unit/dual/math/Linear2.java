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
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Linear2 extends DualFunction
{
    private static final long serialVersionUID = 1L;
    
    protected boolean likeX1;
    protected float alpha, beta, gamma;
      
    public Linear2(boolean likeX1, float alpha, float beta, float gamma) {
        this.likeX1 = likeX1;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public boolean likeX1() { return likeX1; }
    public Linear2 likeX1(boolean flag) { likeX1 = flag; return this; }
    
    public float alpha() { return alpha; }
    public Linear2 alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Linear2 beta(float beta) { this.beta = beta; return this; }
    
    public float gamma() { return gamma; }
    public Linear2 gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { likeX1 = ").append(likeX1);
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    
    @Override
    protected InlineLinear2 create_unit_core() {
        return new InlineLinear2(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinear2">
    public static class InlineLinear2 extends DualCore<Linear2>
    {
        public InlineLinear2(Linear2 unit) { super(unit); }

        public final boolean likeX1() { return ut.likeX1; }
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
        public final float gamma() { return ut.gamma; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
            return eg.linear2(false, ut.likeX1, X1, X2, ut.alpha, ut.beta, ut.gamma);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads, 
                boolean backward_grads1, boolean backward_grads2) 
        {
            if(!backward_grads) return null;

            //(1) deltaX1 = deltaY * alpha
            //(2) deltaX2 = deltaY * beta
            if(backward_grads1 && backward_grads2) 
                return eg.linear_2out(grad_inplace, deltaY, ut.alpha, 0, ut.beta, 0);
            return new Tensor[] { 
                (backward_grads1? eg.linear(grad_inplace, ut.alpha, deltaY, 0) : null), 
                (backward_grads2? eg.linear(grad_inplace, ut.beta,  deltaY, 0) : null) };
        }
        //</editor-fold>
    }
    //</editor-fold>
}
