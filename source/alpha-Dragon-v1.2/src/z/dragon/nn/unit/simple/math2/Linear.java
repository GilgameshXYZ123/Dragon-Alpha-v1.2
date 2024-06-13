/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 * Y = alpha * X + beta.
 * alpha and beta are constants
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Linear extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 562781240400001L;
    
    protected float alpha;
    protected float beta;
    
    public Linear(boolean inplace, float alpha, float beta) {
        super(inplace);
        this.alpha = alpha;
        this.beta = beta;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float alpha() { return alpha; }
    public Linear alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public Linear beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineLinear create_unit_core() {
        return new InlineLinear(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinear">
    public static class InlineLinear extends SimpleInplaceInline<Linear>
    {
        public InlineLinear(Linear unit) { super(unit); }

        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.linear(inplace, ut.alpha, X, ut.beta);
        }

        @Override //Y = alpha*X + beta, deltaX = alpha * deltaY
        protected Tensor __backward__(Engine eg, Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.linear(grad_inplace, ut.alpha, deltaY, 0):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
