/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Rpl extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1L;
    
    protected float alpha;
    protected float beta;
    protected float gamma;
    
    public Rpl(boolean inplace, float alpha, float beta, float gamma) {
        super(inplace);
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float alpha() { return alpha; }
    public Rpl alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public Rpl beta(float beta) { this.beta = beta; return this; }
    
    public final float gamma() { return gamma; }
    public Rpl gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", beta = ").append(beta);
        sb.append(", alpha = ").append(alpha);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    
    @Override
    protected InlineRpl create_unit_core() {
        return new InlineRpl(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineRpl">  
    public static class InlineRpl extends SimpleInplaceInline<Rpl> 
    {
        public InlineRpl(Rpl unit) { super(unit); }
        
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
        public final float gamma() { return ut.gamma; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">  
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.rpl(inplace, ut.alpha, X, ut.beta, ut.gamma);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.rpl_deltaX(grad_inplace, deltaY, holdY(), ut.alpha, ut.gamma):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
