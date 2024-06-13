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
public class Arcsin extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 562781240310001L;
    
    protected float alpha;
    protected float beta;
    
    public Arcsin(boolean inplace, float alpha, float beta) {
        super(inplace);
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float alpha() { return alpha; }
    public Arcsin alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public Arcsin beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineArcsin create_unit_core() {
        return new InlineArcsin(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static_class: InlineArcsin">
    public static class InlineArcsin extends SimpleInplaceInline<Arcsin>  
    {
        public InlineArcsin(Arcsin unit) { super(unit); }
        
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.arcsin(inplace, ut.alpha, X, ut.beta);
        }
       
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
               boolean grad_inplace, boolean backward_grads) {
            return (backward_grads? 
                    eg.arcsin_deltaX(grad_inplace, deltaY, holdY(), ut.alpha):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
