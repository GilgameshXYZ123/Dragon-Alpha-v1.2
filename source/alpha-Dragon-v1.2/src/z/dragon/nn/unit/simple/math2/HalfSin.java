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
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class HalfSin extends SimpleInplaceFunction {
    private static final long serialVersionUID = 562781240380001L;
    
    protected float amp;
    protected float alpha;
    protected float beta;

    public HalfSin(boolean inplace, float amp, float alpha, float beta) {
        super(inplace);
        this.amp = amp;
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float amp() { return alpha; }
    public HalfSin amp(float ampltitude) { this.amp = ampltitude; return this; }
    
    public final float alpha() { return alpha; }
    public HalfSin setAlpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public HalfSin beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", amp = ").append(amp);
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineHalfSin create_unit_core() {
        return new InlineHalfSin(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineHalfSin">
    public static class InlineHalfSin extends SimpleInplaceInline<HalfSin>
    {
        public InlineHalfSin(HalfSin unit) { super(unit); }
        
        public final float amp() { return ut.amp; }
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.halfSin(inplace, ut.amp, ut.alpha, X, ut.beta);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads? 
                    eg.halfSin_deltaX(grad_inplace, deltaY, holdY(), ut.amp, ut.alpha) :
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
