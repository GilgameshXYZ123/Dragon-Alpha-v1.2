/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math1;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.dragon.nn.core.simple.SimpleCore;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Sec extends SimpleFunction
{
    private static final long serialVersionUID = 562781240250001L;
    
    protected float alpha;
    protected float beta;

    public Sec(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float alpha() { return alpha; }
    public Sec alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public Sec beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineSec create_unit_core() {
        return new InlineSec(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineSec">
    public static class InlineSec extends SimpleCore<Sec>
    {
        public InlineSec(Sec unit) { super(unit); }

        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area">  
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            return eg.sec(false, ut.alpha, X, ut.beta);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY,
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.sec_deltaX(grad_inplace, deltaY, holdX(), ut.alpha, ut.beta):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
