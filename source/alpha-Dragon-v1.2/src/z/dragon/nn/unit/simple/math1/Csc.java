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
public class Csc extends SimpleFunction
{
    private static final long serialVersionUID = 562781240230001L;
    
    protected float alpha;
    protected float beta;

    public Csc(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Csc alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Csc beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineCsc create_unit_core() {
        return new InlineCsc(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineCsc">
    public static class InlineCsc extends SimpleCore<Csc>
    {
        public InlineCsc(Csc unit) { super(unit); }
        
        public float alpha() { return ut.alpha; }
        public float beta() { return ut.beta; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area">  
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            return eg.csc(false, ut.alpha, X, ut.beta);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads? 
                    eg.csc_deltaX(grad_inplace, deltaY, holdX(), ut.alpha, ut.beta):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
