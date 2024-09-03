/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math1;

import z.dragon.engine.Engine;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Cos extends SimpleFunction {
    private static final long serialVersionUID =  562781240220001L;
    
    protected float alpha;
    protected float beta;

    public Cos(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public float alpha() { return alpha; }
    public Cos alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Cos beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    
    @Override
    protected InlineCos create_unit_core() {
        return new InlineCos(this);
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="static class: InlineCos">
    public static class InlineCos extends SimpleCore<Cos>
    {
        public InlineCos(Cos unit) { super(unit); }
        
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
    
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
