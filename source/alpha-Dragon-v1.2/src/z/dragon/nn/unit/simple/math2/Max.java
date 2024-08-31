/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Max extends SimpleInplaceFunction {
    private static final long serialVersionUID = 562781240360786L;
    
    protected float alpha;
    protected float beta;
    protected float vmax;
    
    public Max(boolean inplace, float alpha, float beta, float vmax) {
        super(inplace);
        this.alpha = alpha;
        this.beta = beta;
        this.vmax = vmax;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public float alpha() { return alpha; }
    public Max alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Max beta(float beta) { this.beta = beta; return this; }
    
    public float vmax() { return vmax; }
    public Max vmax(float vmax) { this.vmax = vmax; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", vmax = ").append(vmax);
        sb.append(" }");
    }
    
    @Override
    protected SimpleCore<?> create_unit_core() {
        return new InlineMax(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineMax">
    public static class InlineMax extends SimpleInplaceInline<Max> 
    {
        public InlineMax(Max unit) { super(unit); }

        public float alpha() { return ut.alpha; }
        public float beta() { return ut.beta; }
        public float vmax() { return ut.vmax; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.max(inplace, ut.alpha, X, ut.beta, ut.vmax);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;  
            return (is_holdY()?
                    eg.max_deltaX_v1(grad_inplace, deltaY, holdY(), ut.alpha, ut.vmax) ://V1: Y is not changed
                    eg.max_deltaX_v2(grad_inplace, deltaY, holdX(), ut.alpha, ut.beta, ut.vmax));//V2: X is not changed
        }
        //</editor-fold>
    }
    //</editor-fold>
}
