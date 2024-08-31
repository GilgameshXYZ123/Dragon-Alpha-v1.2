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
public class Elu extends SimpleInplaceFunction {   
    private static final long serialVersionUID = 562781240360201L;
    
    protected float alpha;
    protected float k;
    
    public Elu(boolean inplace, float alpha, float negative_slope) {
        super(inplace);
        this.alpha = alpha;
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public float alpha() { return alpha; }
    public Elu alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float negative_slope() { return k; }
    public Elu negative_slope(float negative_slope) { k = negative_slope; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", alpha = ").append(alpha);
        sb.append(", negative_slope = ").append(k);
        sb.append(" }");
    }
    
    @Override
    protected InlineElu create_unit_core() {
        return new InlineElu(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineElu">
    public static class InlineElu extends SimpleInplaceInline<Elu>
    {
        public InlineElu(Elu unit) { super(unit); }
     
        public final float alpha() { return ut.alpha; }
        public final float negative_slope() { return ut.k; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.elu(inplace, X, ut.alpha, ut.k);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY()?
                    eg.elu_deltaX_v1(grad_inplace, deltaY, holdY(), ut.alpha, ut.k) ://V1: Y is not changed
                    eg.elu_deltaX_v2(grad_inplace, deltaY, holdX(), ut.alpha, ut.k));//V2: X is not changed
        }
        //</editor-fold>
    }
    //</editor-fold>
}
