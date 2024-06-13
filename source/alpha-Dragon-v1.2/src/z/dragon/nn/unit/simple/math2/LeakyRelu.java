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
public class LeakyRelu extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 562781240390001L;
    
    protected float k;
    
    public LeakyRelu(boolean inplace, float negative_slope) {
        super(inplace);
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public float negative_slop() { return k; }
    public LeakyRelu negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", negative_slope = ").append(k).append(" }");
    }
    
    @Override
    protected InlineLeakyRelu create_unit_core() {
        return new InlineLeakyRelu(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineLeakyRelu">
    public static class InlineLeakyRelu extends SimpleInplaceInline<LeakyRelu> 
    {
        public InlineLeakyRelu(LeakyRelu unit) { super(unit); }
        
        public float negative_slop() { return ut.k; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.leakyRelu(inplace, X, ut.k);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY()? 
                    eg.leakyRelu_deltaX_v1(grad_inplace, deltaY, holdY(), ut.k) ://V1: Y is not changed
                    eg.leakyRelu_deltaX_v2(grad_inplace, deltaY, holdX(), ut.k));//V2: X is not changed
        }
        //</editor-fold>
    }
    //</editor-fold>
}
