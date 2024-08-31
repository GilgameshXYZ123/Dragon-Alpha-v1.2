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

/**
 *
 * @author Gilgamesh
 */
public class HardSigmoid extends SimpleInplaceFunction  {
    private static final long serialVersionUID = 1231412776191L;
    
    public HardSigmoid(boolean inplace) { super(inplace); }
    
    @Override
    protected SimpleCore<?> create_unit_core() {
        return new InlineHardSigmoid(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineHardSigmoid">
    public static class InlineHardSigmoid extends SimpleInplaceInline<HardSigmoid> 
    {
        public InlineHardSigmoid(HardSigmoid unit) { super(unit); }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.hard_sigmoid(inplace, X);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY()? 
                    eg.hard_sigmoid_deltaX_v1(grad_inplace, deltaY, holdY()) ://V1: Y is not changed
                    eg.hard_sigmoid_deltaX_v2(grad_inplace, deltaY, holdX()));//V2: X is not changed
        }
        //</editor-fold>
    }
    //</editor-fold>
}
