/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Tensor;
import z.dragon.engine.Engine;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Rot180 extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1L;
    
    public Rot180(boolean inplace) { super(inplace); }

    @Override
    protected InlineRot180 create_unit_core() {
        return new InlineRot180(this);
    }

    //<editor-fold defaultstate="collapsed" desc="static class: InlineRot180">
    public static class InlineRot180 extends SimpleInplaceInline<Rot180> 
    {
        public InlineRot180(Rot180 unit) { super(unit); }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.rot180(inplace, X);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads? 
                    eg.rot180(grad_inplace, deltaY) : 
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
