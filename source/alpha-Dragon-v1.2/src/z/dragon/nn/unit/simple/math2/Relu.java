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
public class Relu extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1231412412491L;
    
    public Relu(boolean inplace) { super(inplace); }

    @Override
    protected InlineRelu create_unit_core() {
        return new InlineRelu(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineRelu">
    public static class InlineRelu extends SimpleInplaceInline<Relu>
    {
        public InlineRelu(Relu unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) { 
            return eg.relu(inplace, X);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY,
                boolean grad_inplace, boolean backward_grads) { 
            if(!backward_grads) return null;
            return (is_holdY()?
                    eg.relu_deltaX_v1(grad_inplace, deltaY, holdY()) ://V1: Y is not changed
                    eg.relu_deltaX_v2(grad_inplace, deltaY, holdX()));//V2: X is not changed
        }
        //</editor-fold>
    }
    //</editor-fold>
}
