/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class AdaptiveNaiveMaxPool2D extends AdaptivePool2D
{
    private static final long serialVersionUID = 1L;
    
    public AdaptiveNaiveMaxPool2D (int out_height, int out_width) {
         super(out_height, out_width);
    }

    @Override
    protected InlineAdaptiveNaiveMaxPool2D<?> create_unit_core() {
        return new InlineAdaptiveNaiveMaxPool2D<>(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAdaptiveAvgPool2D">
    public static class InlineAdaptiveNaiveMaxPool2D<T extends AdaptiveNaiveMaxPool2D> 
            extends InlineAdaptivePool2D<T>
    {
        public InlineAdaptiveNaiveMaxPool2D(T unit) { super(unit); }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            __adaptive__(X);
            return eg.pool2D_max(X, FH, FW, ut.OH, ut.OW, sh, sw, 0, 0);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.unpool2D_max(deltaY, holdY(), holdX(), FH, FW, sh, sw, 0, 0);
            return (grad_inplace? 
                    deltaX.dual(()-> { deltaY.delete(); }) :
                    deltaX);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
