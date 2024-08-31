/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class NaiveMaxPool2D extends Pool2D {
    private static final long serialVersionUID = 1L;
    
    public NaiveMaxPool2D(
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width,
            int padding_height, int padding_width,
            int output_height, int output_width)
    {
        super(kernel_height, kernel_width,
              stride_height, stride_width,
              padding_height, padding_width,
              output_height, output_width);
    }
    
    @Override
    protected InlineNaiveMaxPool2D<?> create_unit_core() {
        return new InlineNaiveMaxPool2D<>(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineNaiveMaxPool2D">
    public static class InlineNaiveMaxPool2D
            <T extends NaiveMaxPool2D> extends InlinePool2D<T> {
        public InlineNaiveMaxPool2D(T unit) { super(unit); }

        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            return eg.pool2D_max(X, ut.FH, ut.FW, ut.OH, ut.OW, 
                    ut.sh, ut.sw, ut.ph, ut.pw);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            
            Tensor deltaX = eg.unpool2D_max(deltaY, holdY(), holdX(), 
                    ut.FH, ut.FW, ut.sh, ut.sw, ut.ph, ut.pw);
            return (grad_inplace? 
                    deltaX.dual(()-> { deltaY.delete(); }) :
                    deltaX);
        }
    }
    //</editor-fold>
}
