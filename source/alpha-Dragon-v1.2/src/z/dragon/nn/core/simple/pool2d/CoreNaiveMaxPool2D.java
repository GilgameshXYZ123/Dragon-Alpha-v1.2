/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool2d;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreNaiveMaxPool2D<T extends SimpleUnit> extends CorePool2D<T> 
{
    public CoreNaiveMaxPool2D(T unit, 
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width, 
            int padding_height, int padding_width, 
            int output_height, int output_width) 
    {
        super(unit, 
              kernel_height, kernel_width, 
              stride_height, stride_width, 
              padding_height, padding_width, 
              output_height, output_width);
    }

    //<editor-fold defaultstate="collapsed" desc="static class: InlineNaiveMaxPool2D">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
         return eg.pool2D_max(X, FH, FW, OH, OW, sh, sw, ph, pw);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
            
        Tensor deltaX = eg.unpool2D_max(deltaY, holdY(), holdX(), FH, FW, sh, sw, ph, pw);
        return (grad_inplace? 
                deltaX.dual(()-> { deltaY.delete(); }) :
                deltaX);
    }
    //</editor-fold>
}
