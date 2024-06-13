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
public class CoreAvgUnpool2D<T extends SimpleUnit> extends CoreUnpool2D<T>
{
    protected boolean ignore_padding;
    
    transient protected int IH, IW;
    
    public CoreAvgUnpool2D(T unit, boolean ignore_padding,
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
        this.ignore_padding = ignore_padding;
    }
    
    public boolean ignore_padding() { return ignore_padding; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        int[] dim = X.dim(); IH = dim[1]; IW = dim[2];
        return eg.unpool2D_avg(ignore_padding, 
                X, FH, FW, OH, OW, 
                sh, sw, ph, pw);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.pool2D_avg(ignore_padding, 
                deltaY, FH, FW, IH, IW, 
                sh, sw, ph, pw);
        return (grad_inplace? 
                deltaX.dual(()-> { deltaY.delete(); }) : 
                deltaX);
    }
    //</editor-fold>
}
