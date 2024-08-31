/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreAvgPool1D<T extends SimpleUnit> extends CorePool1D<T> {
    protected boolean ignore_padding;
    transient protected int IH, IW;

    public CoreAvgPool1D(T unit, boolean ignore_padding,
            int kernel_width, 
            int stride_width, 
            int padding_width, 
            int output_width) 
    {
        super(unit,
              kernel_width, 
              stride_width, 
              padding_width,
              output_width);
        this.ignore_padding = ignore_padding;
    }
    
    public final boolean ignore_padding() { return ignore_padding; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        IW = X.dim(-2);//[N,W, C]
        return eg.pool1D_avg(ignore_padding, X, FW, OW, sw, pw);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY,
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
            
        Tensor deltaX = eg.unpool1D_avg(ignore_padding, deltaY, FW, IW, sw, pw);
        return (grad_inplace?
                deltaX.dual(()-> { deltaY.delete(); }) : 
                deltaX);
    }
    //</editor-fold>
}
