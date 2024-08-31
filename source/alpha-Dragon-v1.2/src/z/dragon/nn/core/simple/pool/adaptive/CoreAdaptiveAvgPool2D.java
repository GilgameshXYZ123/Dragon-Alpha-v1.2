/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool.adaptive;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreAdaptiveAvgPool2D<T extends SimpleUnit> extends CoreAdaptivePool2D<T> {
    protected boolean ignore_padding;

    public CoreAdaptiveAvgPool2D(T unit, boolean ignore_padding,
            int out_height, int out_width) 
    {
        super(unit, out_height, out_width);
        this.ignore_padding = ignore_padding;
    }
    
    public boolean ignore_padding() { return ignore_padding; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        __adaptive__(X);
        return eg.pool2D_avg(ignore_padding, 
                X, FH, FW, OH, OW, 
                sh, sw, 0, 0);
    }
        
    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;

        Tensor deltaX = eg.unpool2D_avg(ignore_padding, 
                deltaY, FH, FW, IH, IW,
                sh, sw, 0, 0);
        return (grad_inplace?
                deltaX.dual(()-> { deltaY.delete(); }):
                deltaX);
    }
    //</editor-fold>
}
