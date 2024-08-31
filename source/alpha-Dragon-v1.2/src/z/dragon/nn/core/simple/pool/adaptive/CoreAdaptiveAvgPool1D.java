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
public class CoreAdaptiveAvgPool1D<T extends SimpleUnit> extends CoreAdaptivePool1D<T> {
    protected boolean ignore_padding;

    public CoreAdaptiveAvgPool1D(T unit, boolean ignore_padding, int out_width) {
        super(unit, out_width);
        this.ignore_padding = ignore_padding;
    }
    
    public boolean ignore_padding() { return ignore_padding; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        __adaptive__(X);
        return eg.pool1D_avg(ignore_padding, X, FW, OW, sw, 0);
    }
        
    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;

        Tensor deltaX = eg.unpool1D_avg(ignore_padding, deltaY, FW, IW, sw, 0);
        return (grad_inplace?
                deltaX.dual(()-> { deltaY.delete(); }):
                deltaX);
    }
    //</editor-fold>
}
