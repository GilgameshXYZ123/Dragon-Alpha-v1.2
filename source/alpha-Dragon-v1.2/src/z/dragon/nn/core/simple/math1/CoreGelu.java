/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.math1;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreGelu <T extends SimpleUnit> extends SimpleCore<T>
{
    public CoreGelu(T unit) { super(unit); }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        return eg.gelu(false, X);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.gelu_deltaX(grad_inplace, deltaY, holdX()).c() :
                null);
    }
    //</editor-fold>
}
