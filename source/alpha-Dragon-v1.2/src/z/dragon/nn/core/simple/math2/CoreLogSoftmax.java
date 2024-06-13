/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleInplaceCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreLogSoftmax<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    protected int features;
    
    public CoreLogSoftmax(T unit, boolean inplace, int features) {
        super(unit, inplace);
        this.features = features;
    }
    
    public final int features() { return features; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.log_softmax(inplace, X, features);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.log_softmax_deltaX(grad_inplace, deltaY, holdY(), features):
                null);
    }
    //</editor-fold>
}
