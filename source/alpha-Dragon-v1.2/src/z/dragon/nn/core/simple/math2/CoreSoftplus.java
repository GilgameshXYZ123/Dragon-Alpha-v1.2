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
public class CoreSoftplus<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    public CoreSoftplus(T unit, boolean inplace) {
        super(unit, inplace); 
    }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.softplus(inplace, X);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (is_holdY()?
                eg.softplus_deltaX_v1(grad_inplace, deltaY, holdY()) ://V1: Y is not changed
                eg.softplus_deltaX_v2(grad_inplace, deltaY, holdX()));//V2: X is not changed
    }
    //</editor-fold>
}
