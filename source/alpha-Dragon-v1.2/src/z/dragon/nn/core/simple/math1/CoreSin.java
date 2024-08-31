/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.math1;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;
import z.dragon.nn.core.simple.SimpleCore;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreSin<T extends SimpleUnit> extends SimpleCore<T> {
    protected float alpha;
    protected float beta;

    public CoreSin(T unit, float alpha, float beta) {
        super(unit);
        this.alpha = alpha;
        this.beta = beta;
    }
   
    public final float alpha() { return alpha; }
    public final float beta() { return beta; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        return eg.sin(false, alpha, X, beta);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.sin_deltaX(grad_inplace, deltaY, holdX(), alpha, beta):
                null);
    }
    //</editor-fold>
}
