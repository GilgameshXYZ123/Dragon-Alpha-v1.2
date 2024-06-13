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
public class CoreLinear<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    protected float alpha;
    protected float beta;
    
    public CoreLinear(T unit, boolean inplace, float alpha, float beta) {
        super(unit, inplace); 
        this.alpha = alpha;
        this.beta = beta;
    }
    
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.linear(inplace, alpha, X, beta);
    }

    @Override //Y = alpha*X + beta, deltaX = alpha * deltaY
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.linear(grad_inplace, alpha, deltaY, 0):
                null);
    }
    //</editor-fold>
}
