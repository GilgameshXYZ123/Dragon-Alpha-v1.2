/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.dual.math;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualUnit;

/**
 * Linear2.
 * (1) Y = alpha*X1 + beta*X2 + gamma,
 * (2) deltaX1 = deltaY * alpha
 * (3) deltaX2 = deltaY * beta
 * @author Gilgamesh
 * @param <T>
 */
public class CoreLinear2<T extends DualUnit> extends DualCore<T> 
{
    protected boolean likeX1;
    protected float alpha, beta, gamma;
    
    public CoreLinear2(T unit, boolean likeX1, 
            float alpha, float beta, float gamma) 
    {
        super(unit);
        this.likeX1 = likeX1;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    public boolean likeX1() { return likeX1; }
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    public float gamma() { return gamma; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        return eg.linear2(false, likeX1, X1, X2, alpha, beta, gamma);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads, 
            boolean backward_grads1, boolean backward_grads2) 
    {
        if(!backward_grads) return null;
        if(backward_grads1 && backward_grads2) return eg.linear_2out(grad_inplace, deltaY, alpha, 0, beta, 0);
        return new Tensor[] { 
            (backward_grads1 ? eg.linear(grad_inplace, alpha, deltaY, 0) : null),  //(1) deltaX1 = deltaY * alpha
            (backward_grads2 ? eg.linear(grad_inplace, beta,  deltaY, 0) : null) };//(2) deltaX2 = deltaY * beta
    }
    //</editor-fold>
}
