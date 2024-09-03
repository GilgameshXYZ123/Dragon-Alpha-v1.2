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
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreLinear2Center<T extends DualUnit> extends DualCore<T> {
    protected int dim2;
    protected float alpha, beta, gamma;

    public CoreLinear2Center(T unit, int dim2,
            float alpha, float beta, float gamma) 
    {
        super(unit);
        this.dim2 = dim2;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    public int dim2() { return dim2; }
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    public float gamma() { return gamma; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        return eg.linear2_center(false, X1, X2, dim2, alpha, beta, gamma);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads, 
            boolean backward_grads1, boolean backward_grads2) 
    {
        if(!backward_grads) return null;
        
        Tensor X1 = holdX1(), X2 = holdX2();
        if(backward_grads1 && backward_grads2)//(1) deltaX1 = deltaY * X1 * alpha
            return eg.linear2_center_deltaX(grad_inplace, deltaY, X1, X2, dim2, alpha, beta, gamma);
                    
        
        return new Tensor[] {//(2) deltaX2 = center_sum: deltaY * X2 * beta
            (backward_grads1 ? eg.linear2_center_deltaX1(grad_inplace, deltaY, X1, X2, dim2, alpha) : null),
            (backward_grads2 ? eg.linear2_center_deltaX2(deltaY, X1, X2, dim2, beta) : null)
        };
    }
    //</editor-fold>
}
