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
public class CoreCsc<T extends SimpleUnit> extends SimpleCore<T>
{
    protected float alpha;
    protected float beta;

    public CoreCsc(T unit, float alpha, float beta) {
        super(unit);
        this.alpha = alpha;
        this.beta = beta;
    }
 
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(getClass().getSimpleName());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        return eg.csc(false, alpha, X, beta);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.csc_deltaX(grad_inplace, deltaY, holdX(), alpha, beta):
                null);
    }
    //</editor-fold>
}
