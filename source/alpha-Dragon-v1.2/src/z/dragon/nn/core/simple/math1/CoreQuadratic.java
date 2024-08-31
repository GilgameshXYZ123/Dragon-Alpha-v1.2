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
public class CoreQuadratic<T extends SimpleUnit> extends SimpleCore<T> {
    protected float alpha;
    protected float beta;
    protected float gamma;
    
    public CoreQuadratic(T unit, float alpha, float beta, float gamma) {
        super(unit);
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    public float gamma() { return gamma; }
    
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(getClass().getSimpleName());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        return eg.quadratic(false, X, alpha, beta, gamma);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.quadratic_deltaX(grad_inplace, deltaY, holdX(), alpha, beta):
                null);
    }
    //</editor-fold>
}
