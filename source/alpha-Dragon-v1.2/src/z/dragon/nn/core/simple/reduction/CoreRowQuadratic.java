/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.reduction;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreRowQuadratic<T extends SimpleUnit> extends SimpleCore<T> {
    protected int row_length;
    transient protected int field_length;
    
    protected float alpha;
    protected float beta;
    protected float gamma;

    public CoreRowQuadratic(T unit, int row_length, float alpha, float beta, float gamma) {
        super(unit);
        this.row_length = row_length;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    public float row_length() { return row_length; }
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    public float gamma() { return gamma; }
    
     //<editor-fold defaultstate="collapsed" desc="running-area: propagation">  
    protected Tensor __forward__(Engine eg, Tensor X) {
        Tensor Y = eg.row_quadratic(X, row_length, alpha, beta, gamma);
        field_length = X.length() / Y.length();
        return Y;
    }

    protected Tensor __backward__(Engine eg, Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads ? 
                eg.repeat_quadratic(grad_inplace, deltaY, field_length, alpha, beta, gamma) :
                null);
    }
    //</editor-fold>
}
