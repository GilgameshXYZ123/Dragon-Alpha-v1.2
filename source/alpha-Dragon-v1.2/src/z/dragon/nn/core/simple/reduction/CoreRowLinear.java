/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.reduction;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;
import z.dragon.nn.core.simple.SimpleCore;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreRowLinear<T extends SimpleUnit> extends SimpleCore<T> {
    protected int row_length;
    transient protected int field_length;
    
    protected float alpha;
    protected float beta;
    
    public CoreRowLinear(T unit, int row_length, float alpha, float beta) {
        super(unit);
        this.row_length = row_length;
        this.alpha = alpha;
        this.beta = beta;
    }
    
    public float row_length() { return row_length; }
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">  
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        Tensor Y = eg.row_linear(X, row_length, alpha, beta);
        field_length = X.length() / Y.length();
        return Y;
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads ? 
                eg.repeat_linear(grad_inplace, deltaY, field_length, alpha, beta) :
                null);
    }
    //</editor-fold>
}
