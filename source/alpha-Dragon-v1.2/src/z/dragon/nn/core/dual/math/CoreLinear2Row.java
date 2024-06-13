/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.dual.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.dual.DualUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreLinear2Row<T extends DualUnit> extends DualCore<T> 
{
    transient protected int row_length;
    protected float alpha, beta, gamma;

    public CoreLinear2Row(T unit,
            float alpha, float beta, float gamma) 
    {
        super(unit);
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    public float alpha() { return alpha; }
    public float beta() { return beta; }
    public float gamma() { return gamma; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        row_length = X2.length();
        return eg.linear2_row(false, X1, X2, alpha, beta, gamma);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads,
            boolean backward_grads1, boolean backward_grads2) 
    {
        if(!backward_grads) return null;
        
        int gc_count = 0; Tensor deltaX1 = null, deltaX2 = null;
        if(backward_grads1) { 
            deltaX1 = eg.linear(false, alpha, deltaY, 0);
            gc_count++;//(1) deltaX1 = deltaY * alpha
        }
        if(backward_grads) {
            deltaX2 = eg.field_linear(deltaY, row_length, beta, 0);
            gc_count++;//(2) deltaX2 = field_sum: deltaY * beta
        }
        
        if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, deltaY is not needed
            CountGc gc = new CountGc(gc_count, deltaY);
            if(deltaX1 != null) deltaX1.dual(()-> { gc.countDown(); });
            if(deltaX2 != null) deltaX2.dual(()-> { gc.countDown(); });
        }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
}
