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
public class CoreQuadratic2Center<T extends DualUnit> extends DualCore<T> {
    protected int dim2;
    protected float k11, k12, k22, k1, k2, C; 

    public CoreQuadratic2Center(T unit, int dim2,
            float k11, float k12, float k22,
            float k1, float k2,
            float C)
    {
        super(unit);
        this.dim2 = dim2;
        this.k11 = k11; this.k12 = k12; this.k22 = k22;
        this.k1  = k1; this.k2  = k2; 
        this.C = C;
    }

    public int dim2() { return dim2; }
    public float k11() { return k11; }
    public float k12() { return k12; }
    public float k22() { return k22; }
    public float k1() { return k1; }
    public float k2() { return k2; }
    public float C() { return C; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        return eg.quadratic2_center(false, X1, X2, dim2, k11, k12, k22, k1, k2, C);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads, 
            boolean backward_grads1, boolean backward_grads2) 
    {
        if(!backward_grads) return null;
        
        Tensor X1 = holdX1(), X2 = holdX2();
        if(backward_grads1 && backward_grads2)//(1) deltaX1 = deltaY * (k11*2*X1 + k12*X2 + k1)
            return eg.quadratic2_center_deltaX(grad_inplace, deltaY, X1, X2, dim2, k11, k12, k22, k2, k1, C);
        
        return new Tensor[] {//(2) deltaX2 = center_sum: deltaY * (k22*2*X2 + k12*X1 + k2)
            (backward_grads1 ? eg.quadratic2_center_deltaX1(grad_inplace, deltaY, X1, X2, dim2, k11, k12, k1) : null),
            (backward_grads2 ? eg.quadratic2_center_deltaX2(deltaY, X1, X2, dim2, k22, k12, k2) : null)
        };
    }
    //</editor-fold>
}
