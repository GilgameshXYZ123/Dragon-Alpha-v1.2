/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.dual.blas;

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
public class CoreMatMulT1<T extends DualUnit> extends DualCore<T> {
    public CoreMatMulT1(T unit) { super(unit); }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        return eg.matMulT1(X1, X2);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads,
            boolean backward_grads1, boolean backward_grads2) 
    {
        if(!backward_grads) return null;
        
        int count = 0; Tensor deltaX1 = null, deltaX2 = null;
        //(1) deltaX1[K, N] = matMulT2: X2[K, M] * deltaY^T[M, N]
        //(2) deltaX2[K, M] = matMul:   X1[K, N] * deltaY[N, M]
        if(backward_grads1) { deltaX1 = eg.matMulT2(holdX2(), deltaY); count++; }
        if(backward_grads2) { deltaX2 = eg.matMul(holdX1(), deltaY); count++; }
        
        if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, the deltaY is not needed
            CountGc gc = new CountGc(count, deltaY);
            if(deltaX1 != null) deltaX1.dual(()-> { gc.countDown(); });
            if(deltaX2 != null) deltaX2.dual(()-> { gc.countDown(); });
        }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
}
