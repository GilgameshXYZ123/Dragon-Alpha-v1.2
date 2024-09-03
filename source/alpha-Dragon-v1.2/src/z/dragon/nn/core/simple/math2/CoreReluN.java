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
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
@Passed("CudaFloat32Base")
public class CoreReluN <T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected float N;
    
    public CoreReluN(T unit, boolean inplace, float N) { 
        super(unit, inplace);
        this.N = N;
    } 
    
    public float N() { return N; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.reluN(inplace, X, N);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY,
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (is_holdY() ?
                eg.reluN_deltaX_v1(grad_inplace, deltaY, holdY(), N) ://V1: Y is not changed
                eg.reluN_deltaX_v2(grad_inplace, deltaY, holdX(), N));//V2: X is not changed
    }
    //</editor-fold>
}
