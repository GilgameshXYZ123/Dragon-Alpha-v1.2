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
public class CoreLeakyRelu<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected float k;
    
    public CoreLeakyRelu(T unit, boolean inplace, float negative_slope) {
        super(unit, inplace);
        this.k = negative_slope;
    }
    
    public float negative_slop() { return k; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.leakyRelu(inplace, X, k);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                eg.leakyRelu_deltaX_v1(grad_inplace, deltaY, holdY(), k): 
                eg.leakyRelu_deltaX_v2(grad_inplace, deltaY, holdX(), k));
    }
    //</editor-fold>
}
