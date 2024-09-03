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
public class CoreElu<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected float alpha;
    protected float k;
    
    public CoreElu(T unit, boolean inplace, float alpha, float negative_slope) {
        super(unit, inplace);
        this.alpha = alpha;
        this.k = negative_slope;
    }
     
    public final float alpha() { return alpha; }
    public final float negative_slope() { return k; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.elu(inplace, X, alpha, k);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (is_holdY()?
                eg.elu_deltaX_v1(grad_inplace, deltaY, holdY(), alpha, k): //V1: Y is not changed
                eg.elu_deltaX_v2(grad_inplace, deltaY, holdX(), alpha, k));//V2: X is not changed
    }
    //</editor-fold>
}
