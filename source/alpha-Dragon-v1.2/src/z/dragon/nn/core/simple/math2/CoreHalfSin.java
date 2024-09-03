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
public class CoreHalfSin<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected float amp;
    protected float alpha;
    protected float beta;
    
    public CoreHalfSin(T unit, boolean inplace, 
            float amp, float alpha, float beta) {
        super(unit, inplace);
        this.amp = amp;
        this.alpha = alpha;
        this.beta = beta;
    }
    
    public final float amp() { return amp; }
    public final float alpha() { return alpha; }
    public final float beta() { return beta; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.halfSin(inplace, amp, alpha, X, beta);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.halfSin_deltaX(grad_inplace, deltaY, holdY(), amp, alpha) :
                null);
    }
    //</editor-fold>
}
