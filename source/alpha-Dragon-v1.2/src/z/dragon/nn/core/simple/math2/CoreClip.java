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
public class CoreClip<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    protected float alpha;
    protected float beta;
    protected float vmin;
    protected float vmax;
            
    public CoreClip(T unit, boolean inplace, float alpha, float beta, float vmin, float vmax) {
        super(unit, inplace);
        this.alpha = alpha;
        this.beta = beta;
        this.vmin = vmin;
        this.vmax = vmax;
    }

    public float alpha() { return alpha; }
    public float beta() { return beta; }
    public float vmin() { return vmin; }
    public float vmax() { return vmax; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.clip(inplace, alpha, X, beta, vmin, vmax);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (is_holdY()? 
                eg.clip_deltaX_v1(grad_inplace, deltaY, holdY(), alpha, vmin, vmax) ://V1: Y is not changed
                eg.clip_deltaX_v2(grad_inplace, deltaY, holdX(), alpha, beta, vmin, vmax));//V2: X is not changed
    }
    //</editor-fold>
}
