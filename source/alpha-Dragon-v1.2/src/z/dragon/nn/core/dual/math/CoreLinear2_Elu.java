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
 * Linear2 + Elu.
 *  Y = Elu(alpha*X1 + beta*X2 + gamma, k)
 * @author Gilgamesh
 * @param <T>
 */
public class CoreLinear2_Elu<T extends DualUnit> extends DualCore<T> 
{
    protected boolean likeX1;
    protected float alpha, beta, gamma, theta;
    protected float k;

    public CoreLinear2_Elu(T unit, boolean likeX1, 
            float alpha, float beta, float gamma,
            float theta, float negative_slope) 
    {
        super(unit);
        this.likeX1 = likeX1;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.theta = theta;
        this.k = negative_slope;
    }
    
    public final boolean likeX1() { return likeX1; }
    public final float alpha() { return alpha; }
    public final float beta() { return beta; }
    public final float gamma() { return gamma; }
    
    public final float theta() { return theta; }
    public final float negative_slope() { return k; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        return eg.linear2_elu(false, likeX1, X1, X2, 
                alpha, beta, gamma, theta, k);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads,
            boolean backward_grads1, boolean backward_grads2) {
        if(!backward_grads) return null;
        
        Tensor[] grads = (is_holdY()? //V1: holdY / V2: hold{X1, X2}
                eg.linear2_elu_deltaX_v1(grad_inplace, deltaY, holdY(), alpha, beta, theta, k) :
                eg.linear2_elu_deltaX_v2(grad_inplace, deltaY, holdX1(), holdX2(), alpha, beta, gamma, theta, k));
        
        if(!backward_grads1) { grads[0].remote_delete(); grads[0] = null; }//deltaX1
        if(!backward_grads2) { grads[1].remote_delete(); grads[1] = null; }//deltaX2
        return grads;//{deltaX1, deltaX2}
    }
    //</editor-fold>
}
