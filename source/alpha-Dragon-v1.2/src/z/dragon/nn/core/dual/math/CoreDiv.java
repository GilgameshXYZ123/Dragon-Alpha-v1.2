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
 * Division.
 * (1) Y = (a1*X1 + b1)/(a2*X2 + b2) + gamma
 * (2) deltaX1 = deltaY * a1 / (a2*X2 + b2)
 * (3) deltaX2 = deltaY * -a2 * (a1*X1 + b1)/{(a2*X2 + b2)^2}  
 * @author Gilgamesh
 * @param <T>
 */
public class CoreDiv<T extends DualUnit> extends DualCore<T> {
    protected boolean likeX1;
    protected float alpha1, beta1;
    protected float alpha2, beta2;
    protected float gamma;
    
    public CoreDiv(T unit, boolean likeX1,
            float alpha1, float beta1, 
            float alpha2, float beta2,
            float gamma) 
    {
        super(unit);
        this.likeX1 = likeX1;
        this.alpha1 = alpha1; this.beta1 = beta1;
        this.alpha2 = alpha2; this.beta2 = beta2;
        this.gamma = gamma;
    }
    
    public final boolean likeX1() { return likeX1; }
    public final float alpha1() { return alpha1; }
    public final float alpha2() { return alpha2; }
    public final float beta1() { return beta1; }
    public final float beta2() { return beta2; }
    public final float gamma() { return gamma; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X1, Tensor X2) {
        return eg.div(false, likeX1,//inplace = false
                alpha1, X1, beta1,
                alpha2, X2, beta2,
                gamma);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads, 
            boolean backward_grads1, boolean backward_grads2) 
    {
        if(!backward_grads) return null;
         
        int gc_count = 0; Tensor deltaX1 = null, deltaX2 = null;
        if(backward_grads1 && backward_grads2) {
            Tensor[] deltaX = eg.div_deltaX(false, deltaY, 
                        holdX1(), alpha1, beta1, 
                        holdX2(), alpha2, beta2);
            deltaX1 = deltaX[0]; deltaX2 = deltaX[1];
            gc_count += 2;
        }
        else if(backward_grads1) {//backward_grads2 = false
            deltaX1 = eg.div_deltaX1(grad_inplace, deltaY, 
                    holdX2(), alpha1, alpha2, beta2);
            gc_count++;//(1) deltaX1 = deltaY * a1 / (a2*X2 + b2)
        }
        else {//backward_grads1 = false, backward_grads2 = true
            deltaX2 = eg.div_deltaX2(grad_inplace, deltaY, 
                    holdX1(), alpha1, beta1, 
                    holdX2(), alpha2, beta2);
            gc_count++;//(2) deltaX2 = deltaY * -a2 * (a1*X1 + b1)/{(a2*X2 + b2)^2}  
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
