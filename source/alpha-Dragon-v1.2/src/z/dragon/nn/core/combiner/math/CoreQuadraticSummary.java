/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.combiner.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.combiner.CombinerCore;
import z.dragon.nn.unit.combiner.Combiner;

/**
 * QuadraticSummary.
 * (1) Y = sum(alpha*X^2 + beta*X + gamma
 * (2) deltaX = 2*alpha*(deltaY*X) + beta*deltaY
 * @author Gilgamesh
 * @param <T>
 */
public class CoreQuadraticSummary<T extends Combiner> extends CombinerCore<T> {
    protected float alpha;
    protected float beta;
    protected float gamma;

    public CoreQuadraticSummary(T unit, float alpha, float beta, float gamma) {
        super(unit);
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    public final float alpha() { return alpha; }
    public final float beta() { return beta; }
    public final float gamma() { return gamma; }

    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor[] X) {
        return eg.quadratic_sum(false, alpha, beta, gamma, X);
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            int input_tensor_num, boolean grad_inplace, 
            boolean backward_grads, boolean[] last_need_grads)
    {
        if(!backward_grads) return null;//grads_num = 0;
        
        Tensor[] deltaX = new Tensor[input_tensor_num];
        int[] backward_grads_index = backward_grads_index(last_need_grads);
        int grads_num = backward_grads_index.length;
        
        float alpha2 = 2.0f * alpha;
        if(grads_num == 1) {
            int idx = backward_grads_index[0];
            deltaX[idx] = eg.quadratic2(grad_inplace, deltaY, holdX(idx), 
                    0.0f, alpha2, 0, beta, 0, 0);
            return deltaX;
        }
        
        for(int i=0; i<grads_num; i++) {
            int idx = backward_grads_index[i];
            deltaX[idx] = eg.quadratic2(false, deltaY, holdX(idx), 
                    0.0f, alpha2, 0, 
                    beta, 0, 0);
        }
        
        if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
            CountGc gc = new CountGc(grads_num, deltaY);
            for (Tensor grad : deltaX) grad.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }
    //</editor-fold>
}
